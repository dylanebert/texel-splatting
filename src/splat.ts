import { BVH_WGSL } from "./bvh";
import {
    type SceneConfig,
    FACE_UNIFORMS_WGSL,
    LIGHTING_WGSL,
    POINT_LIGHT_WGSL,
    SLICE_TARGETS,
    STONE_COLOR_WGSL,
    computeFaceMask,
    cubePerspective,
    extractFrustumPlanes,
    aabbInFrustum,
} from "./lighting";
import { lookAt, multiply } from "./math";
import { OKLAB_WGSL, PATH_WGSL } from "./oklab";
import {
    SKY_STRUCT_WGSL,
    SKY_SCENE_STRUCT_WGSL,
    NOISE_WGSL,
    STARS_WGSL,
    MOON_WGSL,
    CLOUDS_WGSL,
    SAMPLE_SKY_WGSL,
    HAZE_WGSL,
    COMPUTE_SKY_DIR_WGSL,
} from "./sky";

const PROBE_SIZE = 384;
const NEAR = 0.1;
const FAR = 200;
const GRID_STEP = 1.0;
const BLEND_DURATION = 0.5;
const EXPANSION = 0.5;
const PROBE_EYE = 0;
const PROBE_GRID = 1;
const PROBE_PREV = 2;
const PROBE_LAYERS = 18;
const FACE_TEXELS = PROBE_SIZE * PROBE_SIZE;

const DEG = Math.PI / 180;
const GRID_CULL_COS = Math.cos(103 * DEG);
const EYE_CULL_COS = Math.cos(98 * DEG);


const PROBE_PARAMS_WGSL = /* wgsl */ `
struct ProbeParams {
    origins: array<vec4<f32>, 3>,
    ranges: array<vec4<f32>, 3>,
    masks: vec4<f32>,
    state: vec4<f32>,
}`;

interface TransitionState {
    origin: [number, number, number];
    prevOrigin: [number, number, number];
    fadeT: number;
    smoothedSpeed: number;
    lastCameraPos: [number, number, number];
    blending: boolean;
    lastTime: number;
}

function createTransitionState(): TransitionState {
    return {
        origin: [NaN, NaN, NaN],
        prevOrigin: [0, 0, 0],
        fadeT: 0,
        smoothedSpeed: 0,
        lastCameraPos: [NaN, NaN, NaN],
        blending: false,
        lastTime: 0,
    };
}

function updateTransition(
    ts: TransitionState,
    eyeX: number,
    eyeY: number,
    eyeZ: number,
    currentTime: number,
): { crossfading: boolean; fadeT: number; triggered: boolean } {
    const dt = Math.min(currentTime - ts.lastTime, 0.1);
    ts.lastTime = currentTime;

    const hasPrev = !Number.isNaN(ts.lastCameraPos[0]);
    if (hasPrev && dt > 0) {
        const dx = eyeX - ts.lastCameraPos[0];
        const dy = eyeY - ts.lastCameraPos[1];
        const dz = eyeZ - ts.lastCameraPos[2];
        const instantSpeed = Math.sqrt(dx * dx + dy * dy + dz * dz) / dt;
        const alpha = 1 - Math.exp(-dt / 0.05);
        ts.smoothedSpeed = alpha * instantSpeed + (1 - alpha) * ts.smoothedSpeed;
    }
    ts.lastCameraPos[0] = eyeX;
    ts.lastCameraPos[1] = eyeY;
    ts.lastCameraPos[2] = eyeZ;

    const snap = (v: number) => Math.round(v / GRID_STEP) * GRID_STEP;

    if (Number.isNaN(ts.origin[0])) {
        ts.origin[0] = snap(eyeX);
        ts.origin[1] = snap(eyeY);
        ts.origin[2] = snap(eyeZ);
        return { crossfading: false, fadeT: 0, triggered: false };
    }

    let triggered = false;
    if (!ts.blending) {
        const sx = snap(eyeX);
        const sy = snap(eyeY);
        const sz = snap(eyeZ);
        if (sx !== ts.origin[0] || sy !== ts.origin[1] || sz !== ts.origin[2]) {
            ts.prevOrigin[0] = ts.origin[0];
            ts.prevOrigin[1] = ts.origin[1];
            ts.prevOrigin[2] = ts.origin[2];
            ts.origin[0] = sx;
            ts.origin[1] = sy;
            ts.origin[2] = sz;
            ts.fadeT = 0;
            ts.blending = true;
            triggered = true;
        }
    } else {
        const baseRate = 1 / BLEND_DURATION;
        const velocityRate = ts.smoothedSpeed / GRID_STEP;
        ts.fadeT += Math.max(baseRate, velocityRate) * dt;
        if (ts.fadeT >= 1) {
            ts.fadeT = 1;
            ts.blending = false;
        }
    }

    return { crossfading: ts.blending, fadeT: ts.fadeT, triggered };
}

export interface SplatConfig extends SceneConfig {
    nodeBuffer: GPUBuffer;
    triBuffer: GPUBuffer;
    triIdBuffer: GPUBuffer;
    lightBuffer: GPUBuffer;
    skyBuffer: GPUBuffer;
    sceneBuffer: GPUBuffer;
}

export interface SplatEncoder {
    encode(
        encoder: GPUCommandEncoder,
        params: {
            cameraPos: [number, number, number];
            cameraFwd: [number, number, number];
            sunDir: [number, number, number];
            sunColor: [number, number, number];
            ambient: [number, number, number];
            shadowFade: number;
            pointLightCount: number;
            time: number;
        },
        colorView: GPUTextureView,
        depthView: GPUTextureView,
        viewProj: Float32Array,
    ): void;
    destroy(): void;
}

export function createSplat(config: SplatConfig): SplatEncoder {
    const { device } = config;

    // 18-layer textures (eye=0-5, grid=6-11, prev=12-17)
    const probeAlbedo = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, PROBE_LAYERS],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const probeNormal = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, PROBE_LAYERS],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const probeRadial = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, PROBE_LAYERS],
        format: "r32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const probeEid = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, PROBE_LAYERS],
        format: "r32uint",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });
    const probeDepth = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, PROBE_LAYERS],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        dimension: "2d",
    });
    const probeLit = device.createTexture({
        size: [PROBE_SIZE, PROBE_SIZE, PROBE_LAYERS],
        format: "rgba8unorm",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        dimension: "2d",
    });

    const faceViews = {
        albedo: Array.from({ length: PROBE_LAYERS }, (_, i) =>
            probeAlbedo.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        normal: Array.from({ length: PROBE_LAYERS }, (_, i) =>
            probeNormal.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        radial: Array.from({ length: PROBE_LAYERS }, (_, i) =>
            probeRadial.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        eid: Array.from({ length: PROBE_LAYERS }, (_, i) =>
            probeEid.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
        depth: Array.from({ length: PROBE_LAYERS }, (_, i) =>
            probeDepth.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 }),
        ),
    };

    const arrayViews = {
        albedo: probeAlbedo.createView({ dimension: "2d-array" }),
        normal: probeNormal.createView({ dimension: "2d-array" }),
        radial: probeRadial.createView({ dimension: "2d-array" }),
        eid: probeEid.createView({ dimension: "2d-array" }),
        lit: probeLit.createView({ dimension: "2d-array" }),
    };

    // One large uniform buffer for all face uniforms, accessed via dynamic offsets.
    // Stride must be a multiple of minUniformBufferOffsetAlignment (typically 256 B).
    const FACE_UNIFORM_BYTES = 96; // 24 floats × 4 bytes
    const FACE_UNIFORM_ALIGN = device.limits.minUniformBufferOffsetAlignment;
    const FACE_UNIFORM_STRIDE = Math.ceil(FACE_UNIFORM_BYTES / FACE_UNIFORM_ALIGN) * FACE_UNIFORM_ALIGN;
    const FACE_UNIFORM_STRIDE_F32 = FACE_UNIFORM_STRIDE >> 2;
    const faceUniformBuffer = device.createBuffer({
        size: PROBE_LAYERS * FACE_UNIFORM_STRIDE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const probeParamsBuffer = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const indirectArgsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const visibilityBuffer = device.createBuffer({
        size: PROBE_LAYERS * FACE_TEXELS * 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const splatSceneBuffer = device.createBuffer({
        size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const edgeMaskBuffer = device.createBuffer({
        size: FACE_TEXELS * PROBE_LAYERS * 4,
        usage: GPUBufferUsage.STORAGE,
    });

    // Compact list of active layer indices uploaded once per frame; lighting dispatch uses
    // only as many Z-slices as there are genuinely active (probe-active AND face-masked) layers.
    const activeLayerIndexBuffer = device.createBuffer({
        size: PROBE_LAYERS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });


    const OCT_ENCODE_WGSL = /* wgsl */ `
fn octEncode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    let s = select(vec2(-1.0), vec2(1.0), p >= vec2(0.0));
    let q = select((1.0 - abs(p.yx)) * s, p, n.z >= 0.0);
    return q * 0.5 + 0.5;
}`;

    // --- G-buffer pipelines (4 MRT: albedo, normal, radial, eid) ---

    const grassGbufferShader = device.createShaderModule({
        code: /* wgsl */ `
            ${FACE_UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> face: FaceUniforms;

            ${OKLAB_WGSL}
            ${PATH_WGSL}
            ${OCT_ENCODE_WGSL}

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) worldPos: vec3f,
                @location(1) localY: f32,
            }

            @vertex fn vs(@location(0) position: vec3f) -> VsOut {
                let world = vec3(
                    position.x * ${config.area}.0,
                    position.y * ${config.height},
                    position.z * ${config.area}.0,
                );
                var out: VsOut;
                out.pos = face.viewProj * vec4f(world, 1);
                out.worldPos = world;
                out.localY = position.y;
                return out;
            }

            struct GbufferOut {
                @location(0) albedo: vec4f,
                @location(1) normal: vec4f,
                @location(2) radial: f32,
                @location(3) eid: u32,
            }

            @fragment fn fs(in: VsOut) -> GbufferOut {
                let t = clamp(in.worldPos.y / ${config.height}, 0.0, 1.0);
                let wp = in.worldPos.xz;
                let h = hash2(floor(wp * ${config.density}.0));
                if (h < t) { discard; }
                if (t > 0.0 && pathGrassDiscard(wp)) { discard; }

                let base = vec3f(${config.baseR}, ${config.baseG}, ${config.baseB});
                let oklab = toOKLab(base);
                let hueVar = hash2(floor(wp * ${config.hueFreq}));
                let l = mix(oklab.x * ${config.rootL}, oklab.x * ${config.tipL}, t) + (hueVar - 0.5) * ${config.hueVar};
                let a = oklab.y + (hueVar - 0.5) * ${config.hueVar};
                let b = mix(oklab.z - 0.01, oklab.z + 0.01, t) + (hueVar - 0.5) * ${config.hueVar};
                var color = fromOKLab(vec3(l, a, b));
                if (t == 0.0) { color = pathGroundColor(wp, color); }

                let dx = in.worldPos.x - face.origin.x;
                let dy = in.worldPos.y - face.origin.y;
                let dz = in.worldPos.z - face.origin.z;
                let chebyshev = max(abs(dx), max(abs(dy), abs(dz)));
                let radial = (chebyshev - face.near) / (face.far - face.near);

                var out: GbufferOut;
                out.albedo = vec4f(color, 1.0);
                out.normal = vec4f(octEncode(vec3f(0.0, 1.0, 0.0)), 0.0, 0.0);
                out.radial = radial;
                out.eid = 1u;
                return out;
            }
        `,
    });

    const stoneGbufferShader = device.createShaderModule({
        code: /* wgsl */ `
            ${FACE_UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> face: FaceUniforms;

            ${OKLAB_WGSL}
            ${STONE_COLOR_WGSL}
            ${OCT_ENCODE_WGSL}

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) normal: vec3f,
                @location(1) worldPos: vec3f,
            }

            @vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f) -> VsOut {
                var out: VsOut;
                out.pos = face.viewProj * vec4f(position, 1);
                out.normal = normal;
                out.worldPos = position;
                return out;
            }

            struct GbufferOut {
                @location(0) albedo: vec4f,
                @location(1) normal: vec4f,
                @location(2) radial: f32,
                @location(3) eid: u32,
            }

            @fragment fn fs(in: VsOut) -> GbufferOut {
                let color = stoneColor(in.worldPos);
                let n = normalize(in.normal);
                let dx = in.worldPos.x - face.origin.x;
                let dy = in.worldPos.y - face.origin.y;
                let dz = in.worldPos.z - face.origin.z;
                let chebyshev = max(abs(dx), max(abs(dy), abs(dz)));
                let radial = (chebyshev - face.near) / (face.far - face.near);

                var out: GbufferOut;
                out.albedo = vec4f(color, 1.0);
                out.normal = vec4f(octEncode(n), 0.0, 0.0);
                out.radial = radial;
                out.eid = 2u;
                return out;
            }
        `,
    });

    const gbufferLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform", hasDynamicOffset: true },
            },
        ],
    });
    const gbufferPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [gbufferLayout],
    });

    const grassGbufferPipeline = device.createRenderPipeline({
        layout: gbufferPipelineLayout,
        vertex: {
            module: grassGbufferShader,
            buffers: [
                {
                    arrayStride: 12,
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
                },
            ],
        },
        fragment: {
            module: grassGbufferShader,
            targets: [
                { format: "rgba8unorm" },
                { format: "rgba8unorm" },
                { format: "r32float" },
                { format: "r32uint" },
            ],
        },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const stoneGbufferPipeline = device.createRenderPipeline({
        layout: gbufferPipelineLayout,
        vertex: {
            module: stoneGbufferShader,
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" },
                        { shaderLocation: 1, offset: 12, format: "float32x3" },
                    ],
                },
            ],
        },
        fragment: {
            module: stoneGbufferShader,
            targets: [
                { format: "rgba8unorm" },
                { format: "rgba8unorm" },
                { format: "r32float" },
                { format: "r32uint" },
            ],
        },
        primitive: { topology: "triangle-list", cullMode: "back", frontFace: "cw" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    // Single bind group for all face uniforms — offset supplied as dynamic offset at draw time
    const faceBindGroup = device.createBindGroup({
        layout: gbufferLayout,
        entries: [{ binding: 0, resource: { buffer: faceUniformBuffer, size: FACE_UNIFORM_BYTES } }],
    });

    // Emissive G-buffer (orb + wisps)
    const orbLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
        ],
    });

    const emissiveGbufferShader = device.createShaderModule({
        code: /* wgsl */ `
            ${FACE_UNIFORMS_WGSL}
            @group(0) @binding(0) var<uniform> face: FaceUniforms;

            struct OrbParams {
                position: vec3f,
                scale: f32,
                color: vec3f,
                _pad: f32,
            }
            @group(1) @binding(0) var<uniform> orb: OrbParams;

            struct VsOut {
                @builtin(position) pos: vec4f,
                @location(0) worldPos: vec3f,
                @location(1) color: vec3f,
            }

            @vertex fn vs(@location(0) position: vec3f) -> VsOut {
                let world = position * orb.scale + orb.position;
                var out: VsOut;
                out.pos = face.viewProj * vec4f(world, 1);
                out.worldPos = world;
                out.color = orb.color;
                return out;
            }

            struct GbufferOut {
                @location(0) albedo: vec4f,
                @location(1) normal: vec4f,
                @location(2) radial: f32,
                @location(3) eid: u32,
            }

            ${OCT_ENCODE_WGSL}

            @fragment fn fs(in: VsOut) -> GbufferOut {
                let dx = in.worldPos.x - face.origin.x;
                let dy = in.worldPos.y - face.origin.y;
                let dz = in.worldPos.z - face.origin.z;
                let chebyshev = max(abs(dx), max(abs(dy), abs(dz)));
                let radial = (chebyshev - face.near) / (face.far - face.near);

                var out: GbufferOut;
                out.albedo = vec4f(min(in.color, vec3f(1)), 0.0);
                out.normal = vec4f(octEncode(vec3f(0.0, 1.0, 0.0)), 0.0, 1.0);
                out.radial = radial;
                out.eid = 3u;
                return out;
            }
        `,
    });

    const emissiveGbufferPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({
            bindGroupLayouts: [gbufferLayout, orbLayout],
        }),
        vertex: {
            module: emissiveGbufferShader,
            buffers: [
                {
                    arrayStride: 24,
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
                },
            ],
        },
        fragment: {
            module: emissiveGbufferShader,
            targets: [
                { format: "rgba8unorm" },
                { format: "rgba8unorm" },
                { format: "r32float" },
                { format: "r32uint" },
            ],
        },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const orbBindGroups = config.orbBuffers.map((buf) =>
        device.createBindGroup({
            layout: orbLayout,
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    );

    // --- Lighting compute (18 layers) ---

    const lightingParamsBuffer = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const LIGHTING_PARAMS_WGSL = /* wgsl */ `
struct LightingParams {
    origin0: vec3f, near0: f32,
    sunDir: vec3f, far0: f32,
    sunColor: vec3f, shadowFade: f32,
    ambient: vec3f, faceMask0: u32,
    origin1: vec3f, faceMask1: u32,
    origin2: vec3f, faceMask2: u32,
    activeProbes: u32, pointLightCount: u32, _pad1: u32, _pad2: u32,
}`;

    const lightingShader = device.createShaderModule({
        code: /* wgsl */ `
            @group(0) @binding(0) var cubeAlbedoTex: texture_2d_array<f32>;
            @group(0) @binding(1) var cubeNormalTex: texture_2d_array<f32>;
            @group(0) @binding(2) var cubeRadialTex: texture_2d_array<f32>;
            @group(0) @binding(3) var cubeLitTex: texture_storage_2d_array<rgba8unorm, write>;

            ${LIGHTING_PARAMS_WGSL}
            @group(0) @binding(4) var<uniform> lp: LightingParams;

            ${POINT_LIGHT_WGSL}
            @group(0) @binding(5) var<uniform> pointLights: array<PointLight, 64>;

            @group(0) @binding(6) var<storage, read> bvhNodes: array<BVHNode>;
            @group(0) @binding(7) var<storage, read> bvhTris: array<BVHTri>;
            @group(0) @binding(8) var<storage, read> bvhTriIds: array<u32>;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(9) var<uniform> sky: Sky;

            ${SKY_SCENE_STRUCT_WGSL}
            @group(0) @binding(10) var<uniform> scene: SkyScene;

            @group(0) @binding(11) var probeEidTex: texture_2d_array<u32>;
            @group(0) @binding(12) var<storage, read> activeLayerIndices: array<u32>;

            ${BVH_WGSL}
            ${OKLAB_WGSL}
            ${NOISE_WGSL}
            ${STARS_WGSL}
            ${MOON_WGSL}
            ${CLOUDS_WGSL}
            ${SAMPLE_SKY_WGSL}
            ${LIGHTING_WGSL}

            fn decodeNormal(e: vec2<f32>) -> vec3<f32> {
                let f = e * 2.0 - 1.0;
                var n = vec3(f, 1.0 - abs(f.x) - abs(f.y));
                if (n.z < 0.0) {
                    let s = select(vec2(-1.0), vec2(1.0), n.xy >= vec2(0.0));
                    n = vec3((1.0 - abs(n.yx)) * s, n.z);
                }
                return normalize(n);
            }

            fn faceUVtoDir(face: u32, u: f32, v: f32) -> vec3f {
                let uv = vec2f(u * 2.0 - 1.0, v * 2.0 - 1.0);
                switch (face) {
                    case 0u: { return normalize(vec3f( 1.0, -uv.y, -uv.x)); }
                    case 1u: { return normalize(vec3f(-1.0, -uv.y,  uv.x)); }
                    case 2u: { return normalize(vec3f( uv.x,  1.0,  uv.y)); }
                    case 3u: { return normalize(vec3f( uv.x, -1.0, -uv.y)); }
                    case 4u: { return normalize(vec3f( uv.x, -uv.y,  1.0)); }
                    default: { return normalize(vec3f(-uv.x, -uv.y, -1.0)); }
                }
            }

            const OUTLINE_NORMAL_THRESH: f32 = 0.7;
            const OUTLINE_DARKEN: f32 = 1.0;
            const OUTLINE_HIGHLIGHT: f32 = 1.0;
            const BANDS: f32 = 32.0;

            fn detectEdge(coord: vec2u, layer: u32, size: u32, centerRadial: f32, viewDir: vec3<f32>) -> i32 {
                let x = coord.x;
                let y = coord.y;
                let centerEid = textureLoad(probeEidTex, coord, layer, 0).r;
                let centerNormal = decodeNormal(textureLoad(cubeNormalTex, vec2i(coord), layer, 0).rg);

                let offsets = array<vec2<i32>, 4>(vec2(1, 0), vec2(-1, 0), vec2(0, 1), vec2(0, -1));

                for (var i = 0u; i < 4u; i++) {
                    let nx = i32(x) + offsets[i].x;
                    let ny = i32(y) + offsets[i].y;
                    if (nx < 0 || ny < 0 || nx >= i32(size) || ny >= i32(size)) { continue; }
                    let nc = vec2u(u32(nx), u32(ny));
                    let nEid = textureLoad(probeEidTex, nc, layer, 0).r;
                    if (nEid != 0u && centerEid != 0u && nEid != centerEid) {
                        let nRadial = textureLoad(cubeRadialTex, vec2i(nc), layer, 0).r;
                        if (centerRadial <= nRadial) { return 1; }
                        continue;
                    }
                    let nNormal = decodeNormal(textureLoad(cubeNormalTex, vec2i(nc), layer, 0).rg);
                    if (dot(centerNormal, nNormal) < OUTLINE_NORMAL_THRESH) {
                        if (dot(centerNormal, viewDir) > dot(nNormal, viewDir)) { return 2; }
                        return 1;
                    }
                }
                return 0;
            }

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) gid: vec3u) {
                let size = ${PROBE_SIZE}u;
                if (gid.x >= size || gid.y >= size) { return; }

                // gid.z is a compact index into the pre-built active-layer list.
                // Only layers where the probe is active AND the face is in the mask are included,
                // so no early-return branches or sky-fill work for masked faces are needed here.
                let layer = activeLayerIndices[gid.z];
                let probeIdx = layer / 6u;
                let face = layer % 6u;

                var origin: vec3f;
                switch (probeIdx) {
                    case 0u: { origin = lp.origin0; }
                    case 1u: { origin = lp.origin1; }
                    default: { origin = lp.origin2; }
                }

                let coords = vec2i(vec2u(gid.x, gid.y));
                let radial = textureLoad(cubeRadialTex, coords, layer, 0).r;

                let uv_u = (f32(gid.x) + 0.5) / f32(size);
                let uv_v = (f32(gid.y) + 0.5) / f32(size);
                let dir = faceUVtoDir(face, uv_u, uv_v);

                if (radial >= 0.999) {
                    let skyColor = posterize(sampleSky(dir));
                    textureStore(cubeLitTex, vec2u(gid.x, gid.y), layer, vec4f(skyColor, 1.0));
                    return;
                }

                let albedoSample = textureLoad(cubeAlbedoTex, coords, layer, 0);
                if (albedoSample.a < 0.5) {
                    textureStore(cubeLitTex, vec2u(gid.x, gid.y), layer, vec4f(posterize(albedoSample.rgb), 1.0));
                    return;
                }
                let albedo = albedoSample.rgb;
                let normalSample = textureLoad(cubeNormalTex, coords, layer, 0);
                let normal = decodeNormal(normalSample.rg);
                let emissive = normalSample.a;

                let chebyshev = radial * (lp.far0 - lp.near0) + lp.near0;
                let absDir = abs(dir);
                let maxComp = max(absDir.x, max(absDir.y, absDir.z));
                let worldPos = origin + dir * (chebyshev / maxComp);

                var color = computeLighting(worldPos, normal, albedo, lp.sunDir, lp.shadowFade, lp.sunColor, lp.ambient, emissive, lp.pointLightCount);

                if (probeIdx >= 1u) {
                    let viewDir = normalize(origin - worldPos);
                    let edge = detectEdge(vec2u(gid.x, gid.y), layer, size, radial, viewDir);
                    if (edge != 0) {
                        let bandSize = 1.0 / BANDS;
                        var lab = toOKLab(color);
                        lab.x += select(OUTLINE_HIGHLIGHT, -OUTLINE_DARKEN, edge == 1) * bandSize;
                        lab.x = clamp(lab.x, 0.0, 1.0);
                        color = max(fromOKLab(lab), vec3(0.0));
                    }
                }
                let posterized = posterize(color);
                textureStore(cubeLitTex, vec2u(gid.x, gid.y), layer, vec4f(posterized, 1.0));
            }
        `,
    });

    const lightingBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                    access: "write-only",
                    format: "rgba8unorm",
                    viewDimension: "2d-array",
                },
            },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 8,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            {
                binding: 11,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "uint", viewDimension: "2d-array" },
            },
            {
                binding: 12,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
        ],
    });

    const lightingPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [lightingBindGroupLayout] }),
        compute: { module: lightingShader },
    });

    const lightingBindGroup = device.createBindGroup({
        layout: lightingBindGroupLayout,
        entries: [
            { binding: 0, resource: arrayViews.albedo },
            { binding: 1, resource: arrayViews.normal },
            { binding: 2, resource: arrayViews.radial },
            { binding: 3, resource: arrayViews.lit },
            { binding: 4, resource: { buffer: lightingParamsBuffer } },
            { binding: 5, resource: { buffer: config.lightBuffer } },
            { binding: 6, resource: { buffer: config.nodeBuffer } },
            { binding: 7, resource: { buffer: config.triBuffer } },
            { binding: 8, resource: { buffer: config.triIdBuffer } },
            { binding: 9, resource: { buffer: config.skyBuffer } },
            { binding: 10, resource: { buffer: config.sceneBuffer } },
            { binding: 11, resource: arrayViews.eid },
            { binding: 12, resource: { buffer: activeLayerIndexBuffer } },
        ],
    });

    // --- Cull compute ---

    const cullShader = device.createShaderModule({
        code: /* wgsl */ `
            ${PROBE_PARAMS_WGSL}

            @group(0) @binding(0) var probeRadial: texture_2d_array<f32>;
            @group(0) @binding(1) var<storage, read_write> vis: array<u32>;
            @group(0) @binding(2) var<storage, read_write> indirectArgs: array<atomic<u32>>;
            @group(0) @binding(3) var<uniform> params: ProbeParams;

            const FACE_TEXELS: u32 = ${FACE_TEXELS}u;
            const PROBE_SIZE_C: u32 = ${PROBE_SIZE}u;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let texelIdx = gid.x;
                let slot = gid.y;
                if (texelIdx >= FACE_TEXELS) { return; }

                let probeIdx = slot / 6u;
                let face = slot % 6u;

                let eyeMask = u32(params.masks.x);
                let gridMask = u32(params.masks.y);
                let prevMask = u32(params.masks.z);
                var mask = gridMask;
                if (probeIdx == ${PROBE_EYE}u) { mask = eyeMask; }
                else if (probeIdx == ${PROBE_PREV}u) { mask = prevMask; }
                if ((mask & (1u << face)) == 0u) { return; }

                let px = texelIdx % PROBE_SIZE_C;
                let py = texelIdx / PROBE_SIZE_C;
                let radial = textureLoad(probeRadial, vec2<u32>(px, py), slot, 0).r;
                if (radial >= 0.999) { return; }

                let idx = atomicAdd(&indirectArgs[1u], 1u);
                vis[idx] = slot * FACE_TEXELS + texelIdx;
            }
        `,
    });

    const cullBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
            },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const cullPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [cullBindGroupLayout] }),
        compute: { module: cullShader },
    });

    const cullBindGroup = device.createBindGroup({
        layout: cullBindGroupLayout,
        entries: [
            { binding: 0, resource: arrayViews.radial },
            { binding: 1, resource: { buffer: visibilityBuffer } },
            { binding: 2, resource: { buffer: indirectArgsBuffer } },
            { binding: 3, resource: { buffer: probeParamsBuffer } },
        ],
    });

    // --- Edge mask compute ---

    const edgeMaskShader = device.createShaderModule({
        code: /* wgsl */ `
            @group(0) @binding(0) var probeRadialTex: texture_2d_array<f32>;
            @group(0) @binding(1) var<storage, read_write> edgeMasks: array<u32>;

            ${LIGHTING_PARAMS_WGSL}
            @group(0) @binding(2) var<uniform> lp: LightingParams;

            const SIZE: u32 = ${PROBE_SIZE}u;
            const FACE_TEXELS: u32 = ${FACE_TEXELS}u;
            const THRESHOLD: f32 = 0.002;

            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let px = gid.x;
                let py = gid.y;
                let layer = gid.z;
                if (px >= SIZE || py >= SIZE) { return; }

                let probeIdx = layer / 6u;
                if ((lp.activeProbes & (1u << probeIdx)) == 0u) { return; }

                let radial = textureLoad(probeRadialTex, vec2<u32>(px, py), layer, 0).r;
                if (radial >= 0.999) {
                    edgeMasks[layer * FACE_TEXELS + py * SIZE + px] = 0u;
                    return;
                }

                var mask = 15u;

                if (px > 0u) {
                    let n = textureLoad(probeRadialTex, vec2<u32>(px - 1u, py), layer, 0).r;
                    if (n >= 0.999 || abs(radial - n) / max(radial, n) >= THRESHOLD) {
                        mask &= ~1u;
                    }
                }
                if (px < SIZE - 1u) {
                    let n = textureLoad(probeRadialTex, vec2<u32>(px + 1u, py), layer, 0).r;
                    if (n >= 0.999 || abs(radial - n) / max(radial, n) >= THRESHOLD) {
                        mask &= ~2u;
                    }
                }
                if (py > 0u) {
                    let n = textureLoad(probeRadialTex, vec2<u32>(px, py - 1u), layer, 0).r;
                    if (n >= 0.999 || abs(radial - n) / max(radial, n) >= THRESHOLD) {
                        mask &= ~4u;
                    }
                }
                if (py < SIZE - 1u) {
                    let n = textureLoad(probeRadialTex, vec2<u32>(px, py + 1u), layer, 0).r;
                    if (n >= 0.999 || abs(radial - n) / max(radial, n) >= THRESHOLD) {
                        mask &= ~8u;
                    }
                }

                edgeMasks[layer * FACE_TEXELS + py * SIZE + px] = mask;
            }
        `,
    });

    const edgeMaskBindGroupLayout = device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
            },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
    });

    const edgeMaskPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [edgeMaskBindGroupLayout] }),
        compute: { module: edgeMaskShader },
    });

    const edgeMaskBindGroup = device.createBindGroup({
        layout: edgeMaskBindGroupLayout,
        entries: [
            { binding: 0, resource: arrayViews.radial },
            { binding: 1, resource: { buffer: edgeMaskBuffer } },
            { binding: 2, resource: { buffer: lightingParamsBuffer } },
        ],
    });

    // --- Display (background + splat) with shared bind group layout ---

    const DIR_TO_FACE_UV_UINT_WGSL = /* wgsl */ `
fn dirToFaceUV(dir: vec3<f32>, size: u32) -> vec3<u32> {
    let a = abs(dir);
    var face: u32; var u: f32; var v: f32; var ma: f32;
    if (a.x >= a.y && a.x >= a.z) {
        ma = a.x;
        if (dir.x > 0.0) { face = 0u; u = -dir.z; v = -dir.y; }
        else              { face = 1u; u =  dir.z; v = -dir.y; }
    } else if (a.y >= a.x && a.y >= a.z) {
        ma = a.y;
        if (dir.y > 0.0) { face = 2u; u = dir.x; v =  dir.z; }
        else              { face = 3u; u = dir.x; v = -dir.z; }
    } else {
        ma = a.z;
        if (dir.z > 0.0) { face = 4u; u =  dir.x; v = -dir.y; }
        else              { face = 5u; u = -dir.x; v = -dir.y; }
    }
    let s = f32(size);
    let px = clamp(u32((u / ma + 1.0) * 0.5 * s), 0u, size - 1u);
    let py = clamp(u32((v / ma + 1.0) * 0.5 * s), 0u, size - 1u);
    return vec3(px, py, face);
}`;

    const DIR_TO_FACE_UV_FLOAT_WGSL = /* wgsl */ `
fn dirToFaceUVf(dir: vec3f) -> vec3f {
    let absDir = abs(dir);
    var face: f32;
    var u: f32;
    var v: f32;
    if (absDir.x >= absDir.y && absDir.x >= absDir.z) {
        if (dir.x > 0.0) {
            face = 0.0; u = -dir.z / absDir.x; v = -dir.y / absDir.x;
        } else {
            face = 1.0; u = dir.z / absDir.x; v = -dir.y / absDir.x;
        }
    } else if (absDir.y >= absDir.x && absDir.y >= absDir.z) {
        if (dir.y > 0.0) {
            face = 2.0; u = dir.x / absDir.y; v = dir.z / absDir.y;
        } else {
            face = 3.0; u = dir.x / absDir.y; v = -dir.z / absDir.y;
        }
    } else {
        if (dir.z > 0.0) {
            face = 4.0; u = dir.x / absDir.z; v = -dir.y / absDir.z;
        } else {
            face = 5.0; u = -dir.x / absDir.z; v = -dir.y / absDir.z;
        }
    }
    return vec3f(face, u * 0.5 + 0.5, v * 0.5 + 0.5);
}`;

    const V = GPUShaderStage.VERTEX;
    const F = GPUShaderStage.FRAGMENT;

    const splatBindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: V | F, buffer: { type: "uniform" } },
            {
                binding: 1,
                visibility: F,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            },
            {
                binding: 2,
                visibility: V | F,
                texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" },
            },
            {
                binding: 3,
                visibility: V,
                texture: { sampleType: "uint", viewDimension: "2d-array" },
            },
            { binding: 4, visibility: V | F, buffer: { type: "uniform" } },
            { binding: 5, visibility: F, buffer: { type: "uniform" } },
            { binding: 6, visibility: V, buffer: { type: "read-only-storage" } },
            { binding: 7, visibility: V | F, buffer: { type: "uniform" } },
            { binding: 8, visibility: V, buffer: { type: "read-only-storage" } },
        ],
    });

    const splatPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [splatBindGroupLayout],
    });

    const splatShader = device.createShaderModule({
        code: /* wgsl */ `
            ${PROBE_PARAMS_WGSL}

            struct SplatScene {
                viewProj: mat4x4f,
                cameraPos: vec3f,
                _pad: f32,
            }

            @group(0) @binding(0) var<uniform> splatScene: SplatScene;
            @group(0) @binding(1) var probeLitTex: texture_2d_array<f32>;
            @group(0) @binding(2) var probeRadialTex: texture_2d_array<f32>;
            @group(0) @binding(3) var probeEidTex: texture_2d_array<u32>;
            @group(0) @binding(4) var<uniform> params: ProbeParams;

            ${SKY_STRUCT_WGSL}
            @group(0) @binding(5) var<uniform> sky: Sky;

            @group(0) @binding(6) var<storage, read> splatVisibility: array<u32>;

            ${SKY_SCENE_STRUCT_WGSL}
            @group(0) @binding(7) var<uniform> scene: SkyScene;

            @group(0) @binding(8) var<storage, read> edgeMasks: array<u32>;

            ${HAZE_WGSL}
            ${COMPUTE_SKY_DIR_WGSL}
            ${OKLAB_WGSL}
            ${NOISE_WGSL}
            ${STARS_WGSL}
            ${MOON_WGSL}
            ${CLOUDS_WGSL}
            ${SAMPLE_SKY_WGSL}

            ${DIR_TO_FACE_UV_UINT_WGSL}
            ${DIR_TO_FACE_UV_FLOAT_WGSL}

            const EXPANSION: f32 = ${EXPANSION};
            const NEAR: f32 = ${NEAR};
            const FAR: f32 = ${FAR};
            const PROBE_SIZE_D: u32 = ${PROBE_SIZE}u;
            const FACE_TEXELS_D: u32 = ${FACE_TEXELS}u;

            fn texelDir(face: u32, u: f32, v: f32) -> vec3<f32> {
                let sc = 2.0 * u - 1.0;
                let tc = 2.0 * v - 1.0;
                switch (face) {
                    case 0u: { return vec3(1.0, -tc, -sc); }
                    case 1u: { return vec3(-1.0, -tc, sc); }
                    case 2u: { return vec3(sc, 1.0, tc); }
                    case 3u: { return vec3(sc, -1.0, -tc); }
                    case 4u: { return vec3(sc, -tc, 1.0); }
                    default: { return vec3(-sc, -tc, -1.0); }
                }
            }

            fn bayer4(pos: vec2<u32>) -> f32 {
                let x = pos.x % 4u;
                let y = pos.y % 4u;
                let bayer = array<f32, 16>(
                     0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
                    12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0,
                     3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
                    15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0,
                );
                return bayer[y * 4u + x];
            }

            // --- Splat vertex/fragment ---

            struct SplatVsOut {
                @builtin(position) position: vec4<f32>,
                @location(0) @interpolate(flat) texelXY: u32,
                @location(1) @interpolate(flat) texelInfo: u32,
                @location(2) @interpolate(flat) hazePos: vec3<f32>,
            }

            @vertex
            fn vs(@builtin(instance_index) iid: u32, @builtin(vertex_index) vid: u32) -> SplatVsOut {
                let size = PROBE_SIZE_D;
                let faceTexels = FACE_TEXELS_D;

                let entry = splatVisibility[iid];
                let slot = entry / faceTexels;
                let texelIdx = entry % faceTexels;
                let probeIdx = slot / 6u;
                let face = slot % 6u;
                let layer = slot;
                let px = texelIdx % size;
                let py = texelIdx / size;
                let isPrev = probeIdx == ${PROBE_PREV}u;

                let radial = textureLoad(probeRadialTex, vec2<u32>(px, py), layer, 0).r;

                var output: SplatVsOut;
                output.texelXY = 0u;
                output.texelInfo = 0u;
                output.hazePos = vec3(0.0);

                let near = params.ranges[probeIdx].x;
                let far = params.ranges[probeIdx].y;
                let chebyshev = near + radial * (far - near);
                let origin = params.origins[probeIdx].xyz;

                let sizeF = f32(size);
                let halfTexel = 0.5 / sizeF;
                let expansion = EXPANSION / sizeF;
                let hs = max(halfTexel + expansion, 0.0);

                let centerU = (f32(px) + 0.5) / sizeF;
                let centerV = (f32(py) + 0.5) / sizeF;

                let centerDir = texelDir(face, centerU, centerV);
                let cMC = max(abs(centerDir.x), max(abs(centerDir.y), abs(centerDir.z)));
                let centerPos = origin + centerDir * (chebyshev / cMC);

                var faceNormal: vec3<f32>;
                switch (face) {
                    case 0u: { faceNormal = vec3(1.0, 0.0, 0.0); }
                    case 1u: { faceNormal = vec3(-1.0, 0.0, 0.0); }
                    case 2u: { faceNormal = vec3(0.0, 1.0, 0.0); }
                    case 3u: { faceNormal = vec3(0.0, -1.0, 0.0); }
                    case 4u: { faceNormal = vec3(0.0, 0.0, 1.0); }
                    default: { faceNormal = vec3(0.0, 0.0, -1.0); }
                }
                let viewDir = normalize(centerPos - splatScene.cameraPos);
                let cosTheta = max(abs(dot(viewDir, faceNormal)), 0.14);
                let tanTheta = sqrt(1.0 - cosTheta * cosTheta) / cosTheta;
                let hsEdge = halfTexel * 1.15 + 0.0005 * tanTheta;

                let maskIdx = layer * faceTexels + py * size + px;
                let emask = edgeMasks[maskIdx];
                let hsL = select(hs, hsEdge, (emask & 1u) != 0u);
                let hsR = select(hs, hsEdge, (emask & 2u) != 0u);
                let hsB = select(hs, hsEdge, (emask & 4u) != 0u);
                let hsT = select(hs, hsEdge, (emask & 8u) != 0u);

                let u0 = centerU - hsL;
                let v0 = centerV - hsB;
                let u1 = centerU + hsR;
                let v1 = centerV + hsT;

                var cu: f32;
                var cv: f32;
                switch (vid) {
                    case 0u, 3u: { cu = u0; cv = v0; }
                    case 1u:     { cu = u1; cv = v0; }
                    case 2u, 4u: { cu = u1; cv = v1; }
                    default:     { cu = u0; cv = v1; }
                }

                let rawDir = texelDir(face, cu, cv);
                let maxComp = max(abs(rawDir.x), max(abs(rawDir.y), abs(rawDir.z)));
                let worldPos = origin + rawDir * (chebyshev / maxComp);

                output.position = splatScene.viewProj * vec4(worldPos, 1.0);
                let tid = face * faceTexels + py * size + px;
                let h = (tid * 2654435761u) >> 24u;
                output.position.z += f32(h) * 1e-9 * output.position.w;
                if (probeIdx == ${PROBE_EYE}u) {
                    output.position.z += 0.001 * output.position.w;
                }

                if (isPrev) {
                    let eyeOrigin = params.origins[0u].xyz;
                    let toTexel = centerPos - eyeOrigin;
                    let eyeFUV = dirToFaceUV(toTexel, size);
                    let eyeFace = eyeFUV.z;
                    let eyeMask = u32(params.masks.x);
                    if ((eyeMask & (1u << eyeFace)) != 0u) {
                        let prevEid = textureLoad(probeEidTex, vec2<u32>(px, py), layer, 0).r;
                        let eyeEid = textureLoad(probeEidTex, vec2<u32>(eyeFUV.xy), eyeFace, 0).r;
                        if (prevEid != eyeEid) {
                            output.position = vec4(0.0, 0.0, 0.0, 0.0);
                            return output;
                        }
                    }
                }
                output.position.z = min(output.position.z, output.position.w);

                output.texelXY = px | (py << 16u);
                output.texelInfo = layer | (face << 8u) | (probeIdx << 16u);
                output.hazePos = centerPos;

                return output;
            }

            @fragment
            fn fs(input: SplatVsOut) -> @location(0) vec4<f32> {
                let px = input.texelXY & 0xFFFFu;
                let py = input.texelXY >> 16u;
                let layer = input.texelInfo & 0xFFu;
                let probeIdx = (input.texelInfo >> 16u) & 0xFFu;

                let fadeT = params.state.y;
                let isPrev = probeIdx == ${PROBE_PREV}u;

                var litColor = textureLoad(probeLitTex, vec2<u32>(px, py), layer, 0);

                let hazeDist = length(input.hazePos - splatScene.cameraPos);
                litColor = vec4(applyHaze(litColor.rgb, hazeDist), litColor.a);

                var color = litColor.rgb;
                let isEye = probeIdx == ${PROBE_EYE}u;
                var alpha = 1.0;
                if (!isEye) {
                    if (isPrev) {
                        alpha = fadeT;
                    } else if (fadeT > 0.0 && fadeT < 1.0) {
                        alpha = -fadeT;
                    }
                }

                if (alpha < 0.0) {
                    let threshold = bayer4(vec2<u32>(input.position.xy));
                    if (threshold >= -alpha) { discard; }
                } else if (alpha < 1.0) {
                    let threshold = bayer4(vec2<u32>(input.position.xy));
                    if (threshold < alpha) { discard; }
                }

                return vec4(color, 1.0);
            }

            // --- Background vertex/fragment ---

            struct SkyVsOut {
                @builtin(position) position: vec4<f32>,
            }

            @vertex
            fn skyVs(@builtin(vertex_index) vid: u32) -> SkyVsOut {
                var pos = array<vec2<f32>, 3>(
                    vec2(-1.0, -1.0),
                    vec2(3.0, -1.0),
                    vec2(-1.0, 3.0),
                );
                var output: SkyVsOut;
                output.position = vec4(pos[vid], 1.0, 1.0);
                return output;
            }

            @fragment
            fn skyFs(input: SkyVsOut) -> @location(0) vec4<f32> {
                let screenX = input.position.x / scene.viewport.x;
                let screenY = input.position.y / scene.viewport.y;
                let dir = computeSkyDir(screenX, screenY);

                let size = PROBE_SIZE_D;
                let fuv = dirToFaceUVf(dir);
                let face = u32(fuv.x);
                let fSize = f32(size);
                let fpx = clamp(u32(fuv.y * fSize), 0u, size - 1u);
                let fpy = clamp(u32(fuv.z * fSize), 0u, size - 1u);

                let radial = textureLoad(probeRadialTex, vec2<u32>(fpx, fpy), face, 0).r;
                let isSky = radial >= 0.999;

                if (isSky) {
                    let color = textureLoad(probeLitTex, vec2<u32>(fpx, fpy), face, 0).rgb;
                    return vec4(color, 1.0);
                }

                let litColor = textureLoad(probeLitTex, vec2<u32>(fpx, fpy), face, 0).rgb;
                let chebyshev = radial * (FAR - NEAR) + NEAR;
                let normDir = normalize(dir);
                let maxC = max(abs(normDir.x), max(abs(normDir.y), abs(normDir.z)));
                let hazeDist = chebyshev / maxC;
                return vec4(applyHaze(litColor, hazeDist), 1.0);
            }
        `,
    });

    const splatDepthStencil: GPUDepthStencilState = {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
    };

    const splatPipeline = device.createRenderPipeline({
        layout: splatPipelineLayout,
        vertex: { module: splatShader, entryPoint: "vs" },
        fragment: {
            module: splatShader,
            entryPoint: "fs",
            targets: [{ format: "rgba8unorm" }],
        },
        depthStencil: splatDepthStencil,
        primitive: { topology: "triangle-list", cullMode: "none" },
    });

    const backgroundPipeline = device.createRenderPipeline({
        layout: splatPipelineLayout,
        vertex: { module: splatShader, entryPoint: "skyVs" },
        fragment: {
            module: splatShader,
            entryPoint: "skyFs",
            targets: [{ format: "rgba8unorm" }],
        },
        depthStencil: splatDepthStencil,
        primitive: { topology: "triangle-list", cullMode: "none" },
    });

    const displayBindGroup = device.createBindGroup({
        layout: splatBindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: splatSceneBuffer } },
            { binding: 1, resource: arrayViews.lit },
            { binding: 2, resource: arrayViews.radial },
            { binding: 3, resource: arrayViews.eid },
            { binding: 4, resource: { buffer: probeParamsBuffer } },
            { binding: 5, resource: { buffer: config.skyBuffer } },
            { binding: 6, resource: { buffer: visibilityBuffer } },
            { binding: 7, resource: { buffer: config.sceneBuffer } },
            { binding: 8, resource: { buffer: edgeMaskBuffer } },
        ],
    });

    const proj = cubePerspective(NEAR, FAR);
    // Staging for all face uniforms — written once per frame then uploaded in one writeBuffer call
    const faceUniformStaging = new Float32Array(PROBE_LAYERS * FACE_UNIFORM_STRIDE_F32);
    const ts = createTransitionState();
    const probeData = new Float32Array(32);
    const indirectData = new Uint32Array([6, 0, 0, 0]);
    const lightingParamsData = new Float32Array(32);
    // Persistent alias — avoids creating a new Uint32Array view every frame
    const lightingParamsU32 = new Uint32Array(lightingParamsData.buffer);
    // Reusable scratch buffers to eliminate per-frame heap allocations in encode()
    // Compact list of active face layers built each frame; length = activeLayerCount
    const activeLayerIndices = new Uint32Array(PROBE_LAYERS);
    const facePlanes: (number[][] | null)[] = new Array(PROBE_LAYERS).fill(null);
    const splatSceneData = new Float32Array(20);
    const origins: [[number, number, number], [number, number, number], [number, number, number]] = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ];
    let probeFrame = 0;

    function getActiveProbes(frame: number, crossfading: boolean): number {
        const eye = 1 << PROBE_EYE;
        if (frame === 0) return 0b111;
        if (frame % 2 === 0) return eye | (1 << PROBE_GRID);
        return eye | (crossfading ? 1 << PROBE_PREV : 1 << PROBE_GRID);
    }

    return {
        encode(encoder, params, colorView, depthView, viewProj) {
            const [cx, cy, cz] = params.cameraPos;
            const [fx, fy, fz] = params.cameraFwd;

            const result = updateTransition(ts, cx, cy, cz, params.time);
            const { crossfading, fadeT } = result;

            if (result.triggered) probeFrame = 0;
            const activeProbes = getActiveProbes(probeFrame, crossfading);

            const eyeMask = computeFaceMask(fx, fy, fz, EYE_CULL_COS);
            const gridMask = computeFaceMask(fx, fy, fz, GRID_CULL_COS);
            const prevMask = crossfading ? gridMask : 0;

            // Mutate pre-allocated origins array instead of allocating new tuples each frame
            origins[0][0] = cx; origins[0][1] = cy; origins[0][2] = cz;
            origins[1][0] = ts.origin[0]; origins[1][1] = ts.origin[1]; origins[1][2] = ts.origin[2];
            origins[2][0] = ts.prevOrigin[0]; origins[2][1] = ts.prevOrigin[1]; origins[2][2] = ts.prevOrigin[2];

            // Reset reusable frustum cache without allocating a new array each frame
            for (let i = 0; i < PROBE_LAYERS; i++) facePlanes[i] = null;

            // Upload face uniforms and compute frustum planes for each probe
            for (let p = 0; p < 3; p++) {
                if (!(activeProbes & (1 << p))) continue;
                let mask: number;
                if (p === PROBE_EYE) mask = eyeMask;
                else if (p === PROBE_GRID) mask = gridMask;
                else mask = prevMask;

                const [ox, oy, oz] = origins[p];
                const base = p * 6;

                for (let i = 0; i < 6; i++) {
                    if (!(mask & (1 << i))) continue;
                    const { dir, up } = SLICE_TARGETS[i];
                    const view = lookAt(
                        ox,
                        oy,
                        oz,
                        ox + dir[0],
                        oy + dir[1],
                        oz + dir[2],
                        up[0],
                        up[1],
                        up[2],
                    );
                    const vp = multiply(proj, view);
                    facePlanes[base + i] = extractFrustumPlanes(vp);
                    vp[1] *= -1;
                    vp[5] *= -1;
                    vp[9] *= -1;
                    vp[13] *= -1;
                    const stagingBase = (base + i) * FACE_UNIFORM_STRIDE_F32;
                    faceUniformStaging.set(vp, stagingBase);
                    faceUniformStaging[stagingBase + 16] = ox;
                    faceUniformStaging[stagingBase + 17] = oy;
                    faceUniformStaging[stagingBase + 18] = oz;
                    faceUniformStaging[stagingBase + 19] = NEAR;
                    faceUniformStaging[stagingBase + 20] = 0;
                    faceUniformStaging[stagingBase + 21] = 0;
                    faceUniformStaging[stagingBase + 22] = 0;
                    faceUniformStaging[stagingBase + 23] = FAR;
                }
            }
            // One upload for all active face uniforms instead of one per face
            device.queue.writeBuffer(faceUniformBuffer, 0, faceUniformStaging);

            const { grass, stones, orbs } = config.meshAABBs;

            // G-buffer capture per visible face per probe
            for (let p = 0; p < 3; p++) {
                if (!(activeProbes & (1 << p))) continue;
                let mask: number;
                if (p === PROBE_EYE) mask = eyeMask;
                else if (p === PROBE_GRID) mask = gridMask;
                else mask = prevMask;

                const base = p * 6;
                for (let i = 0; i < 6; i++) {
                    if (!(mask & (1 << i))) continue;
                    const layer = base + i;
                    const planes = facePlanes[layer]!;

                    const pass = encoder.beginRenderPass({
                        colorAttachments: [
                            {
                                view: faceViews.albedo[layer],
                                clearValue: [0, 0, 0, 0],
                                loadOp: "clear",
                                storeOp: "store",
                            },
                            {
                                view: faceViews.normal[layer],
                                clearValue: [0.5, 0.5, 0, 0],
                                loadOp: "clear",
                                storeOp: "store",
                            },
                            {
                                view: faceViews.radial[layer],
                                clearValue: [1, 0, 0, 0],
                                loadOp: "clear",
                                storeOp: "store",
                            },
                            {
                                view: faceViews.eid[layer],
                                clearValue: [0, 0, 0, 0],
                                loadOp: "clear",
                                storeOp: "store",
                            },
                        ],
                        depthStencilAttachment: {
                            view: faceViews.depth[layer],
                            depthClearValue: 1,
                            depthLoadOp: "clear",
                            depthStoreOp: "store",
                        },
                    });

                    const dynamicOffset = layer * FACE_UNIFORM_STRIDE;

                    if (aabbInFrustum(planes, grass)) {
                        pass.setPipeline(grassGbufferPipeline);
                        pass.setBindGroup(0, faceBindGroup, [dynamicOffset]);
                        pass.setVertexBuffer(0, config.vertexBuffer);
                        pass.setIndexBuffer(config.indexBuffer, "uint16");
                        pass.drawIndexed(config.indexCount);
                    }

                    if (aabbInFrustum(planes, stones)) {
                        pass.setPipeline(stoneGbufferPipeline);
                        pass.setBindGroup(0, faceBindGroup, [dynamicOffset]);
                        pass.setVertexBuffer(0, config.stoneVertexBuffer);
                        pass.setIndexBuffer(config.stoneIndexBuffer, "uint16");
                        pass.drawIndexed(config.stoneIndexCount);
                    }

                    if (aabbInFrustum(planes, orbs)) {
                        pass.setPipeline(emissiveGbufferPipeline);
                        pass.setBindGroup(0, faceBindGroup, [dynamicOffset]);
                        pass.setVertexBuffer(0, config.orbVertexBuffer);
                        pass.setIndexBuffer(config.orbIndexBuffer, "uint16");
                        for (let j = 0; j < 4; j++) {
                            pass.setBindGroup(1, orbBindGroups[j]);
                            pass.drawIndexed(config.orbIndexCount);
                        }
                    }

                    pass.end();
                }
            }

            // Lighting compute
            lightingParamsData[0] = cx;
            lightingParamsData[1] = cy;
            lightingParamsData[2] = cz;
            lightingParamsData[3] = NEAR;
            lightingParamsData[4] = params.sunDir[0];
            lightingParamsData[5] = params.sunDir[1];
            lightingParamsData[6] = params.sunDir[2];
            lightingParamsData[7] = FAR;
            lightingParamsData[8] = params.sunColor[0];
            lightingParamsData[9] = params.sunColor[1];
            lightingParamsData[10] = params.sunColor[2];
            lightingParamsData[11] = params.shadowFade;
            lightingParamsData[12] = params.ambient[0];
            lightingParamsData[13] = params.ambient[1];
            lightingParamsData[14] = params.ambient[2];
            // Use pre-allocated Uint32Array alias — no transient typed-array views per frame
            lightingParamsU32[15] = eyeMask;
            lightingParamsData[16] = ts.origin[0];
            lightingParamsData[17] = ts.origin[1];
            lightingParamsData[18] = ts.origin[2];
            lightingParamsU32[19] = gridMask;
            lightingParamsData[20] = ts.prevOrigin[0];
            lightingParamsData[21] = ts.prevOrigin[1];
            lightingParamsData[22] = ts.prevOrigin[2];
            lightingParamsU32[23] = prevMask;
            lightingParamsU32[24] = activeProbes;
            lightingParamsU32[25] = params.pointLightCount;
            device.queue.writeBuffer(lightingParamsBuffer, 0, lightingParamsData);

            // Build compact list of layers that are both probe-active and face-masked.
            // Masked-out faces produce no vis entries (cull skips them), so lighting work
            // for those layers is entirely wasted — skip them at the dispatch level.
            let activeLayerCount = 0;
            for (let probeIdx = 0; probeIdx < 3; probeIdx++) {
                if (!(activeProbes & (1 << probeIdx))) continue;
                const probeMask = probeIdx === PROBE_EYE ? eyeMask
                    : probeIdx === PROBE_GRID ? gridMask
                    : prevMask;
                for (let face = 0; face < 6; face++) {
                    if (probeMask & (1 << face)) {
                        activeLayerIndices[activeLayerCount++] = probeIdx * 6 + face;
                    }
                }
            }
            device.queue.writeBuffer(activeLayerIndexBuffer, 0, activeLayerIndices, 0, activeLayerCount);

            const wg = Math.ceil(PROBE_SIZE / 8);

            const computePass = encoder.beginComputePass();
            computePass.setPipeline(lightingPipeline);
            computePass.setBindGroup(0, lightingBindGroup);
            computePass.dispatchWorkgroups(wg, wg, activeLayerCount);
            computePass.end();

            // Edge mask compute
            const edgeMaskPass = encoder.beginComputePass();
            edgeMaskPass.setPipeline(edgeMaskPipeline);
            edgeMaskPass.setBindGroup(0, edgeMaskBindGroup);
            edgeMaskPass.dispatchWorkgroups(wg, wg, prevMask > 0 ? PROBE_LAYERS : 12);
            edgeMaskPass.end();

            // Probe params
            probeData.fill(0);
            for (let i = 0; i < 3; i++) {
                const [ox, oy, oz] = origins[i];
                probeData[i * 4] = ox;
                probeData[i * 4 + 1] = oy;
                probeData[i * 4 + 2] = oz;
            }
            probeData[12] = NEAR;
            probeData[13] = FAR;
            probeData[16] = NEAR;
            probeData[17] = FAR;
            probeData[20] = NEAR;
            probeData[21] = FAR;
            probeData[24] = eyeMask;
            probeData[25] = gridMask;
            probeData[26] = prevMask;
            probeData[28] = 0; // mode 0
            probeData[29] = fadeT;
            device.queue.writeBuffer(probeParamsBuffer, 0, probeData);

            // Reset indirect args and run cull
            indirectData[1] = 0;
            device.queue.writeBuffer(indirectArgsBuffer, 0, indirectData);

            const cullPass = encoder.beginComputePass();
            cullPass.setPipeline(cullPipeline);
            cullPass.setBindGroup(0, cullBindGroup);
            cullPass.dispatchWorkgroups(Math.ceil(FACE_TEXELS / 64), prevMask > 0 ? PROBE_LAYERS : 12);
            cullPass.end();

            // Upload splat scene buffer (viewProj + cameraPos) — reuse pre-allocated array
            splatSceneData.set(viewProj, 0);
            splatSceneData[16] = cx;
            splatSceneData[17] = cy;
            splatSceneData[18] = cz;
            device.queue.writeBuffer(splatSceneBuffer, 0, splatSceneData);

            // Display render pass
            const displayPass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: colorView,
                        clearValue: [0, 0, 0, 1],
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
                depthStencilAttachment: {
                    view: depthView,
                    depthClearValue: 1,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });

            displayPass.setPipeline(backgroundPipeline);
            displayPass.setBindGroup(0, displayBindGroup);
            displayPass.draw(3);

            displayPass.setPipeline(splatPipeline);
            displayPass.setBindGroup(0, displayBindGroup);
            displayPass.drawIndirect(indirectArgsBuffer, 0);

            displayPass.end();

            probeFrame++;
        },

        destroy() {
            probeAlbedo.destroy();
            probeNormal.destroy();
            probeRadial.destroy();
            probeEid.destroy();
            probeDepth.destroy();
            probeLit.destroy();
            faceUniformBuffer.destroy();
            probeParamsBuffer.destroy();
            indirectArgsBuffer.destroy();
            visibilityBuffer.destroy();
            splatSceneBuffer.destroy();
            lightingParamsBuffer.destroy();
            edgeMaskBuffer.destroy();
            activeLayerIndexBuffer.destroy();
        },
    };
}
