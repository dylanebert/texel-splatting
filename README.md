# Texel Splatting

<p align="center">
  <a href="https://arxiv.org/abs/2603.14587"><img src="https://img.shields.io/badge/arXiv-2603.14587-b31b1b.svg" alt="arXiv"></a>
  <a href="https://dylanebert.com/texel-splatting"><img src="https://img.shields.io/badge/project-live-4285F4.svg" alt="Project"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-5b9a41.svg" alt="License"></a>
  <a href="https://ko-fi.com/individualkex"><img src="https://img.shields.io/badge/Ko--fi-support-ff5e5b?logo=ko-fi" alt="Ko-fi"></a>
</p>

![Three views of a stone circle rendered with texel splatting](hero.jpg)

Perspective-stable 3D pixel art. Render into cubemaps from fixed origins, splat to screen as world-space quads. Eliminates shimmer under rotation and translation.

## Demo

Try it [live](https://dylanebert.com/texel-splatting). Requires [WebGPU](https://caniuse.com/webgpu).

### Run locally

Install [Bun](https://bun.sh), then:

```bash
git clone https://github.com/dylanebert/texel-splatting.git
cd texel-splatting
bun install
bun dev
```

Opens at `localhost:3000`.

## Structure

`src/` is the technique. `demo/` is the demo scene.

## Citation

```bibtex
@article{ebert2026texelsplatting,
  title={Texel Splatting: Perspective-Stable 3D Pixel Art},
  author={Ebert, Dylan},
  year={2026}
}
```

## License

[MIT](LICENSE)
