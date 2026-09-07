import { defineConfig } from "vite";

export default defineConfig({
    server: { port: 3000 },
    base: "./",
    build: {
        target: "esnext",
        outDir: "site/demo",
        emptyOutDir: true,
        // `hidden` emits a `.map` beside every chunk but no `//# sourceMappingURL` comment: the maps
        // go to Datadog from `.github/workflows/pages.yml` and are deleted before the Pages artifact
        // is built, so a visitor never sees them and a scraper never finds a link to them.
        //
        // The landing page `site/index.html` is not a Vite input (Vite's entry is the repo-root
        // `index.html`, which builds the demo bundle this page embeds in an iframe), so nothing here
        // can fill its RUM `version` token — the workflow does that with a replace step instead.
        sourcemap: "hidden",
    },
});
