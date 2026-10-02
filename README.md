# 🐧 Penguinify

Upload a photo or snap one with your webcam. Faces are detected in the browser (SCRFD) and replaced with penguin stickers. Nothing leaves your device.

## Run it
It's a static site (no build step). Either:
- open `index.html` directly, or
- serve it: `python3 -m http.server 8000` then visit http://localhost:8000 (needed for the webcam on some browsers)
- or deploy the folder to GitHub Pages / Netlify / Vercel.

Internet is needed once for ONNX Runtime (jsDelivr), the face model (Hugging Face, user-triggered) and Google Fonts.

## Features
- Face detection with SCRFD-10GF (InsightFace, ~17 MB ONNX) via ONNX Runtime Web (WebGPU if available, else WASM). The model is downloaded once on request and cached in IndexedDB, then works offline. You can also pick your own scrfd_10g.onnx / det_10g.onnx file.
- Built for small faces in group photos: one full pass at up to 1920px, plus "Scan deeper" with overlapping enlarged tiles
- Multi-face detection, sticker auto-sized and tilted to match head roll
- 14 penguin styles (Classic, Party, Cool, Emperor, Derp, Grumpy, Nerd, Pirate, Wizard, Punk, Chick, Sleepy, Lovey, Chef)
- "Mix" mode (default) deals a different penguin to every face; Shuffle re-deals, or pick a style per penguin / "Use for all"
- Drag, resize (slider or mouse wheel), tilt, flip, remove, add by hand
- Upload, drag-and-drop, paste, or webcam
- Chaos button (random styles + noot noot), snow and fish overlays
- Download as PNG
