/* Penguinify – browser-only face detection (SCRFD via ONNX Runtime Web), penguins drawn as SVG stickers. */
const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
// SCRFD-10GF (InsightFace buffalo_l detector, ~17 MB). Mirrors are tried in order.
const MODEL_URLS = [
  "https://huggingface.co/globalnebula/insightface-buffalo-l-onnx/resolve/main/scrfd_10g.onnx",
  "https://huggingface.co/maze/faceX/resolve/main/det_10g.onnx",
  "https://huggingface.co/Nomnommish/FaceDetection/resolve/main/det_10g.onnx",
];
const MODEL_KEY = "scrfd_10g_v1", MODEL_MIN_BYTES = 5e6, MODEL_APPROX_BYTES = 16.9e6;
const MAX_DIM = 3200;

/* ---------- penguin art ---------- */
const STYLES = [
  { id: "classic", name: "Classic" },
  { id: "party",   name: "Party" },
  { id: "cool",    name: "Cool" },
  { id: "emperor", name: "Emperor" },
  { id: "derp",    name: "Derp" },
  { id: "grumpy",  name: "Grumpy" },
  { id: "nerd",    name: "Nerd" },
  { id: "pirate",  name: "Pirate" },
  { id: "wizard",  name: "Wizard" },
  { id: "punk",    name: "Punk" },
  { id: "baby",    name: "Chick" },
  { id: "sleepy",  name: "Sleepy" },
  { id: "lovey",   name: "Lovey" },
  { id: "chef",    name: "Chef" },
];

function penguinSVG(style) {
  let headFill = "#141a26", faceFill = "#fff";
  if (style === "baby") { headFill = "#9aa6b4"; faceFill = "#f6f3ea"; }
  const head = `<ellipse cx="100" cy="108" rx="92" ry="98" fill="${headFill}"/>`;
  const face = `<ellipse cx="68" cy="108" rx="40" ry="50" fill="${faceFill}"/><ellipse cx="132" cy="108" rx="40" ry="50" fill="${faceFill}"/><ellipse cx="100" cy="138" rx="50" ry="44" fill="${faceFill}"/>`;
  let cheeks = `<circle cx="46" cy="128" r="10" fill="#ffb4b4" opacity=".8"/><circle cx="154" cy="128" r="10" fill="#ffb4b4" opacity=".8"/>`;
  const beak = `<path d="M78 120 Q100 108 122 120 Q118 152 100 158 Q82 152 78 120Z" fill="#ff8a00" stroke="#c25f00" stroke-width="3" stroke-linejoin="round"/><path d="M86 126 Q100 120 114 126" stroke="#ffc266" stroke-width="4" fill="none" stroke-linecap="round"/>`;
  const eye = (x, y, px = 0, py = 0, r = 14, pr = 7) =>
    `<circle cx="${x}" cy="${y}" r="${r}" fill="#fff" stroke="#141a26" stroke-width="3"/><circle cx="${x + px}" cy="${y + py}" r="${pr}" fill="#141a26"/><circle cx="${x + px + 2}" cy="${y + py - 2}" r="2.4" fill="#fff"/>`;
  const heart = (x, y, s, c = "#ff2d6f") =>
    `<path d="M0 8 C-14 -6 -8 -18 0 -9 C8 -18 14 -6 0 8Z" fill="${c}" transform="translate(${x} ${y}) scale(${s})"/>`;
  let eyes = eye(70, 92, 1, 1) + eye(130, 92, -1, 1);
  let extra = "", under = "";

  switch (style) {
    case "party":
      extra = `<path d="M100 -30 L58 38 Q100 50 142 38Z" fill="#ff3d81"/><path d="M80 6 L92 -2 M72 20 L104 8 M64 32 L118 16" stroke="#ffe14d" stroke-width="6" stroke-linecap="round"/><circle cx="100" cy="-32" r="10" fill="#ffe14d"/>`;
      break;
    case "cool":
      eyes = `<path d="M40 78 H160 L152 110 Q148 118 138 118 H112 Q104 118 102 108 Q100 102 98 108 Q96 118 88 118 H62 Q52 118 48 110Z" fill="#05070c"/><path d="M52 86 L72 86 L58 98Z" fill="#fff" opacity=".35"/><path d="M112 86 L132 86 L118 98Z" fill="#fff" opacity=".35"/>`;
      break;
    case "emperor":
      under = `<ellipse cx="22" cy="132" rx="16" ry="34" fill="#ffc93c" transform="rotate(14 22 132)"/><ellipse cx="178" cy="132" rx="16" ry="34" fill="#ffc93c" transform="rotate(-14 178 132)"/>`;
      extra = `<path d="M54 22 L66 -14 L84 12 L100 -22 L116 12 L134 -14 L146 22 Q100 34 54 22Z" fill="#ffd23f" stroke="#b8860b" stroke-width="3" stroke-linejoin="round"/><circle cx="66" cy="-14" r="5" fill="#e63946"/><circle cx="100" cy="-22" r="6" fill="#3a86ff"/><circle cx="134" cy="-14" r="5" fill="#e63946"/>`;
      break;
    case "derp":
      eyes = eye(68, 94, -5, -4, 16, 6) + eye(132, 86, 6, 5, 12, 8);
      extra = `<path d="M92 150 Q100 188 108 150Z" fill="#ff5d73" stroke="#c4324a" stroke-width="3" stroke-linejoin="round"/>`;
      break;
    case "grumpy":
      eyes = eye(70, 96, 2, 3, 13, 7) + eye(130, 96, -2, 3, 13, 7)
        + `<path d="M44 66 L90 86" stroke="#141a26" stroke-width="10" stroke-linecap="round"/><path d="M156 66 L110 86" stroke="#141a26" stroke-width="10" stroke-linecap="round"/>`;
      break;
    case "nerd":
      eyes = eye(70, 94, 2, 2, 12, 6) + eye(130, 94, -2, 2, 12, 6)
        + `<circle cx="70" cy="94" r="24" fill="#bfe6ff" fill-opacity=".35" stroke="#7a4b1e" stroke-width="5"/><circle cx="130" cy="94" r="24" fill="#bfe6ff" fill-opacity=".35" stroke="#7a4b1e" stroke-width="5"/><path d="M94 92 Q100 86 106 92" stroke="#7a4b1e" stroke-width="5" fill="none"/><path d="M46 90 L14 84 M154 90 L186 84" stroke="#7a4b1e" stroke-width="5" stroke-linecap="round"/><rect x="95" y="84" width="10" height="16" fill="#fff" opacity=".9" rx="2"/><path d="M92 100 H108" stroke="#aaa" stroke-width="2"/>`;
      extra = `<path d="M100 12 Q92 -14 108 -18 M100 12 Q112 -6 120 -10" stroke="#141a26" stroke-width="5" fill="none" stroke-linecap="round"/>`;
      break;
    case "pirate":
      eyes = eye(70, 92, 1, 1) + `<ellipse cx="130" cy="94" rx="20" ry="18" fill="#05070c"/><path d="M118 82 L28 64 M142 84 L186 70" stroke="#05070c" stroke-width="5" stroke-linecap="round"/>`;
      extra = `<path d="M10 66 Q100 14 190 66 L190 84 Q100 36 10 84Z" fill="#d62839"/><circle cx="46" cy="64" r="4" fill="#fff"/><circle cx="82" cy="46" r="4" fill="#fff"/><circle cx="120" cy="46" r="4" fill="#fff"/><circle cx="156" cy="62" r="4" fill="#fff"/><path d="M186 70 Q204 78 198 96 Q192 84 180 82Z" fill="#d62839"/><circle cx="16" cy="134" r="7" fill="none" stroke="#ffc93c" stroke-width="4"/>`;
      break;
    case "wizard":
      extra = `<path d="M104 -48 Q84 -10 40 40 Q100 56 160 40 Q122 0 104 -48Z" fill="#5b3fd1"/><ellipse cx="100" cy="42" rx="84" ry="16" fill="#4a32b0"/><path d="M52 34 Q100 52 148 34 L148 42 Q100 60 52 42Z" fill="#ffd23f"/><path d="M96 -6 l3 8 8 1 -6 5 2 8 -7 -4 -7 4 2 -8 -6 -5 8 -1z" fill="#ffe14d" transform="translate(4 10)"/><circle cx="82" cy="20" r="3" fill="#ffe14d"/><circle cx="126" cy="14" r="3" fill="#ffe14d"/>`;
      break;
    case "punk":
      eyes = eye(70, 94, 1, 1) + eye(130, 94, -1, 1)
        + `<path d="M48 76 L92 90" stroke="#141a26" stroke-width="8" stroke-linecap="round"/><path d="M152 76 L108 90" stroke="#141a26" stroke-width="8" stroke-linecap="round"/>`;
      extra = `<path d="M62 16 L68 -26 L82 4 L90 -40 L100 0 L110 -40 L118 4 L132 -26 L138 16 Q100 28 62 16Z" fill="#ff2d95"/><path d="M82 4 L90 -40 L100 0Z" fill="#7cf03d" opacity=".9"/><circle cx="22" cy="112" r="5" fill="#d9dde4" stroke="#8a93a1"/><circle cx="178" cy="112" r="5" fill="#d9dde4" stroke="#8a93a1"/><circle cx="26" cy="126" r="4" fill="#d9dde4" stroke="#8a93a1"/><circle cx="174" cy="126" r="4" fill="#d9dde4" stroke="#8a93a1"/>`;
      break;
    case "baby":
      eyes = eye(68, 98, 0, 1, 19, 11) + eye(132, 98, 0, 1, 19, 11)
        + `<circle cx="73" cy="92" r="4" fill="#fff"/><circle cx="137" cy="92" r="4" fill="#fff"/>`;
      cheeks = `<circle cx="42" cy="130" r="13" fill="#ff9aa9" opacity=".8"/><circle cx="158" cy="130" r="13" fill="#ff9aa9" opacity=".8"/>`;
      extra = `<path d="M100 14 Q86 -12 102 -20 M100 14 Q104 -8 120 -14 M100 14 Q116 2 126 6" stroke="#9aa6b4" stroke-width="7" fill="none" stroke-linecap="round"/>`;
      break;
    case "sleepy":
      eyes = `<path d="M54 94 Q70 108 86 94" stroke="#141a26" stroke-width="5" fill="none" stroke-linecap="round"/><path d="M114 94 Q130 108 146 94" stroke="#141a26" stroke-width="5" fill="none" stroke-linecap="round"/>`;
      extra = `<path d="M40 44 Q56 -22 130 -10 Q178 0 176 52 Q104 22 40 44Z" fill="#4361ee"/><path d="M40 44 Q104 22 176 52 L174 62 Q104 34 42 56Z" fill="#fff"/><circle cx="178" cy="68" r="12" fill="#fff" stroke="#cfd8e3" stroke-width="2"/><text x="150" y="-4" font-family="Arial,sans-serif" font-weight="800" font-size="30" fill="#4361ee" stroke="#fff" stroke-width="3" paint-order="stroke">z</text><text x="172" y="-24" font-family="Arial,sans-serif" font-weight="800" font-size="22" fill="#4361ee" stroke="#fff" stroke-width="3" paint-order="stroke">z</text>`;
      break;
    case "lovey":
      eyes = heart(70, 96, 1.5) + heart(130, 96, 1.5);
      cheeks = `<circle cx="44" cy="128" r="12" fill="#ff8fa8" opacity=".9"/><circle cx="156" cy="128" r="12" fill="#ff8fa8" opacity=".9"/>`;
      extra = heart(60, 14, .9, "#ff6f9c") + heart(100, -6, 1.2) + heart(142, 16, .8, "#ff6f9c");
      break;
    case "chef":
      extra = `<ellipse cx="62" cy="14" rx="28" ry="26" fill="#fff" stroke="#cfd8e3" stroke-width="3"/><ellipse cx="138" cy="14" rx="28" ry="26" fill="#fff" stroke="#cfd8e3" stroke-width="3"/><ellipse cx="100" cy="-8" rx="34" ry="32" fill="#fff" stroke="#cfd8e3" stroke-width="3"/><rect x="58" y="24" width="84" height="28" rx="6" fill="#fff" stroke="#cfd8e3" stroke-width="3"/><path d="M76 30 V48 M100 30 V48 M124 30 V48" stroke="#e6edf4" stroke-width="3"/><path d="M70 150 Q84 138 100 148 Q116 138 130 150 Q116 164 100 156 Q84 164 70 150Z" fill="#3a2a1a"/>`;
      break;
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -50 240 270" width="240" height="270">${under}${head}${face}${cheeks}${eyes}${beak}${extra}</svg>`;
}
// Sticker is drawn so that the body ellipse (width ~184) is centred in the 240x270 viewBox.
const VB = { x: -20, y: -50, w: 240, h: 270, cx: 100, cy: 108, bodyW: 184 };

const styleImgs = {};
function loadStyles() {
  return Promise.all(STYLES.map(s => new Promise(res => {
    const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(penguinSVG(s.id));
    const img = new Image();
    img.onload = () => { styleImgs[s.id] = img; img.dataset.src = url; res(); };
    img.src = url;
    s.url = url;
  })));
}

/* ---------- state ---------- */
const $ = id => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");
let base = null;              // canvas holding the source photo
let penguins = [];            // {x,y,w,rot,flip,style}
let sel = -1;
let currentStyle = "mix";   // "mix" = a different penguin for every face
let snowOn = false, fishOn = false;
let seed = 7;

/* ---------- rendering ---------- */
function rand() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }

function render(forExport = false) {
  if (!base) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(base, 0, 0);

  if (fishOn) {
    seed = 42;
    const n = Math.round(canvas.width / 120);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) {
      ctx.save();
      ctx.translate(rand() * canvas.width, rand() * canvas.height);
      ctx.rotate((rand() - .5) * 1.6);
      ctx.font = `${canvas.width / 16 * (.6 + rand() * .8)}px serif`;
      ctx.fillText("🐟", 0, 0);
      ctx.restore();
    }
  }

  penguins.forEach((p, i) => {
    const img = styleImgs[p.style];
    const s = p.w / VB.bodyW;
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot * Math.PI / 180);
    ctx.scale(p.flip ? -s : s, s);
    ctx.shadowColor = "rgba(0,0,0,.35)"; ctx.shadowBlur = 14 * s; ctx.shadowOffsetY = 5 * s;
    ctx.drawImage(img, -(VB.cx - VB.x), -(VB.cy - VB.y), VB.w, VB.h);
    ctx.restore();
    if (!forExport && i === sel) {
      ctx.save();
      ctx.translate(p.x, p.y); ctx.rotate(p.rot * Math.PI / 180);
      ctx.strokeStyle = "#ff8a00"; ctx.lineWidth = Math.max(3, canvas.width / 300);
      ctx.setLineDash([12, 8]);
      ctx.beginPath(); ctx.ellipse(0, 0, p.w / 2 + 6, p.w * .55 + 6, 0, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
    }
  });

  if (snowOn) {
    seed = 99;
    ctx.fillStyle = "rgba(255,255,255,.9)";
    const n = Math.round(canvas.width * canvas.height / 9000);
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.arc(rand() * canvas.width, rand() * canvas.height, 1.5 + rand() * canvas.width / 260, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/* ---------- loading images ---------- */
function showCanvas() {
  $("drop").hidden = true; $("canvasWrap").hidden = false;
  $("download").disabled = false; $("another").disabled = false;
}

function setBase(source, w, h) {
  const k = Math.min(1, MAX_DIM / Math.max(w, h));
  const W = Math.round(w * k), H = Math.round(h * k);
  base = document.createElement("canvas"); base.width = W; base.height = H;
  base.getContext("2d").drawImage(source, 0, 0, W, H);
  canvas.width = W; canvas.height = H;
  penguins = []; sel = -1;
  showCanvas(); render();
}

function status(msg, sticky) {
  const el = $("status");
  el.textContent = msg; el.classList.toggle("show", !!msg);
  clearTimeout(status.t);
  if (msg && !sticky) status.t = setTimeout(() => el.classList.remove("show"), 3200);
}
function toast(msg) {
  const t = $("toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- model storage (IndexedDB, works on file:// too) ---------- */
const idb = () => new Promise((res, rej) => {
  const r = indexedDB.open("penguinify", 1);
  r.onupgradeneeded = () => r.result.createObjectStore("models");
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
});
const idbOp = async (mode, fn) => {
  const db = await idb();
  return new Promise((res, rej) => {
    const q = fn(db.transaction("models", mode).objectStore("models"));
    q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
  });
};
const modelGet = () => idbOp("readonly", s => s.get(MODEL_KEY)).catch(() => null);
const modelPut = buf => idbOp("readwrite", s => s.put(buf, MODEL_KEY));
const modelDel = () => idbOp("readwrite", s => s.delete(MODEL_KEY));

let session = null, usingWebGPU = false, modelReady = false, pendingDetect = false;

function modelUI(state, msg) {
  const bar = $("modelBar");
  bar.dataset.state = state;
  $("modelTitle").textContent = {
    needed: "Download the face model to detect faces",
    loading: "Downloading face model…",
    ready: "Face model ready",
    error: "Couldn’t get the face model",
  }[state];
  $("modelSub").textContent = msg || {
    needed: "SCRFD-10G, about 17 MB. Downloaded once, then kept in your browser and used offline.",
    ready: "Cached in this browser. Detection runs on your device.",
    loading: "",
  }[state] || "";
  $("modelDl").hidden = state === "loading" || state === "ready";
  $("modelPick").hidden = state === "loading" || state === "ready";
  $("modelDel").hidden = state !== "ready";
  $("modelProg").hidden = state !== "loading";
}

async function downloadModel() {
  modelUI("loading", "Starting…");
  let lastErr;
  for (const url of MODEL_URLS) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const total = +r.headers.get("content-length") || MODEL_APPROX_BYTES;
      const reader = r.body.getReader(), chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        const pct = Math.min(99, Math.round(100 * got / total));
        $("modelProg").firstElementChild.style.width = pct + "%";
        $("modelSub").textContent = `${(got / 1e6).toFixed(1)} MB of about ${(total / 1e6).toFixed(0)} MB (${pct}%)`;
      }
      const buf = await new Blob(chunks).arrayBuffer();
      if (buf.byteLength < MODEL_MIN_BYTES) throw new Error("file too small");
      await modelPut(buf);
      return onModelReady();
    } catch (e) { lastErr = e; console.warn("model download failed", url, e); }
  }
  modelUI("error", "Download failed (" + (lastErr?.message || "network") + "). Check your connection and try again, or download scrfd_10g.onnx / det_10g.onnx yourself and choose it with “Use my file”.");
  $("modelDl").hidden = false; $("modelPick").hidden = false;
}

async function onModelReady() {
  modelReady = true; session = null;
  modelUI("ready"); toast("Face model saved in your browser");
  if (pendingDetect && base) { pendingDetect = false; penguinify(1); }
}

async function loadOrt() {
  if (window.ort) return;
  await new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = ORT_BASE + "ort.min.js"; s.onload = res; s.onerror = () => rej(new Error("Couldn’t load ONNX Runtime"));
    document.head.appendChild(s);
  });
  ort.env.wasm.wasmPaths = ORT_BASE;
}

async function getSession(forceWasm) {
  if (session && !forceWasm) return session;
  const buf = await modelGet();
  if (!buf) { modelReady = false; modelUI("needed"); throw new Error("nomodel"); }
  await loadOrt();
  const bytes = () => new Uint8Array(buf.slice(0));
  if (!forceWasm && navigator.gpu) {
    try { session = await ort.InferenceSession.create(bytes(), { executionProviders: ["webgpu"] }); usingWebGPU = true; return session; }
    catch (e) { console.warn("WebGPU unavailable, using WASM", e); }
  }
  session = await ort.InferenceSession.create(bytes(), { executionProviders: ["wasm"] });
  usingWebGPU = false; return session;
}

/* ---------- SCRFD inference ---------- */
const work = document.createElement("canvas"), wg = work.getContext("2d", { willReadFrequently: true });

async function scrfdRun(sx, sy, sw, sh, scale, thr) {
  const iw = Math.max(32, Math.round(sw * scale)), ih = Math.max(32, Math.round(sh * scale));
  const PW = Math.ceil(iw / 32) * 32, PH = Math.ceil(ih / 32) * 32;
  work.width = PW; work.height = PH;
  wg.fillStyle = "#000"; wg.fillRect(0, 0, PW, PH);
  wg.imageSmoothingQuality = "high";
  wg.drawImage(base, sx, sy, sw, sh, 0, 0, iw, ih);
  const px = wg.getImageData(0, 0, PW, PH).data, n = PW * PH, data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = (px[i * 4] - 127.5) / 128; data[n + i] = (px[i * 4 + 1] - 127.5) / 128; data[2 * n + i] = (px[i * 4 + 2] - 127.5) / 128;
  }
  let out;
  try {
    const s = await getSession();
    out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", data, [1, 3, PH, PW]) });
  } catch (e) {
    if (e.message === "nomodel" || !usingWebGPU) throw e;
    console.warn("WebGPU run failed, retrying on WASM", e);
    const s = await getSession(true);
    out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", data, [1, 3, PH, PW]) });
  }

  // group outputs: scores (last dim 1), boxes (4), landmarks (10); strides 8/16/32 = largest to smallest
  const outs = Object.values(out), grp = d => outs.filter(t => t.dims[t.dims.length - 1] === d)
    .sort((a, b) => b.dims[b.dims.length - 2] - a.dims[a.dims.length - 2]);
  const S = grp(1), B = grp(4), K = grp(10), strides = [8, 16, 32], faces = [];
  S.forEach((st, si) => {
    const stride = strides[si], fw = PW / stride, sc = st.data, bb = B[si].data, kp = K[si]?.data;
    for (let i = 0; i < sc.length; i++) {
      if (sc[i] < thr) continue;
      const a = i >> 1, cx = (a % fw) * stride, cy = Math.floor(a / fw) * stride;
      const x1 = cx - bb[i * 4] * stride, y1 = cy - bb[i * 4 + 1] * stride;
      const x2 = cx + bb[i * 4 + 2] * stride, y2 = cy + bb[i * 4 + 3] * stride;
      let rot = 0;
      if (kp) {
        const dx = (kp[i * 10 + 2] - kp[i * 10]) * stride, dy = (kp[i * 10 + 3] - kp[i * 10 + 1]) * stride;
        rot = Math.atan2(dy, dx) * 180 / Math.PI; if (Math.abs(rot) > 80) rot = 0;
      }
      faces.push({ x: sx + x1 / scale, y: sy + y1 / scale, w: (x2 - x1) / scale, h: (y2 - y1) / scale, score: sc[i], rot });
    }
  });
  return nms(faces, 0.4);
}

const tick = () => new Promise(r => setTimeout(r, 0));
function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = a.w * a.h + b.w * b.h - inter, small = Math.min(a.w * a.h, b.w * b.h);
  return Math.max(inter / ua, inter / small * 0.8);
}
function nms(list, thr = 0.4) {
  list.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of list) if (!keep.some(k => iou(k, d) > thr)) keep.push(d);
  return keep;
}

async function detectFaces(level) {
  const W = base.width, H = base.height, long = Math.max(W, H);
  let faces = [];
  if (level === 1) {
    // one pass over the whole photo, scaled so small faces are still ~10px+ wide
    const scale = long > 1920 ? 1920 / long : long < 1280 ? Math.min(3, 1280 / long) : 1;
    status("Detecting faces…", true); await tick();
    faces = await scrfdRun(0, 0, W, H, scale, 0.45);
  } else {
    // deeper: overlapping 640px windows, each enlarged 1.5x, lower threshold
    const T = 640, stride = 448, scale = 1.5, tiles = [];
    const xs = [], ys = [];
    for (let x = 0; x < W - T; x += stride) xs.push(x); xs.push(Math.max(0, W - T));
    for (let y = 0; y < H - T; y += stride) ys.push(y); ys.push(Math.max(0, H - T));
    for (const y of ys) for (const x of xs) tiles.push({ x, y, w: Math.min(T, W), h: Math.min(T, H) });
    for (let i = 0; i < tiles.length; i++) {
      status(`Scanning deeper… ${i + 1} of ${tiles.length}`, true); await tick();
      const t = tiles[i], m = 0.02 * T;
      const res = await scrfdRun(t.x, t.y, t.w, t.h, scale, 0.35);
      for (const f of res) {
        // skip faces cut by an inner tile edge; a neighbouring tile sees them whole
        if ((f.x < t.x + m && t.x > 0) || (f.y < t.y + m && t.y > 0) ||
            (f.x + f.w > t.x + t.w - m && t.x + t.w < W) || (f.y + f.h > t.y + t.h - m && t.y + t.h < H)) continue;
        faces.push(f);
      }
    }
    faces = nms(faces, 0.4);
  }
  return faces.filter(f => { const r = f.w / f.h; return r > 0.5 && r < 2.2 && f.w > 6; });
}

const shuffled = a => { a = [...a]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
function leastUsedStyle() {
  const c = Object.fromEntries(STYLES.map(s => [s.id, 0]));
  penguins.forEach(p => c[p.style]++);
  const m = Math.min(...Object.values(c));
  return shuffled(STYLES.filter(s => c[s.id] === m))[0].id;
}
function assignDistinct() {
  const deck = shuffled(STYLES);
  penguins.forEach((p, i) => p.style = deck[i % deck.length].id);
}
const newStyle = () => currentStyle === "mix" ? leastUsedStyle() : currentStyle;

function faceToPenguin(f) {
  const w = Math.max(f.w, f.h * .9) * 1.85;
  return { x: f.x + f.w / 2, y: f.y + f.h / 2 - f.h * 0.05, w, base: w, rot: f.rot, flip: false, style: currentStyle === "mix" ? "classic" : currentStyle };
}

let scanning = false;
async function penguinify(level = 1) {
  if (scanning || !base) return;
  if (!(await modelGet())) {
    pendingDetect = level === 1;
    modelReady = false; modelUI("needed");
    status("Download the face model above and detection will start automatically.", true);
    const bar = $("modelBar"); bar.classList.remove("flash"); void bar.offsetWidth; bar.classList.add("flash");
    bar.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return;
  }
  scanning = true; setBusy(true);
  try {
    const faces = await detectFaces(level);
    let added = 0;
    if (level === 1) {
      penguins = faces.map(faceToPenguin); added = penguins.length;
      if (currentStyle === "mix") assignDistinct();
    } else {
      for (const f of faces) {
        const p = faceToPenguin(f);
        if (penguins.some(q => Math.hypot(q.x - p.x, q.y - p.y) < Math.max(q.w, p.w) * 0.45)) continue;
        p.style = newStyle(); penguins.push(p); added++;
      }
    }
    sel = penguins.length ? Math.min(Math.max(sel, 0), penguins.length - 1) : -1;
    render(); syncControls();
    if (level === 1) {
      status(penguins.length ? `Found ${penguins.length} face${penguins.length > 1 ? "s" : ""}. Missing someone? Try “Scan deeper”.` : "No faces found. Try “Scan deeper”, or add a penguin by hand.");
      if (penguins.length) noot();
    } else {
      status(added ? `Scan deeper found ${added} more.` : "Nothing new found.");
      if (added) noot();
    }
  } catch (e) {
    console.error(e);
    if (e.message !== "nomodel") status("Face detection failed: " + (e.message || e) + ". You can still add penguins by hand.");
  } finally { scanning = false; setBusy(false); }
}

function setBusy(b) {
  ["deeper", "download", "chaos", "add", "shuffle"].forEach(id => $(id).disabled = b || (id === "download" && !base));
}

/* model bar wiring */
$("modelDl").onclick = downloadModel;
$("modelFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const buf = await f.arrayBuffer();
  if (buf.byteLength < MODEL_MIN_BYTES) { modelUI("error", "That file looks too small to be the SCRFD model."); return; }
  await modelPut(buf); onModelReady();
};
$("modelDel").onclick = async () => { await modelDel(); session = null; modelReady = false; modelUI("needed"); toast("Model removed from this browser"); };
modelGet().then(b => { if (b) { modelReady = true; modelUI("ready"); } else modelUI("needed"); });

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) { toast("That doesn’t look like an image."); return; }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { setBase(img, img.naturalWidth, img.naturalHeight); URL.revokeObjectURL(url); penguinify(1); };
  img.onerror = () => toast("Couldn’t read that image.");
  img.src = url;
}

/* ---------- controls ---------- */
function buildChips() {
  const wrap = $("chips");
  const mix = document.createElement("button");
  mix.type = "button"; mix.className = "chip mix"; mix.dataset.id = "mix";
  mix.title = "A different penguin for every face";
  mix.innerHTML = `<span class="mixart">🎲</span><span>Mix</span>`;
  mix.onclick = () => { currentStyle = "mix"; if (penguins.length) assignDistinct(); syncControls(); render(); };
  wrap.appendChild(mix);
  STYLES.forEach(s => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.dataset.id = s.id;
    b.innerHTML = `<img alt="" src="${s.url}"><span>${s.name}</span>`;
    b.onclick = () => {
      if (sel >= 0) penguins[sel].style = s.id; else currentStyle = s.id;
      syncControls(); render();
    };
    wrap.appendChild(b);
  });
  $("dropArt").innerHTML = `<img alt="Penguin" src="${STYLES[0].url}">`;
}

function syncControls() {
  const p = penguins[sel];
  document.querySelectorAll(".chip").forEach(c =>
    c.setAttribute("aria-pressed", String(c.dataset.id === (p ? p.style : currentStyle))));
  ["size", "rot", "flip", "del", "useAll"].forEach(id => $(id).disabled = !p);
  $("selHint").hidden = !!p && penguins.length > 0 ? true : false;
  if (p) {
    const face0 = p.base || (p.base = p.w);
    $("size").value = Math.round(p.w / face0 * 100);
    $("rot").value = Math.round(p.rot);
  }
}

$("size").oninput = e => { const p = penguins[sel]; if (!p) return; p.w = (p.base || p.w) * e.target.value / 100; render(); };
$("rot").oninput = e => { const p = penguins[sel]; if (!p) return; p.rot = +e.target.value; render(); };
$("useAll").onclick = () => { const p = penguins[sel]; if (p) { penguins.forEach(q => q.style = p.style); render(); } };
$("shuffle").onclick = () => { if (!penguins.length) { toast("Add a penguin first."); return; } currentStyle = "mix"; assignDistinct(); syncControls(); render(); noot(); };
$("flip").onclick = () => { const p = penguins[sel]; if (p) { p.flip = !p.flip; render(); } };
$("del").onclick = () => { if (sel >= 0) { penguins.splice(sel, 1); sel = penguins.length ? 0 : -1; syncControls(); render(); } };
$("add").onclick = () => {
  if (!base) return;
  const w = Math.min(base.width, base.height) * .3;
  penguins.push({ x: base.width / 2, y: base.height / 2, w, base: w, rot: 0, flip: false, style: newStyle() });
  sel = penguins.length - 1; syncControls(); render();
};
$("snow").onclick = e => { snowOn = !snowOn; e.currentTarget.setAttribute("aria-pressed", snowOn); render(); };
$("fish").onclick = e => { fishOn = !fishOn; e.currentTarget.setAttribute("aria-pressed", fishOn); render(); };
$("chaos").onclick = () => {
  if (!penguins.length) { toast("Add a penguin first."); return; }
  penguins.forEach(p => {
    p.style = STYLES[Math.floor(Math.random() * STYLES.length)].id;
    p.rot += (Math.random() - .5) * 50;
    p.flip = Math.random() > .5;
    p.w *= .85 + Math.random() * .5;
    p.base = p.base || p.w;
  });
  canvas.classList.remove("shake"); void canvas.offsetWidth; canvas.classList.add("shake");
  noot(); noot(true); syncControls(); render();
};
$("download").onclick = () => {
  render(true);
  canvas.toBlob(blob => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "penguinified.png";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    render(); toast("Saved as penguinified.png");
  }, "image/png");
};
$("deeper").onclick = () => base && penguinify(2);
$("another").onclick = () => {
  base = null; penguins = []; sel = -1;
  $("canvasWrap").hidden = true; $("drop").hidden = false;
  $("download").disabled = true; $("another").disabled = true;
  $("file").value = ""; syncControls();
};

/* ---------- drag to move / select ---------- */
let drag = null;
function pos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) * canvas.width / r.width, y: (e.clientY - r.top) * canvas.height / r.height };
}
canvas.addEventListener("pointerdown", e => {
  const m = pos(e);
  let hit = -1;
  for (let i = penguins.length - 1; i >= 0; i--) {
    const p = penguins[i];
    if (Math.hypot(m.x - p.x, m.y - p.y) < p.w * .5) { hit = i; break; }
  }
  sel = hit; syncControls(); render();
  if (hit >= 0) {
    drag = { dx: penguins[hit].x - m.x, dy: penguins[hit].y - m.y };
    canvas.setPointerCapture(e.pointerId); canvas.classList.add("drag");
  }
});
canvas.addEventListener("pointermove", e => {
  if (!drag || sel < 0) return;
  const m = pos(e); penguins[sel].x = m.x + drag.dx; penguins[sel].y = m.y + drag.dy; render();
});
const endDrag = () => { drag = null; canvas.classList.remove("drag"); };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("wheel", e => {
  if (sel < 0) return; e.preventDefault();
  penguins[sel].w *= e.deltaY < 0 ? 1.05 : .95; penguins[sel].base = penguins[sel].base || penguins[sel].w; render();
}, { passive: false });

/* ---------- input: file, drop, paste, demo ---------- */
$("file").onchange = e => loadFile(e.target.files[0]);
const dropEl = $("stage");
["dragenter", "dragover"].forEach(t => dropEl.addEventListener(t, e => { e.preventDefault(); $("drop").classList.add("over"); }));
["dragleave", "drop"].forEach(t => dropEl.addEventListener(t, e => { e.preventDefault(); $("drop").classList.remove("over"); }));
dropEl.addEventListener("drop", e => loadFile(e.dataTransfer.files[0]));
window.addEventListener("paste", e => {
  const f = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith("image/"));
  if (f) loadFile(f);
});
$("demoBtn").onclick = () => {
  const c = document.createElement("canvas"); c.width = 1200; c.height = 800;
  const g = c.getContext("2d");
  const grd = g.createLinearGradient(0, 0, 0, 800);
  grd.addColorStop(0, "#9fd3ee"); grd.addColorStop(.6, "#eaf6fb"); grd.addColorStop(1, "#ffffff");
  g.fillStyle = grd; g.fillRect(0, 0, 1200, 800);
  setBase(c, 1200, 800); $("add").click();
  status("Blank ice sheet. Add as many penguins as you like.");
};

/* ---------- webcam ---------- */
let stream = null;
$("camBtn").onclick = async () => {
  $("camModal").hidden = false; $("camErr").textContent = "";
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } } });
    $("video").srcObject = stream; await $("video").play();
  } catch (e) {
    $("camErr").textContent = "Camera unavailable. Allow camera access, and note that browsers need https or localhost for this.";
  }
};
function closeCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; $("camModal").hidden = true;
}
$("camClose").onclick = closeCam;
$("snap").onclick = () => {
  const v = $("video");
  if (!v.videoWidth) return;
  const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
  const g = c.getContext("2d"); g.translate(c.width, 0); g.scale(-1, 1); g.drawImage(v, 0, 0);
  closeCam(); setBase(c, c.width, c.height); penguinify(1);
};

/* ---------- noot noot ---------- */
let audio;
function noot(high) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
    o.type = "sawtooth";
    o.frequency.setValueAtTime(high ? 620 : 520, t);
    o.frequency.exponentialRampToValueAtTime(high ? 380 : 300, t + .16);
    g.gain.setValueAtTime(.0001, t); g.gain.exponentialRampToValueAtTime(.12, t + .02);
    g.gain.exponentialRampToValueAtTime(.0001, t + .22);
    o.connect(g).connect(audio.destination); o.start(t); o.stop(t + .25);
  } catch (_) {}
}

/* ---------- init ---------- */
loadStyles().then(() => { buildChips(); syncControls(); });
