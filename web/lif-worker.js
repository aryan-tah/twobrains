/* In-browser leaky integrate-and-fire simulation of a whole connectome (event-driven, CSC by presynaptic neuron).
   Same model and parameters as twobrains/lif.py. Only "hot" neurons (non-resting state) are integrated each step,
   so a 139k-neuron brain runs an assay in well under a second. Runs in a Web Worker. */
"use strict";
let N = 0, nnz = 0, indptr, indices, wdata, P, keep;

function load(buf) {
  const dv = new DataView(buf); const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4)); if (magic !== "TBN1") throw new Error("bad net file");
  N = dv.getUint32(4, true); nnz = dv.getUint32(8, true); let o = 16;
  indptr = new Int32Array(buf, o, N + 1); o += 4 * (N + 1);
  indices = new Int32Array(buf, o, nnz); o += 4 * nnz;
  wdata = new Int16Array(buf, o, nnz); o += 2 * nnz;
  const pos = new Float32Array(buf.slice(o, o + N * 12));
  keep = new Uint8Array(N).fill(1);
  return pos;
}

function simulate(inp, hz, tMs, seed, binMs) {
  const dt = P.dt, steps = Math.round(tMs / dt), dm = Math.exp(-dt / P.tau_m), ds = Math.exp(-dt / P.tau_syn), da = Math.exp(-dt / P.tau_adapt);
  const vrest = P.v_rest, vreset = P.v_reset, vth = P.v_th, trfc = P.t_rfc, badapt = P.b_adapt, gcap = P.g_cap, wsyn = P.w_syn;
  const v = new Float32Array(N).fill(vrest), g = new Float32Array(N), a = new Float32Array(N), rfc = new Float32Array(N), cnt = new Uint16Array(N);
  const hot = new Uint8Array(N); let hotList = new Int32Array(1 << 16), nHot = 0;
  const touch = i => { if (!hot[i]) { hot[i] = 1; if (nHot === hotList.length) { const t = new Int32Array(nHot * 2); t.set(hotList); hotList = t; } hotList[nHot++] = i; } };
  const dsteps = Math.max(1, Math.round(P.t_dly / dt)); const queue = []; for (let i = 0; i < dsteps; i++) queue.push([]); let qi = 0;
  let s = (seed >>> 0) || 1; const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const pIn = hz * dt / 1000, inpK = inp.filter(i => keep[i]);
  const nb = Math.ceil(tMs / binMs), bins = []; for (let b = 0; b < nb; b++) bins.push([]);
  let t = 0;
  for (let step = 0; step < steps; step++) {
    const q = queue[qi];
    for (let k = 0; k < q.length; k++) { const pre = q[k]; for (let j = indptr[pre], e = indptr[pre + 1]; j < e; j++) { const post = indices[j]; if (keep[post]) { g[post] += wsyn * wdata[j]; touch(post); } } }
    for (let k = 0; k < inpK.length; k++) if (rnd() < pIn) { v[inpK[k]] = 1e3; touch(inpK[k]); }
    const out = [], bin = bins[Math.min(nb - 1, Math.floor(t / binMs))]; let w = 0;
    for (let h = 0; h < nHot; h++) {
      const i = hotList[h]; let gi = g[i]; if (gi > gcap) gi = gcap; else if (gi < -gcap) gi = -gcap;
      let vi = v[i];
      if (t >= rfc[i]) { vi = vrest + (vi - vrest) * dm + gi * (1 - dm); if (badapt) a[i] *= da;
        if (vi >= vth + a[i]) { vi = vreset; rfc[i] = t + trfc; a[i] += badapt; out.push(i); cnt[i]++; bin.push(i); } }
      v[i] = vi; gi *= ds; g[i] = gi;
      const still = Math.abs(vi - vrest) > 1e-3 || Math.abs(gi) > 1e-3 || a[i] > 1e-2 || t < rfc[i];
      if (still) hotList[w++] = i; else hot[i] = 0;
    }
    nHot = w; queue[qi] = out; qi = (qi + 1) % dsteps; t += dt;
  }
  return { cnt, bins };
}

onmessage = e => {
  const m = e.data;
  if (m.type === "load") { const pos = load(m.buf); P = m.params; postMessage({ type: "loaded", N, nnz, pos }, [pos.buffer]); }
  else if (m.type === "params") { P = m.params; postMessage({ type: "paramsok" }); }
  else if (m.type === "keep") { keep = m.keep; let alive = 0, syn = 0; for (let i = 0; i < N; i++) if (keep[i]) { alive++; for (let j = indptr[i], en = indptr[i + 1]; j < en; j++) if (keep[indices[j]]) syn += Math.abs(wdata[j]); } postMessage({ type: "keepok", alive, syn }); }
  else if (m.type === "assay") {
    const t0 = performance.now(); const { cnt, bins } = simulate(m.inp, m.hz, m.tMs || 300, m.seed || 1, m.binMs || 10);
    let score; if (m.mode === "responders") { let r = 0; for (const i of m.out) if (cnt[i] > 0) r++; score = r; } else { let s = 0; for (const i of m.out) s += cnt[i]; score = s / m.out.length / ((m.tMs || 300) / 1000); }
    let fired = 0; for (let i = 0; i < N; i++) if (cnt[i]) fired++;
    postMessage({ type: "result", key: m.key, score, fired, bins: bins.map(b => Int32Array.from(b)), ms: performance.now() - t0 });
  }
};
