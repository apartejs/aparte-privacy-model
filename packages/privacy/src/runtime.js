// @aparte/secrets — finds passwords, API keys, tokens and private keys in text,
// anywhere JavaScript runs (browser, Node, workers). No dependency, no network.
//
// Reads the `.bin` files published at https://huggingface.co/apartejs/aparte-secrets
// (format 1, the same container as aparte-titler): a JSON header, the BPE merges
// and the quantized weights.
//
//   import { Secrets } from "@aparte/secrets";
//   const secrets = new Secrets(await fetch(modelUrl).then((r) => r.arrayBuffer()));
//   secrets.detect("my db password is Hunter2024!");
//   // -> [{ start: 18, end: 29, text: "Hunter2024!", subtype: "password", score: 0.69 }]
//
// The Python reference implementation (python/aparte_secrets/reader.py) gives the
// same spans; the test in test/ checks it text by text.

// Python's str.isspace(): JS's \s adds U+FEFF and lacks \x1c-\x1f and \x85
const SPACE = "\\t\\n\\v\\f\\r \\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const ONE_SPACE = new RegExp("^[" + SPACE + "]$", "u");
const EDGES = new RegExp("^[" + SPACE + "]+|[" + SPACE + "]+$", "gu");

// ------------------------------------------------------------ file reading

function fp16(u) {
  const s = u & 0x8000 ? -1 : 1, e = (u >> 10) & 0x1f, m = u & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// bit stream, least significant bit first: n signed integers of `bits` bits
function unpackBits(bytes, n, bits) {
  const out = new Int32Array(n), offset = 1 << (bits - 1);
  if (bits === 8) { for (let i = 0; i < n; i++) out[i] = bytes[i] - 128; return out; }
  let pos = 0;
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let k = 0; k < bits; k++, pos++) if (bytes[pos >> 3] & (1 << (pos & 7))) v |= 1 << k;
    out[i] = v - offset;
  }
  return out;
}

/** Parse a `.bin` model file. Returns { header, merges, weights }. */
export function readModel(buffer) {
  const bytes = new Uint8Array(buffer), view = new DataView(buffer);
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "LMAP") throw new Error("not an aparte model file");
  let p = 4;
  const headerLength = view.getUint32(p, true); p += 4;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(p, p + headerLength))); p += headerLength;
  if ((header.format ?? 1) !== 1) throw new Error("model format " + header.format + " is not supported by this runtime (format 1)");
  if (header.task !== "secrets" && header.task !== "privacy") throw new Error("not an aparte model (task " + header.task + ")");
  const mergesLength = view.getUint32(p, true); p += 4;
  const merges = new Uint16Array(buffer.slice(p, p + mergesLength)); p += mergesLength;
  const weights = {};
  for (const t of header.tensors) {
    const shape = t.shape, count = shape.reduce((a, b) => a * b, 1);
    const w = new Float32Array(count);
    if (t.type === "f32") {
      for (let i = 0; i < count; i++) w[i] = view.getFloat32(p + 4 * i, true);
    } else if (t.type === "f16") {
      for (let i = 0; i < count; i++) w[i] = fp16(view.getUint16(p + 2 * i, true));
    } else {
      // quantized rows: packed integers, then one fp16 scale per row
      const bits = parseInt(t.type.slice(1)), rows = shape[0], cols = count / rows;
      const packedLength = Math.ceil((count * bits) / 8);
      const q = unpackBits(bytes.subarray(p, p + packedLength), count, bits);
      for (let r = 0; r < rows; r++) {
        const scale = fp16(view.getUint16(p + packedLength + 2 * r, true));
        for (let c = 0; c < cols; c++) w[r * cols + c] = q[r * cols + c] * scale;
      }
    }
    weights[t.name] = { data: w, shape };
    p += t.bytes;
  }
  return { header, merges, weights };
}

// ------------------------------------------------------- units, tokenizer

class Tokenizer {
  constructor(header, merges) {
    this.byteIds = header.byte_ids;          // token id of each of the 256 bytes
    this.base = header.merges_base;          // merge k produces token id base + k
    this.rank = new Map();
    for (let k = 0; k < merges.length / 2; k++) this.rank.set(merges[2 * k] * 65536 + merges[2 * k + 1], k);
    this.units = new RegExp(header.pre_tokenizer.pattern, "gu");
    this.encoder = new TextEncoder();
  }

  /** Token ids, [start, end) offsets in UTF-16 units, the unit of each token, unit offsets. */
  encode(text) {
    const ids = [], offsets = [], unitOf = [], units = [];
    for (const m of text.matchAll(this.units)) {
      const start = m.index, piece = m[0], u = units.length;
      units.push([start, start + piece.length]);
      const charOfByte = [], endOfByte = [], tokens = [];
      let i = 0;
      for (const ch of piece) {                 // per code point
        const n = ch.length;                    // 1 or 2 UTF-16 units
        for (const b of this.encoder.encode(ch)) { tokens.push([this.byteIds[b], 1]); charOfByte.push(i); endOfByte.push(i + n); }
        i += n;
      }
      for (;;) {                                // merge the lowest-ranked pair first
        let best = -1, at = -1;
        for (let j = 0; j < tokens.length - 1; j++) {
          const r = this.rank.get(tokens[j][0] * 65536 + tokens[j + 1][0]);
          if (r !== undefined && (best < 0 || r < best)) { best = r; at = j; }
        }
        if (best < 0) break;
        tokens.splice(at, 2, [this.base + best, tokens[at][1] + tokens[at + 1][1]]);
      }
      let pos = 0;
      for (const [id, n] of tokens) {
        ids.push(id);
        offsets.push([start + charOfByte[pos], start + endOfByte[pos + n - 1]]);
        unitOf.push(u);
        pos += n;
      }
    }
    return { ids, offsets, unitOf, units };
  }
}

// length class of a unit: log2 scale of its non-blank code points (0 for blank)
function lengthClass(piece, nClasses) {
  let n = 0;
  for (const _ of piece.replace(EDGES, "")) n++;   // eslint-disable-line no-unused-vars
  return Math.min(nClasses - 1, n === 0 ? 0 : 32 - Math.clz32(n));
}

// ------------------------------------------------------------------ model

function layerNorm(x, L, D, g, b, out) {
  for (let t = 0; t < L; t++) {
    let mean = 0; for (let d = 0; d < D; d++) mean += x[t * D + d]; mean /= D;
    let v = 0; for (let d = 0; d < D; d++) { const e = x[t * D + d] - mean; v += e * e; } v /= D;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let d = 0; d < D; d++) out[t * D + d] = (x[t * D + d] - mean) * inv * g[d] + b[d];
  }
  return out;
}

function erf(x) {  // Abramowitz-Stegun 7.1.26, error < 1.5e-7
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

// y[L x N] = x[L x K] · W[N x K]^T + b[N]   (torch layout: rows are outputs)
function linear(x, L, K, W, b, N, out) {
  for (let t = 0; t < L; t++) for (let n = 0; n < N; n++) {
    let s = b ? b[n] : 0;
    for (let k = 0; k < K; k++) s += x[t * K + k] * W[n * K + k];
    out[t * N + n] = s;
  }
  return out;
}

class Model {
  constructor(header, weights) {
    this.h = header; this.w = Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.data]));
    this.D = header.dim; this.H = header.heads; this.embDim = header.dim_emb;
    this.C = header.tensors.find((t) => t.name === "head.weight").shape[0];
  }

  /** `C` logits per token for one window; every token attends to every token of the window. */
  logits(ids, lengths) {
    const { w, D, H, h, C } = this, L = ids.length, Dh = D / H;
    const x = new Float32Array(L * D), e = new Float32Array(L * this.embDim);
    for (let t = 0; t < L; t++) for (let d = 0; d < this.embDim; d++) e[t * this.embDim + d] = w["emb_tok.weight"][ids[t] * this.embDim + d];
    linear(e, L, this.embDim, w["proj_emb.weight"], null, D, x);
    for (let t = 0; t < L; t++) {
      for (let d = 0; d < D; d += 2) {           // sinusoidal positions, relative to the window
        const f = Math.exp(d * (-Math.log(10000) / D));
        x[t * D + d] += Math.sin(t * f); x[t * D + d + 1] += Math.cos(t * f);
      }
      for (let d = 0; d < D; d++) x[t * D + d] += w["emb_len.weight"][lengths[t] * D + d];
    }
    const y = new Float32Array(L * D), qkv = new Float32Array(L * 3 * D), att = new Float32Array(L * D);
    const f1 = new Float32Array(L * 4 * D), f2 = new Float32Array(L * D), s = new Float32Array(L);
    for (let c = 0; c < h.layers; c++) {
      const k = "blocks." + c + ".";
      layerNorm(x, L, D, w[k + "n1.weight"], w[k + "n1.bias"], y);
      linear(y, L, D, w[k + "att.qkv.weight"], w[k + "att.qkv.bias"], 3 * D, qkv);
      att.fill(0);
      for (let head = 0; head < H; head++) {
        const o = head * Dh;
        for (let i = 0; i < L; i++) {
          let max = -Infinity;
          for (let j = 0; j < L; j++) {
            let d = 0;
            for (let q = 0; q < Dh; q++) d += qkv[i * 3 * D + o + q] * qkv[j * 3 * D + D + o + q];
            s[j] = d / Math.sqrt(Dh); if (s[j] > max) max = s[j];
          }
          let sum = 0;
          for (let j = 0; j < L; j++) { s[j] = Math.exp(s[j] - max); sum += s[j]; }
          for (let j = 0; j < L; j++) {
            const p = s[j] / sum;
            for (let q = 0; q < Dh; q++) att[i * D + o + q] += p * qkv[j * 3 * D + 2 * D + o + q];
          }
        }
      }
      linear(att, L, D, w[k + "att.output.weight"], w[k + "att.output.bias"], D, f2);
      for (let i = 0; i < L * D; i++) x[i] += f2[i];
      layerNorm(x, L, D, w[k + "n2.weight"], w[k + "n2.bias"], y);
      linear(y, L, D, w[k + "ff.net.0.weight"], w[k + "ff.net.0.bias"], 4 * D, f1);
      for (let i = 0; i < L * 4 * D; i++) f1[i] = gelu(f1[i]);
      linear(f1, L, 4 * D, w[k + "ff.net.2.weight"], w[k + "ff.net.2.bias"], D, f2);
      for (let i = 0; i < L * D; i++) x[i] += f2[i];
    }
    layerNorm(x, L, D, w["norm.weight"], w["norm.bias"], y);
    return linear(y, L, D, w["head.weight"], w["head.bias"], C, new Float32Array(L * C));
  }
}

function windows(n, size, stride) {
  if (n <= size) return [0];
  const starts = [];
  for (let d = 0; d <= n - size; d += stride) starts.push(d);
  if (starts[starts.length - 1] + size < n) starts.push(n - size);
  return starts;
}

// --------------------------------------------------------------- decoding

// allowed BIOES moves, one subtype per span: state k > 0 is subtype (k-1)/4, tag (k-1)%4 (B, I, E, S)
function transitions(C) {
  const tag = (k) => (k === 0 ? -1 : (k - 1) % 4), sub = (k) => (k === 0 ? -1 : Math.floor((k - 1) / 4));
  const opens = (k) => k === 0 || tag(k) === 0 || tag(k) === 3;
  const M = new Float64Array(C * C).fill(-1e9);
  for (let j = 0; j < C; j++) for (let k = 0; k < C; k++) {
    const ok = tag(j) === 0 || tag(j) === 1 ? (tag(k) === 1 || tag(k) === 2) && sub(k) === sub(j) : opens(k);
    if (ok) M[j * C + k] = 0;
  }
  const start = Float64Array.from({ length: C }, (_, k) => (opens(k) ? 0 : -1e9));
  const end = Float64Array.from({ length: C }, (_, k) => (k === 0 || tag(k) === 2 || tag(k) === 3 ? 0 : -1e9));
  return { M, start, end };
}

function viterbi(logp, L, C, { M, start, end }) {
  let score = new Float64Array(C);
  for (let k = 0; k < C; k++) score[k] = logp[k] + start[k];
  const back = new Int32Array(L * C);
  for (let t = 1; t < L; t++) {
    const next = new Float64Array(C);
    for (let k = 0; k < C; k++) {
      let best = -Infinity, arg = 0;
      for (let j = 0; j < C; j++) { const v = score[j] + M[j * C + k]; if (v > best) { best = v; arg = j; } }
      back[t * C + k] = arg; next[k] = best + logp[t * C + k];
    }
    score = next;
  }
  let k = 0, best = -Infinity;
  for (let j = 0; j < C; j++) if (score[j] + end[j] > best) { best = score[j] + end[j]; k = j; }
  const path = new Int32Array(L); path[L - 1] = k;
  for (let t = L - 1; t > 0; t--) { k = back[t * C + k]; path[t - 1] = k; }
  return path;
}

export class Secrets {
  /** @param {ArrayBuffer} buffer the contents of a `.bin` model file */
  constructor(buffer) {
    const { header, merges, weights } = readModel(buffer);
    this.header = header; this.tokenizer = new Tokenizer(header, merges); this.model = new Model(header, weights);
    this.subtypes = header.labels.subtypes;
    this.C = 1 + 4 * this.subtypes.length;
    this.trans = transitions(this.C);
  }

  /**
   * Every secret found in the text: [{ start, end, text, subtype, score }] in text order.
   * Offsets are UTF-16 indices (`text.slice(start, end)`). `score` is the model's confidence,
   * between 0 and 1; the model reports raw spans, filtering is up to the caller (`options`).
   *
   * @param {string} text
   * @param {{ minScore?: number, ignore?: Iterable<string | RegExp> }} [options]
   */
  detect(text, options) {
    const h = this.header, C = this.C;
    const { ids, offsets, unitOf, units } = this.tokenizer.encode(text);
    if (!ids.length) return [];
    const unitLen = units.map(([a, b]) => lengthClass(text.slice(a, b), h.unit_length_classes));
    const n = ids.length, total = new Float64Array(n * C), count = new Float64Array(n);
    for (const d of windows(n, h.window, h.stride)) {
      const end = Math.min(d + h.window, n), L = end - d;
      const z = this.model.logits(ids.slice(d, end), unitOf.slice(d, end).map((u) => unitLen[u]));
      // one decision per unit inside the window: mean of its tokens' logits, then log-softmax
      const u0 = unitOf[d], nu = unitOf[end - 1] - u0 + 1;
      const sums = new Float64Array(nu * C), cnt = new Float64Array(nu);
      for (let t = 0; t < L; t++) { const u = unitOf[d + t] - u0; cnt[u]++; for (let c = 0; c < C; c++) sums[u * C + c] += z[t * C + c]; }
      for (let t = 0; t < L; t++) {
        const u = unitOf[d + t] - u0;
        let max = -Infinity;
        for (let c = 0; c < C; c++) max = Math.max(max, sums[u * C + c] / cnt[u]);
        let s = 0;
        for (let c = 0; c < C; c++) s += Math.exp(sums[u * C + c] / cnt[u] - max);
        const lse = max + Math.log(s);
        for (let c = 0; c < C; c++) total[(d + t) * C + c] += sums[u * C + c] / cnt[u] - lse;
        count[d + t]++;
      }
    }
    // mean log-probabilities per unit (units that have tokens, in text order)
    const kept = [], index = new Map();
    for (const u of unitOf) if (!index.has(u)) { index.set(u, kept.length); kept.push(u); }
    const K = kept.length, lp = new Float64Array(K * C), nk = new Float64Array(K);
    for (let t = 0; t < n; t++) {
      const i = index.get(unitOf[t]); nk[i]++;
      for (let c = 0; c < C; c++) lp[i * C + c] += total[t * C + c] / count[t];
    }
    for (let i = 0; i < K; i++) for (let c = 0; c < C; c++) lp[i * C + c] /= nk[i];
    const path = viterbi(lp, K, C, this.trans);
    const spans = [];
    let first = -1;
    for (let i = 0; i < K; i++) {
      const e = path[i];
      if (e === 0) { first = -1; continue; }
      const tag = (e - 1) % 4;
      if (tag === 0 || tag === 3) first = i;
      if ((tag === 2 || tag === 3) && first >= 0) {
        let a = units[kept[first]][0];
        const b = units[kept[i]][1];
        while (a < b && ONE_SPACE.test(text[a])) a++;
        let score = Infinity;
        for (let j = first; j <= i; j++) score = Math.min(score, 1 - Math.exp(lp[j * C]));
        spans.push({ start: a, end: b, text: text.slice(a, b), subtype: this.subtypes[Math.floor((e - 1) / 4)],
                     score: Math.round(score * 1e4) / 1e4 });
        first = -1;
      }
    }
    return filter(spans, options);
  }

  /**
   * Replace every secret with a placeholder, reversibly: `{ text, secrets }`, where
   * `secrets` maps each placeholder back to its value (keep it on the client).
   *
   * @param {string} text
   * @param {{ minScore?: number, ignore?: Iterable<string | RegExp>, placeholder?: (subtype: string, n: number) => string }} [options]
   */
  mask(text, options) {
    const make = options?.placeholder ?? ((subtype, n) => `[${subtype.toUpperCase()}_${n}]`);
    const found = this.detect(text, options), secrets = {}, seen = new Map();
    // a value found once is masked everywhere it appears: leaving a second copy in clear would defeat the
    // purpose (the model may only flag the occurrence whose context gives it away)
    const spans = [...found];
    for (const value of new Set(found.map((s) => s.text))) {
      const subtype = found.find((s) => s.text === value).subtype;
      for (let i = text.indexOf(value); i >= 0; i = text.indexOf(value, i + value.length)) {
        if (!spans.some((s) => s.start < i + value.length && i < s.end)) spans.push({ start: i, end: i + value.length, text: value, subtype });
      }
    }
    spans.sort((a, b) => a.start - b.start);
    let out = "", pos = 0, n = 0;
    for (const s of spans) {
      if (s.start < pos) continue;
      let ph = seen.get(s.text);
      if (!ph) { ph = make(s.subtype, ++n); seen.set(s.text, ph); secrets[ph] = s.text; }
      out += text.slice(pos, s.start) + ph; pos = s.end;
    }
    return { text: out + text.slice(pos), secrets };
  }

  /** Put the values back: the inverse of `mask` (e.g. on an AI's answer). */
  unmask(text, secrets) {
    let out = text;
    for (const [ph, value] of Object.entries(secrets)) out = out.split(ph).join(value);
    return out;
  }
}

// the caller's policy, applied after the model: never part of the model itself
function filter(spans, options) {
  if (!options) return spans;
  const min = options.minScore ?? 0;
  const ignore = [...(options.ignore ?? [])];
  return spans.filter((s) => s.score >= min && !ignore.some((r) => (typeof r === "string" ? s.text === r : r.test(s.text))));
}

/** Convenience: build a Secrets detector from an ArrayBuffer (or a Promise of one). */
export async function loadSecrets(buffer) { return new Secrets(await buffer); }
