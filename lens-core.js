// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Intuition Labs LLC
//
// lens-core — the whole lens-drop pipeline as one dependency-free ES module,
// shared by lens.html (browser) and selftest-lens.mjs (node). No network, no
// DOM: bytes in, verdicts and receipts out.
//
// Two input shapes:
//   - a torch .pt Jacobian-lens file (Neuronpedia's pre-fit format: zip with
//     STORED entries, pickle protocol-2 subset, fp16 storages), or
//   - a plain-JSON lens {d_model, vectors: number[][]}.
//
// The cone math (pursuit / coneDistance / deltaMu) is the same original
// algorithm as @intuitionlabs/jspace's cones.ts, ported to plain JS. The
// exact D_mu formulation is the Anthropic paper's; ours follows it in spirit.

// ---------- fp16 ----------

export function halfToNumber(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

// ---------- zip (central directory, STORED only) ----------

export function readZipEntries(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const u32 = (o) => dv.getUint32(o, true);
  const u16 = (o) => dv.getUint16(o, true);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= 0; i--) if (u32(i) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('not a zip: EOCD signature not found');
  const count = u16(eocd + 10);
  let off = u32(eocd + 16);
  const entries = new Map();
  for (let n = 0; n < count; n++) {
    if (u32(off) !== 0x02014b50) throw new Error('bad central directory entry');
    const method = u16(off + 10);
    const usize = u32(off + 24);
    const nameLen = u16(off + 28);
    const extraLen = u16(off + 30);
    const commentLen = u16(off + 32);
    const localOff = u32(off + 42);
    const name = new TextDecoder().decode(bytes.subarray(off + 46, off + 46 + nameLen));
    entries.set(name, { method, usize, localOff });
    off += 46 + nameLen + extraLen + commentLen;
  }
  return {
    names: () => [...entries.keys()],
    read(name) {
      const e = entries.get(name);
      if (!e) throw new Error(`zip entry not found: ${name}`);
      if (e.method !== 0) throw new Error(`entry ${name}: only STORED zip entries are supported`);
      const lo = e.localOff;
      if (u32(lo) !== 0x04034b50) throw new Error('bad local file header');
      const start = lo + 30 + u16(lo + 26) + u16(lo + 28);
      return bytes.subarray(start, start + e.usize);
    },
  };
}

// ---------- pickle (protocol-2 subset used by torch lens saves) ----------

export function unpickle(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const td = new TextDecoder();
  const stack = [];
  const memo = new Map();
  const marks = [];
  let p = 0;
  const popToMark = () => stack.splice(marks.pop());
  while (p < bytes.length) {
    const op = bytes[p++];
    switch (op) {
      case 0x80: p += 1; break;                       // PROTO
      case 0x7d: stack.push({}); break;               // EMPTY_DICT
      case 0x5d: stack.push([]); break;               // EMPTY_LIST
      case 0x29: stack.push([]); break;               // EMPTY_TUPLE
      case 0x28: marks.push(stack.length); break;     // MARK
      case 0x71: memo.set(bytes[p++], stack[stack.length - 1]); break; // BINPUT
      case 0x68: stack.push(memo.get(bytes[p++])); break;              // BINGET
      case 0x58: {                                    // BINUNICODE
        const len = dv.getUint32(p, true); p += 4;
        stack.push(td.decode(bytes.subarray(p, p + len))); p += len;
        break;
      }
      case 0x4b: stack.push(bytes[p++]); break;                        // BININT1
      case 0x4d: stack.push(dv.getUint16(p, true)); p += 2; break;     // BININT2
      case 0x4a: stack.push(dv.getInt32(p, true)); p += 4; break;      // BININT
      case 0x63: {                                    // GLOBAL
        let nl1 = p; while (bytes[nl1] !== 0x0a) nl1++;
        let nl2 = nl1 + 1; while (bytes[nl2] !== 0x0a) nl2++;
        const mod = td.decode(bytes.subarray(p, nl1));
        const name = td.decode(bytes.subarray(nl1 + 1, nl2));
        p = nl2 + 1;
        stack.push({ __global__: `${mod}.${name}` });
        break;
      }
      case 0x74: stack.push(popToMark()); break;      // TUPLE
      case 0x85: stack.push([stack.pop()]); break;    // TUPLE1
      case 0x86: { const b = stack.pop(), a = stack.pop(); stack.push([a, b]); break; }
      case 0x87: { const c = stack.pop(), b = stack.pop(), a = stack.pop(); stack.push([a, b, c]); break; }
      case 0x51: {                                    // BINPERSID
        const pid = stack.pop();
        if (!Array.isArray(pid) || pid[0] !== 'storage') throw new Error('unsupported persistent id');
        stack.push({ __storage__: { cls: pid[1].__global__, key: pid[2], numel: pid[4] } });
        break;
      }
      case 0x89: stack.push(false); break;            // NEWFALSE
      case 0x88: stack.push(true); break;             // NEWTRUE
      case 0x52: {                                    // REDUCE
        const args = stack.pop();
        const fn = stack.pop();
        if (fn.__global__ === 'collections.OrderedDict') { stack.push({}); break; }
        if (fn.__global__ === 'torch._utils._rebuild_tensor_v2') {
          const [storage, offset, shape, stride] = args;
          stack.push({ __tensor__: { storage: storage.__storage__, offset, shape, stride } });
          break;
        }
        throw new Error(`unsupported REDUCE target: ${fn.__global__}`);
      }
      case 0x75: {                                    // SETITEMS
        const items = popToMark();
        const dict = stack[stack.length - 1];
        for (let i = 0; i < items.length; i += 2) dict[items[i]] = items[i + 1];
        break;
      }
      case 0x65: {                                    // APPENDS
        const items = popToMark();
        stack[stack.length - 1].push(...items);
        break;
      }
      case 0x2e: return stack.pop();                  // STOP
      default: throw new Error(`unsupported pickle opcode 0x${op.toString(16)}`);
    }
  }
  throw new Error('pickle ended without STOP');
}

// ---------- the lens file ----------

/** Parse a torch .pt lens file; returns {layers:[int], d_model, n_prompts, rows(layer, a, b)}. */
export function parsePtLens(bytes) {
  const zip = readZipEntries(bytes);
  const root = zip.names()[0].split('/')[0];
  const byteorder = new TextDecoder().decode(zip.read(`${root}/byteorder`)).trim();
  if (byteorder !== 'little') throw new Error(`unsupported byteorder: ${byteorder}`);
  const obj = unpickle(zip.read(`${root}/data.pkl`));
  if (!obj || !obj.J) throw new Error('no J dict in lens file — is this a Jacobian-lens .pt?');
  const layers = Object.keys(obj.J).map(Number).sort((a, b) => a - b);
  return {
    layers,
    d_model: obj.d_model,
    n_prompts: obj.n_prompts,
    rows(layer, a, b) {
      const t = obj.J[layer];
      if (!t || !t.__tensor__) throw new Error(`layer ${layer} not in lens`);
      const { storage, offset, shape, stride } = t.__tensor__;
      if (storage.cls !== 'torch.HalfStorage') throw new Error(`unsupported storage: ${storage.cls}`);
      const [rows, cols] = shape;
      if (stride[0] !== cols || stride[1] !== 1) throw new Error('non-contiguous tensor');
      const raw = zip.read(`${root}/data/${storage.key}`);
      const rdv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      const out = [];
      for (let r = a; r < Math.min(b, rows); r++) {
        const v = new Array(cols);
        for (let c = 0; c < cols; c++) v[c] = halfToNumber(rdv.getUint16((offset + r * cols + c) * 2, true));
        out.push(v);
      }
      return out;
    },
  };
}

// ---------- cone math (same algorithm as @intuitionlabs/jspace cones.ts) ----------

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => { const n = norm(a); return n === 0 ? a.map(() => 0) : a.map((x) => x / n); };

export function pursuit(x, basis, k, iters = 64) {
  const dim = x.length;
  const r = x.slice();
  const kk = Math.max(0, Math.min(Math.trunc(k) || 0, basis.length));
  const chosen = [];
  const coeff = new Map();
  for (let t = 0; t < kk; t++) {
    let bestI = -1, bestDot = 1e-12;
    for (let i = 0; i < basis.length; i++) {
      if (coeff.has(i)) continue;
      const d = dot(r, unit(basis[i]));
      if (d > bestDot) { bestDot = d; bestI = i; }
    }
    if (bestI < 0) break;
    chosen.push(bestI);
    coeff.set(bestI, 0);
    for (let pass = 0; pass < iters; pass++) {
      for (const j of chosen) {
        const vj = basis[j];
        const vjvj = dot(vj, vj);
        if (vjvj <= 0) continue;
        const cOld = coeff.get(j);
        let cNew = cOld + dot(r, vj) / vjvj;
        if (cNew < 0) cNew = 0;
        const delta = cNew - cOld;
        if (delta !== 0) {
          for (let d = 0; d < dim; d++) r[d] -= delta * vj[d];
          coeff.set(j, cNew);
        }
      }
    }
  }
  return { indices: chosen.slice(), coeffs: chosen.map((i) => coeff.get(i)), residual: norm(r) };
}

export const coneDistance = (x, basis, k) => pursuit(x, basis, k).residual;

export function deltaMu(basisF, basisG, sample, k) {
  if (!Array.isArray(sample) || sample.length === 0) return 0;
  let s = 0;
  for (const x of sample) {
    const d = coneDistance(x, basisF, k) - coneDistance(x, basisG, k);
    s += d * d;
  }
  return Math.sqrt(s / sample.length);
}

// ---------- the gate (mirrors examples/lens-gate/gate.ts) ----------

export const PURSUIT_K = 3;
export const EXPLAIN_TOL = 0.85;

export function gateProbe(x, basis, tol = EXPLAIN_TOL) {
  const n = norm(x);
  if (n === 0) return { rho: 0, verdict: 'skip' };
  const rho = coneDistance(x, basis, PURSUIT_K) / n;
  return { rho, verdict: rho <= tol ? 'commit' : 'escalate' };
}

/** The two-layer run: bases = rows 0..47 of layers a and b; probes = rows
 * 48..79 of both. Returns deltaMu, the verdict split, and the receipts. */
export function runTwoLayerGate(lens, layerA, layerB, sha256hex) {
  const F = lens.rows(layerA, 0, 48);
  const G = lens.rows(layerB, 0, 48);
  const probes = [
    ...lens.rows(layerA, 48, 80).map((x, i) => ({ id: `L${layerA}.r${48 + i}`, x })),
    ...lens.rows(layerB, 48, 80).map((x, i) => ({ id: `L${layerB}.r${48 + i}`, x })),
  ];
  const dmu = deltaMu(F, G, probes.map((p) => p.x), PURSUIT_K);
  const perProbe = probes.map(({ id, x }) => ({ probe: id, ...gateProbe(x, F) }));
  const commits = perProbe.filter((p) => p.verdict === 'commit').length;
  const escalates = perProbe.filter((p) => p.verdict === 'escalate').length;
  const witness = sha256hex ? `sha256:${sha256hex}` : undefined;
  const receipts = [
    {
      op: 'lens.deltaMu',
      r: dmu,
      // no scalar: deltaMu is not one of the four R's (law L1)
      witness,
      note: `Δ_μ between layer-${layerA} and layer-${layerB} lens cones over ${probes.length} held-out lens directions (k=${PURSUIT_K}). Computed from the file you dropped; nothing left this tab.`,
    },
    {
      op: 'lens.gate',
      witness,
      note: `${commits}/${probes.length} held-out probes commit at EXPLAIN_TOL=${EXPLAIN_TOL} (relative residual of a k=${PURSUIT_K} nonnegative cone fit); ${escalates} escalate. The tolerance is page-local and disclosed; it is not a frozen jspace constant.`,
    },
  ];
  return { deltaMu: dmu, perProbe, commits, escalates, receipts };
}

/** The plain-JSON run: basis = first half of the vectors (≤48), probes = the rest (≤32). */
export function runJsonGate(lensJson) {
  const vs = lensJson.vectors;
  if (!Array.isArray(vs) || vs.length < 4) throw new Error('lens JSON needs a vectors: number[][] with at least 4 rows');
  const split = Math.min(48, Math.ceil(vs.length / 2));
  const basis = vs.slice(0, split);
  const probes = vs.slice(split, split + 32).map((x, i) => ({ id: `r${split + i}`, x }));
  const perProbe = probes.map(({ id, x }) => ({ probe: id, ...gateProbe(x, basis) }));
  const commits = perProbe.filter((p) => p.verdict === 'commit').length;
  const escalates = perProbe.filter((p) => p.verdict === 'escalate').length;
  const receipts = [
    {
      op: 'lens.gate',
      witness: lensJson.source && lensJson.source.sha256 ? `sha256:${lensJson.source.sha256}` : undefined,
      note: `${commits}/${probes.length} held-out rows commit against the first ${basis.length} rows at EXPLAIN_TOL=${EXPLAIN_TOL} (k=${PURSUIT_K}); ${escalates} escalate. Page-local tolerance, disclosed.`,
    },
  ];
  return { perProbe, commits, escalates, receipts };
}
