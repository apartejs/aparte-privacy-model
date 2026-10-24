"""Reference implementation of the aparte model format, in numpy (aparte-privacy and aparte-secrets).

Everything a runtime has to do, without torch or the `tokenizers` library —
the executable specification the JavaScript, Kotlin and Swift runtimes are
tested against:

  1. read the `.bin` (format 1, same container as aparte-titler): JSON header,
     BPE merges (uint16 pairs), tensors (integers packed on `bits` bits + one
     fp16 scale per row; fp16 vectors; or float32 throughout for fp32)
  2. split the text into units (`pre_tokenizer.pattern`: runs of letters,
     digits, `_` and `-` with an optional leading space; every other symbol
     alone; whitespace), then byte-level BPE inside each unit
  3. windows of `window` tokens every `stride` tokens; in each window the
     transformer gives 21 logits per token, averaged over the tokens of each
     unit, then log-softmax; overlapping windows are averaged per token
  4. one decision per unit: mean of its tokens' log-probabilities, then a
     constrained BIOES Viterbi over units (one subtype per span)
  5. spans: (start, end, subtype, score), score = min over the span's units
     of 1 - P(O)

    python reader.py secrets-v1-int4.bin "my db password is Hunter2024!"
    python reader.py secrets-v1-int4.bin --check texts.jsonl tokenizer.json   # tokens vs HF tokenizers
    python reader.py secrets-v1-int4.bin --references in.jsonl out.jsonl
"""
import io
import json
import math
import re
import struct
import sys

import numpy as np

# --------------------------------------------------------------- reading


def read_model(path):
    """(header, merges as an (n, 2) uint16 array, {tensor name: float32 array})."""
    with open(path, "rb") as fh:
        assert fh.read(4) == b"LMAP", "not an aparte model file"
        n = struct.unpack("<I", fh.read(4))[0]
        header = json.loads(fh.read(n).decode("utf-8"))
        assert header.get("format", 1) == 1, "format %s is not supported" % header.get("format")
        assert header.get("task") in ("secrets", "privacy"), "not an aparte model (task %r)" % header.get("task")
        n = struct.unpack("<I", fh.read(4))[0]
        merges = np.frombuffer(fh.read(n), dtype=np.uint16).reshape(-1, 2)
        weights = {}
        for t in header["tensors"]:
            blob = fh.read(t["bytes"])
            shape = tuple(t["shape"])
            if t["type"] == "f32":
                weights[t["name"]] = np.frombuffer(blob, dtype=np.float32).reshape(shape)
            elif t["type"] == "f16":
                weights[t["name"]] = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(shape)
            else:
                bits = int(t["type"][1:])
                count = int(np.prod(shape))
                packed = (count * bits + 7) // 8
                q = unpack_bits(np.frombuffer(blob[:packed], dtype=np.uint8), count, bits)
                scale = np.frombuffer(blob[packed:], dtype=np.float16).astype(np.float32).reshape(shape[0], 1)
                weights[t["name"]] = (q.reshape(shape) * scale).astype(np.float32)
    return header, merges, weights


def unpack_bits(data, n, bits):
    """Bit stream, least significant bit first: n signed integers of `bits` bits."""
    if bits == 8:
        return data[:n].astype(np.int32) - 128
    b = np.unpackbits(data, bitorder="little")[:n * bits].reshape(n, bits)
    return (b.astype(np.int32) << np.arange(bits)).sum(axis=1) - (1 << (bits - 1))


# ------------------------------------------------------- units, tokenizer


def unit_regex(pattern):
    """The header's pattern uses \\p{L} and \\p{N}; Python's `re` has no \\p{...}.
    [\\p{L}\\p{N}_] is exactly Python's \\w (str.isalnum() or "_")."""
    return re.compile(pattern.replace(r"\p{L}\p{N}_", r"\w"))


class Tokenizer:
    def __init__(self, header, merges):
        self.byte_ids = header["byte_ids"]          # token id of each of the 256 bytes
        self.base = header["merges_base"]          # merge k produces token id base + k
        self.rank = {(int(a), int(b)): k for k, (a, b) in enumerate(merges)}
        self.units = unit_regex(header["pre_tokenizer"]["pattern"])

    def encode(self, text):
        """Token ids, (start, end) character offsets, the unit of each token, unit offsets."""
        ids, offsets, unit_of, units = [], [], [], []
        for m in self.units.finditer(text):
            start = m.start()
            u = len(units)
            units.append((m.start(), m.end()))
            char_of_byte, tokens = [], []
            for i, ch in enumerate(m.group()):
                for b in ch.encode("utf-8"):
                    tokens.append((self.byte_ids[b], 1))
                    char_of_byte.append(i)
            while len(tokens) > 1:                  # merge the lowest-ranked pair first
                best, at = None, -1
                for j in range(len(tokens) - 1):
                    r = self.rank.get((tokens[j][0], tokens[j + 1][0]))
                    if r is not None and (best is None or r < best):
                        best, at = r, j
                if best is None:
                    break
                tokens[at:at + 2] = [(self.base + best, tokens[at][1] + tokens[at + 1][1])]
            pos = 0
            for tid, n in tokens:
                ids.append(tid)
                offsets.append((start + char_of_byte[pos], start + char_of_byte[pos + n - 1] + 1))
                unit_of.append(u)
                pos += n
        return ids, offsets, unit_of, units


def length_classes(text, units, n_classes):
    """Length class of each unit: log2 scale of its non-blank characters (0 for blank)."""
    return [min(n_classes - 1, len(text[a:b].strip()).bit_length()) for a, b in units]


# ----------------------------------------------------------------- model


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


_erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


class Model:
    def __init__(self, header, weights):
        self.h, self.w = header, weights
        self.D, self.H = header["dim"], header["heads"]
        D, P = self.D, header["max_pos"]
        pos = np.arange(P)[:, None].astype(np.float32)
        freq = np.exp(np.arange(0, D, 2).astype(np.float32) * (-math.log(10000.0) / D))
        self.table_pos = np.zeros((P, D), dtype=np.float32)
        self.table_pos[:, 0::2] = np.sin(pos * freq)
        self.table_pos[:, 1::2] = np.cos(pos * freq)

    def logits(self, ids, lengths):
        """21 logits per token for one window (every token sees every token of the window)."""
        w, D, H = self.w, self.D, self.H
        L = len(ids)
        x = w["emb_tok.weight"][ids] @ w["proj_emb.weight"].T + self.table_pos[:L] + w["emb_len.weight"][lengths]
        Dh = D // H
        for c in range(self.h["layers"]):
            k = "blocks.%d." % c
            y = layer_norm(x, w[k + "n1.weight"], w[k + "n1.bias"])
            qkv = y @ w[k + "att.qkv.weight"].T + w[k + "att.qkv.bias"]
            q, kk, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
            out = np.zeros((L, D), dtype=np.float32)
            for head in range(H):
                sl = slice(head * Dh, (head + 1) * Dh)
                s = (q[:, sl] @ kk[:, sl].T) / math.sqrt(Dh)
                s = np.exp(s - s.max(axis=-1, keepdims=True))
                out[:, sl] = (s / s.sum(axis=-1, keepdims=True)) @ v[:, sl]
            x = x + out @ w[k + "att.output.weight"].T + w[k + "att.output.bias"]
            y = layer_norm(x, w[k + "n2.weight"], w[k + "n2.bias"])
            f = gelu(y @ w[k + "ff.net.0.weight"].T + w[k + "ff.net.0.bias"])
            x = x + f @ w[k + "ff.net.2.weight"].T + w[k + "ff.net.2.bias"]
        xn = layer_norm(x, w["norm.weight"], w["norm.bias"])
        return xn @ w["head.weight"].T + w["head.bias"]            # (L, n_classes)


def windows(n, size, stride):
    if n <= size:
        return [0]
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] + size < n:
        starts.append(n - size)
    return starts


# -------------------------------------------------------------- decoding


def transitions(n_classes):
    """Allowed BIOES moves with one subtype per span. State k > 0: subtype (k-1)//4, tag (k-1)%4."""
    tag = lambda k: None if k == 0 else (k - 1) % 4        # noqa: E731
    sub = lambda k: None if k == 0 else (k - 1) // 4       # noqa: E731
    opens = lambda k: k == 0 or tag(k) in (0, 3)           # noqa: E731  O, B, S
    M = np.full((n_classes, n_classes), -1e9)
    for j in range(n_classes):
        for k in range(n_classes):
            if tag(j) in (0, 1):                           # after B or I: I or E of the same subtype
                ok = tag(k) in (1, 2) and sub(k) == sub(j)
            else:                                          # after O, E, S: O, B or S of any subtype
                ok = opens(k)
            if ok:
                M[j, k] = 0.0
    start = np.array([0.0 if opens(k) else -1e9 for k in range(n_classes)])
    end = np.array([0.0 if k == 0 or tag(k) in (2, 3) else -1e9 for k in range(n_classes)])
    return M, start, end


def viterbi(logp, M, start, end):
    L, C = logp.shape
    score = logp[0] + start
    back = np.zeros((L, C), dtype=np.int64)
    for t in range(1, L):
        cand = score[:, None] + M
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    k = int((score + end).argmax())
    path = [k]
    for t in range(L - 1, 0, -1):
        k = int(back[t, k])
        path.append(k)
    return path[::-1]


class Secrets:
    def __init__(self, path):
        self.header, merges, weights = read_model(path)
        self.tokenizer = Tokenizer(self.header, merges)
        self.model = Model(self.header, weights)
        self.subtypes = self.header["labels"]["subtypes"]
        self.n_classes = 1 + 4 * len(self.subtypes)
        self.trans = transitions(self.n_classes)

    def detect(self, text):
        """[{start, end, subtype, score}] in text order."""
        h = self.header
        ids, offsets, unit_of, units = self.tokenizer.encode(text)
        if not ids:
            return []
        lengths = length_classes(text, units, h["unit_length_classes"])
        tok_len = np.array([lengths[u] for u in unit_of])
        n, C = len(ids), self.n_classes
        total = np.zeros((n, C), dtype=np.float64)
        count = np.zeros((n, 1), dtype=np.float64)
        ids_a, unit_a = np.array(ids), np.array(unit_of)
        for d in windows(n, h["window"], h["stride"]):
            sl = slice(d, min(d + h["window"], n))
            z = self.model.logits(ids_a[sl], tok_len[sl])
            u = unit_a[sl] - unit_a[d]                     # unit numbers inside the window
            sums = np.zeros((u.max() + 1, C), dtype=np.float32)
            np.add.at(sums, u, z)
            z = sums[u] / np.bincount(u)[u][:, None]       # one decision per unit inside the window
            total[sl] += log_softmax(z)
            count[sl] += 1
        logp = total / count
        kept = sorted(set(unit_of))                        # units that have tokens, in order
        idx = {u: i for i, u in enumerate(kept)}
        lp_u = np.zeros((len(kept), C))
        n_u = np.zeros((len(kept), 1))
        for t, u in enumerate(unit_of):
            lp_u[idx[u]] += logp[t]
            n_u[idx[u]] += 1
        lp_u = lp_u / n_u
        path = viterbi(lp_u, *self.trans)
        p_not_o = 1.0 - np.exp(lp_u[:, 0])
        spans, first = [], None
        for i, e in enumerate(path):
            if e == 0:
                first = None
                continue
            tag = (e - 1) % 4
            if tag in (0, 3):
                first = i
            if tag in (2, 3) and first is not None:
                a, b = units[kept[first]][0], units[kept[i]][1]
                while a < b and text[a].isspace():
                    a += 1
                spans.append({"start": a, "end": b, "subtype": self.subtypes[(e - 1) // 4],
                              "score": round(float(p_not_o[first:i + 1].min()), 4)})
                first = None
        return spans


# ------------------------------------------------------------ utilities


def check(path, texts_file, tokenizer_json):
    """Tokens and offsets against the HF `tokenizers` library (needs the tokenizer.json)."""
    from tokenizers import Tokenizer as HFTokenizer
    s = Secrets(path)
    hf = HFTokenizer.from_file(tokenizer_json)
    n = same_ids = same_offsets = 0
    for line in io.open(texts_file, encoding="utf-8"):
        text = json.loads(line)["text"]
        ids, offsets, _, _ = s.tokenizer.encode(text)
        enc = hf.encode(text)
        n += 1
        same_ids += ids == enc.ids
        same_offsets += [tuple(o) for o in offsets] == [tuple(o) for o in enc.offsets]
    print("%d texts: identical tokens %d, identical offsets %d" % (n, same_ids, same_offsets))


def references(path, inputs, output):
    """Reference spans for the runtime equality tests."""
    s = Secrets(path)
    n = 0
    with io.open(output, "w", encoding="utf-8") as out:
        for f in inputs:
            for line in io.open(f, encoding="utf-8"):
                text = json.loads(line)["text"]
                out.write(json.dumps({"text": text, "spans": s.detect(text)}, ensure_ascii=False) + "\n")
                n += 1
    print("%d references -> %s" % (n, output))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 2 and sys.argv[2] == "--check":
        check(sys.argv[1], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 2 and sys.argv[2] == "--references":
        references(sys.argv[1], sys.argv[3:-1], sys.argv[-1])
    else:
        text = sys.argv[2]
        for sp in Secrets(sys.argv[1]).detect(text):
            print("%s  %-11s %.4f  %r" % ((sp["start"], sp["end"]), sp["subtype"], sp["score"], text[sp["start"]:sp["end"]]))
