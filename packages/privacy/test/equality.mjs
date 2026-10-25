// Equality test of the JS runtime against the reference reader (python/aparte_privacy/reader.py),
// at the same precision: same spans (same text, same subtype), score within 2e-4, text by text.
//
//   node packages/privacy/test/equality.mjs privacy-v1.1-int8.bin references_int8.jsonl
import { readFileSync } from "node:fs";
import { Privacy } from "../src/runtime.js";

const [, , modelPath, refsPath] = process.argv;
const buf = readFileSync(modelPath);
const model = new Privacy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const lines = readFileSync(refsPath, "utf-8").split("\n").filter(Boolean).map((l) => JSON.parse(l));

// Python offsets count code points; the runtime returns UTF-16 offsets: compare the spans' text
const cp = (s, a, b) => [...s].slice(a, b).join("");
let ok = 0, chars = 0;
const diffs = [];
const t0 = performance.now();
for (const { text, spans } of lines) {
  const mine = model.detect(text);
  chars += text.length;
  const ref = spans.map((s) => ({ text: cp(text, s.start, s.end), subtype: s.subtype, score: s.score }));
  const same = mine.length === ref.length && mine.every((s, i) =>
    s.text === ref[i].text && s.subtype === ref[i].subtype && Math.abs(s.score - ref[i].score) <= 2e-4);
  if (same) ok++;
  else if (diffs.length < 5) diffs.push({ excerpt: text.slice(0, 80), ref, js: mine.map(({ text, subtype, score }) => ({ text, subtype, score })) });
}
const ms = performance.now() - t0;
console.log(`identical spans: ${ok}/${lines.length} | ${(ms / lines.length).toFixed(1)} ms per text, ${(1000 * ms / chars).toFixed(1)} µs per character`);
for (const d of diffs) console.log(JSON.stringify(d));
process.exit(ok === lines.length ? 0 : 1);
