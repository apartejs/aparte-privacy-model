// @aparte/privacy — personal data in text, on the device.
//
//   import { loadPrivacy } from "@aparte/privacy";
//   const privacy = await loadPrivacy();
//   privacy.detect("Appelle Jean Dupont au 06 12 34 56 78");
//   // [{ start, end, text, subtype: "private_person", score }]
//
// The runtime is the one of @aparte/secrets (same file format, same spans); the bundled model is the
// privacy model (7 categories), not the secrets one. The model reports raw spans: what to keep is your
// policy, applied after it (minScore, ignore).
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Secrets } from "./runtime.js";

export { Secrets };

/** Load the bundled privacy model (int8). Load it once and keep the result. */
export async function loadPrivacy(source) {
  const buf = source
    ? new Uint8Array(await (await fetch(source)).arrayBuffer())
    : await readFile(fileURLToPath(new URL("../model/privacy-v1-int8.bin", import.meta.url)));
  const bytes = buf.buffer ? buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) : buf;
  return new Secrets(bytes);
}
