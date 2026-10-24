// @aparte/privacy — the Node entry, picked by the "node" export condition.
//
// Same API as the portable entry (index.js); the only difference is that the
// bundled model is read from the file system instead of fetched, so it works
// with no server and no network.
export { Privacy, Secrets, readModel, modelUrl } from "./common.js";
import { modelUrl, fetchBytes, loader } from "./common.js";

async function readBundled() {
  if (modelUrl.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const buf = await readFile(fileURLToPath(modelUrl));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  return fetchBytes(modelUrl);
}

/** Load the model once and return a ready detector. No argument = the bundled model. */
export const loadPrivacy = loader(readBundled);
