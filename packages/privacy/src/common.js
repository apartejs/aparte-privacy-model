// What every entry shares: the detector class, where the bundled model sits, and
// how to turn whatever the caller hands us into bytes.
//
// Reading a file from disk is Node-only and lives in index.node.js, picked by
// the "node" condition of package.json. The portable entry (index.js) is the
// default on purpose: a tool that knows nothing about export conditions gets
// the one that builds everywhere, instead of failing on a Node built-in.
export { Privacy, readModel } from "./runtime.js";
import { Privacy } from "./runtime.js";

/** URL of the bundled model file (int8), as resolved from this module. */
export const modelUrl = new URL("../model/privacy-v1-int8.bin", import.meta.url);

/** Fetch a model file over the network. */
export async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("could not load the aparte-privacy model from " + url + " (" + response.status + ")");
  return response.arrayBuffer();
}

/** Anything a caller may reasonably hand us, as an ArrayBuffer. */
export async function toBytes(source) {
  const value = await source;
  if (value instanceof ArrayBuffer) return value;
  if (ArrayBuffer.isView(value)) return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  if (typeof Response !== "undefined" && value instanceof Response) return value.arrayBuffer();
  if (value instanceof URL || typeof value === "string") return fetchBytes(value);
  throw new TypeError("loadPrivacy() takes a URL, a string, an ArrayBuffer, a typed array or a Response");
}

/** Builds the package's loadPrivacy: no argument = the bundled model, cached. */
export function loader(readBundled) {
  let cached;
  return async function loadPrivacy(source) {
    if (source === undefined) return (cached ??= readBundled().then((bytes) => new Privacy(bytes)));
    return new Privacy(await toBytes(source));
  };
}
