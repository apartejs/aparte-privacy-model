// @aparte/privacy — the runtime plus the model (int8, ~196 KB), ready to use with no configuration.
//
//   import { loadPrivacy } from "@aparte/privacy";
//   const privacy = await loadPrivacy();
//   privacy.detect("Appelle Jean Dupont au 06 12 34 56 78");
//
// This is the portable entry: it fetches the model, and imports nothing that a
// browser bundler cannot resolve. Node picks index.node.js instead (the "node"
// condition), which reads the bundled file straight from disk.
//
// A bundler rewrites neither `import.meta.url` nor the model's path, so once
// your app is bundled the default URL no longer points at the package: serve
// the `.bin` yourself and pass it in — `loadPrivacy(url)`, or with Vite
// `import modelUrl from "@aparte/privacy/model?url"`. See the README.
export { Privacy, readModel, modelUrl } from "./common.js";
import { modelUrl, fetchBytes, loader } from "./common.js";

/** Load the model once and return a ready detector. No argument = the bundled model. */
export const loadPrivacy = loader(() => fetchBytes(modelUrl));
