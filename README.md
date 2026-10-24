# aparte-privacy

Finds personal data in text on the device: names, addresses, emails, phone numbers, URLs, dates and account numbers. A 196 KB model, no API call, no network.

This repository holds the code (npm package, tests). The model files are on [Hugging Face](https://huggingface.co/apartejs/aparte-privacy). Secrets (passwords, API keys) are in [aparte-secrets](https://github.com/apartejs/aparte-secrets-model).

## Packages

| path | package |
|---|---|
| `packages/privacy/` | `@aparte/privacy` (npm): runtime and the int8 model |
| `python/aparte_privacy/reader.py` | the reference implementation, numpy only |

Kotlin: [aparte-privacy-kotlin](https://github.com/apartejs/aparte-privacy-kotlin). Swift: [aparte-privacy-swift](https://github.com/apartejs/aparte-privacy-swift).

## Use

```js
import { loadPrivacy } from "@aparte/privacy";
const privacy = await loadPrivacy();
privacy.detect("Appelle Jean Dupont au 06 12 34 56 78");
```

See the package README for the options and the limits: names are the weakest category, and short messages are its weak spot.

## License

MIT, Copyright (c) 2026 Paul Richez.
