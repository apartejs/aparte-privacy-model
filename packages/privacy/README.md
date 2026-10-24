# @aparte/privacy

Finds personal data in text on your device: **names, addresses, emails, phone numbers, URLs, dates and account numbers**. A 196 KB model, no API call, no network, no dependency.

```js
import { loadPrivacy } from "@aparte/privacy";

const privacy = await loadPrivacy();
privacy.detect("Appelle Jean Dupont au 06 12 34 56 78 ou écris à jean.dupont@gmail.com");
// [{ text: "Jean Dupont", subtype: "private_person", score: 0.99, ... },
//  { text: "06 12 34 56 78", subtype: "private_phone", ... },
//  { text: "jean.dupont@gmail.com", subtype: "private_email", ... }]
```

Each span has `start` and `end` (UTF-16 offsets), its `text`, its `subtype` (one of the seven categories above) and a `score` from 0 to 1. The model reports raw spans: `minScore` and `ignore` are your policy, applied after it.

For secrets (passwords, API keys, tokens), use [`@aparte/secrets`](https://www.npmjs.com/package/@aparte/secrets): the two models run side by side on the same text.

## Limits

- Names are the weakest category: about 0.66 F1 on our test set, against 0.9 and more for the other categories. Plan for a human check on names in high-stakes uses.
- Public figures and fictional characters are masked too: the model cannot know who is famous. Use `ignore` for names you want to keep.
- English and French. Other languages are not tested.
- The test set is a benchmark we wrote and checked by hand; it is not an independent audit.

## License

MIT, Copyright (c) 2026 Paul Richez.
