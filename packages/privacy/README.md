# @aparte/privacy

Finds personal data in text on your device: **names, addresses, emails, phone numbers, URLs, dates and account numbers**. A 196 KB model, no API call, no network, no dependency. Browser, Node and workers.

```js
import { loadPrivacy } from "@aparte/privacy";

const privacy = await loadPrivacy();
privacy.detect("Appelle Jean Dupont au 06 12 34 56 78 ou écris à jean.dupont@gmail.com");
// [{ start: 8,  end: 19, text: "Jean Dupont",           subtype: "private_person", score: 0.986 },
//  { start: 23, end: 37, text: "06 12 34 56 78",        subtype: "private_phone",  score: 0.9838 },
//  { start: 49, end: 70, text: "jean.dupont@gmail.com", subtype: "private_email",  score: 0.9798 }]
```

Each span has `start` and `end` (UTF-16 offsets: `text.slice(start, end)`), its `text`, its `subtype` (one of `private_person`, `private_address`, `private_email`, `private_phone`, `private_url`, `private_date`, `account_number`) and a `score` from 0 to 1.

## Options

`detect(text, { minScore, ignore })` and `mask(text, { minScore, ignore, placeholder })`. The model reports raw spans; `ignore` (exact strings or RegExps) and `minScore` are your policy, applied after it. `mask(text)` replaces every value with a placeholder (`[PRIVATE_PERSON_1]`…) and returns `{ text, values }`; `unmask(answer, values)` puts them back, on the client.

For secrets (passwords, API keys, tokens), use [`@aparte/secrets`](https://www.npmjs.com/package/@aparte/secrets): the two models run side by side on the same text.

## In a bundled app

A bundler rewrites neither `import.meta.url` nor the model's path: serve the `.bin` yourself and pass it in, `loadPrivacy(url)`, or with Vite `import modelUrl from "@aparte/privacy/model?url"`.

## Limits

- **Short messages are its weak spot.** On a message of a few words, it misses most names and some phone numbers: on 100 short synthetic messages, names are found 9 times out of 100 and phone numbers 59 out of 100 (score ≥ 0.5). Inside a longer request, it finds them far more often. Do not rely on it alone for short messages.
- **Names are the weakest category** in general: 0.66 F1 on our test set, against 0.9 and more for the other categories.
- **Public figures and fictional characters are masked too**: the model cannot know who is famous. Use `ignore` for the names you want to keep.
- **English and French**; other languages are not tested.
- The numbers come from our own test set (real chat messages with synthetic personal data inserted), not from an independent audit.

## License

MIT, Copyright (c) 2026 Paul Richez.
