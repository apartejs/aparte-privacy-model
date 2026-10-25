# Changelog

All aparte-privacy packages share one version: `<model version>.<package fix>`.

## 1.1.0

Model v1.1 (int8, same size and speed as v1). Short messages, the weak spot of v1, are fixed: on 1,067 short chat messages, F1 0.94 overall and 0.85 on names (v1: 0.73 and 0.35). On the test set, 0.92 overall and 0.77 on names (v1: 0.88 and 0.66). Same API.

## 1.0.0

First release: the aparte-privacy v1 model (int8) and `@aparte/privacy`, with `detect`, `minScore` and `ignore`. Seven categories: names, addresses, emails, phone numbers, URLs, dates and account numbers.
