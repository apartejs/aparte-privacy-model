# Changelog

## 1.1.0

The aparte-privacy v1.1 model (int8, 196 KB, same speed). Short messages are fixed: on 1,067 short chat messages (« Merci Sophie ! », « rappelle-moi au … »), F1 0.94 overall and 0.85 on names, against 0.73 and 0.35 for v1. On the test set: 0.92 overall, 0.77 on names (v1: 0.88 and 0.66). Public figures, fictional characters and names written in lowercase are found more often. Same API, same spans as the Python reader (200 reference texts).

## 1.0.0

First release: the aparte-privacy v1 model (int8, 196 KB) with the aparte runtime. Seven categories: personal names, addresses, emails, phone numbers, URLs, dates and account numbers. Same spans as the Python reader (checked on 60 texts of the benchmark).
