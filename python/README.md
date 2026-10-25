# aparte-privacy reader (Python)

The reference implementation of the `.bin` format, in numpy only: the executable specification the JavaScript, Kotlin and Swift runtimes are tested against. It reads both aparte-privacy and aparte-secrets files.

```bash
pip install -r requirements.txt
python aparte_privacy/reader.py privacy-v1.1-int8.bin "Appelle Jean Dupont au 06 12 34 56 78 ou écris à jean.dupont@gmail.com"
# (8, 19)  private_person 0.9997  'Jean Dupont'
# (23, 37)  private_phone 0.9995  '06 12 34 56 78'
# (49, 70)  private_email 0.9998  'jean.dupont@gmail.com'
```

Offsets count code points, as Python strings do.
