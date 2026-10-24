// API tests on the bundled model. Made-up personal data only: no real name, number or address lives in
// this repository. The text-by-text equality test against the Python reader runs on references stored
// on Hugging Face (see equality.mjs).
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadPrivacy } from "../src/index.node.js";

const privacy = await loadPrivacy();
const CATEGORIES = ["private_person", "private_address", "private_email", "private_phone", "private_url", "private_date", "account_number"];

test("the bundled file is an aparte-privacy model", () => {
  assert.equal(privacy.header.task, "privacy");
  assert.equal(privacy.header.precision, "int8");
  assert.deepEqual(privacy.header.labels.subtypes, CATEGORIES);
});

test("finds a name, a phone number and an email, with their kinds and offsets", () => {
  const text = "Appelle Jean Dupont au 06 12 34 56 78 ou écris à jean.dupont@gmail.com";
  const found = privacy.detect(text);
  for (const kind of ["private_person", "private_phone", "private_email"]) {
    const s = found.find((x) => x.subtype === kind);
    assert.ok(s, kind + " missing in " + JSON.stringify(found));
    assert.equal(text.slice(s.start, s.end), s.text);
    assert.ok(s.score > 0 && s.score <= 1);
  }
});

test("a text without personal data gives nothing", () => {
  assert.deepEqual(privacy.detect(""), []);
  assert.deepEqual(privacy.detect("Can you explain how photosynthesis works?"), []);
});

test("ignore and minScore are applied after the model", () => {
  const text = "My name is John Smith and my email is john.smith@gmail.com";
  const name = privacy.detect(text).find((x) => x.subtype === "private_person").text;
  assert.ok(!privacy.detect(text, { ignore: [name] }).some((x) => x.text === name));
  assert.deepEqual(privacy.detect(text, { minScore: 1.01 }), []);
});

test("mask and unmask round-trip", () => {
  const text = "Appelle Jean Dupont au 06 12 34 56 78 ou écris à jean.dupont@gmail.com";
  const masked = privacy.mask(text);
  assert.ok(!masked.text.includes("Jean Dupont") && !masked.text.includes("06 12 34 56 78"), masked.text);
  assert.ok(!("secrets" in masked));
  assert.equal(Object.keys(masked.values).length, 3);
  assert.equal(privacy.unmask(masked.text, masked.values), text);
});

test("the loader caches the bundled model", async () => {
  assert.equal(await loadPrivacy(), privacy);
});
