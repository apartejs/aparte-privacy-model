/** A piece of personal data found in the text. Offsets are UTF-16 indices: `text.slice(start, end)`. */
export interface Span {
  start: number;
  end: number;
  text: string;
  subtype:
    | "private_person"
    | "private_address"
    | "private_email"
    | "private_phone"
    | "private_url"
    | "private_date"
    | "account_number";
  /** The model's confidence, between 0 and 1. */
  score: number;
}

/** The caller's policy, applied after the model (the model itself always reports raw spans). */
export interface DetectOptions {
  /** Drop spans scored below this (default 0). */
  minScore?: number;
  /** Values to ignore: an exact string, or a RegExp tested on the span's text. */
  ignore?: Iterable<string | RegExp>;
}

export interface MaskOptions extends DetectOptions {
  /** Placeholder for the n-th distinct value (default `[PRIVATE_PERSON_1]`, `[PRIVATE_PHONE_2]`…). */
  placeholder?: (subtype: string, n: number) => string;
}

export class Privacy {
  /** @param buffer the contents of a `.bin` model file */
  constructor(buffer: ArrayBuffer);
  /** The model file's JSON header (name, version, precision, architecture…). */
  readonly header: Record<string, unknown>;
  /** Every piece of personal data found in the text, in text order. */
  detect(text: string, options?: DetectOptions): Span[];
  /** Replace every value with a placeholder; `values` maps placeholders back to the original values (keep it on the client). */
  mask(text: string, options?: MaskOptions): { text: string; values: Record<string, string> };
  /** Put the values back: the inverse of `mask`. */
  unmask(text: string, values: Record<string, string>): string;
}

/** Parse a `.bin` model file. */
export function readModel(buffer: ArrayBuffer): { header: Record<string, unknown>; merges: Uint16Array; weights: Record<string, { data: Float32Array; shape: number[] }> };

/** URL of the bundled model file (int8), as resolved from the package. */
export const modelUrl: URL;

/**
 * Load a model and return a ready detector.
 *
 * With no argument, the model bundled in this package: read from disk in Node,
 * fetched from `modelUrl` elsewhere. Once your app is bundled that URL no
 * longer points at the package, so pass the model in yourself — a URL of the
 * file you serve, or its bytes.
 */
export function loadPrivacy(source?: URL | string | ArrayBuffer | ArrayBufferView | Response | Promise<ArrayBuffer | Response>): Promise<Privacy>;
