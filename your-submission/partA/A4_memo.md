# A4 — Recommendation memo

**To:** Serving / Product leadership  **Re:** tokenizer routing & Indic cost
**Basis:** FLORES-200, 1012 parallel sentences × 6 languages; 5 tokenizers;
code in `partA/`.

## Corrected headline numbers

- **Do not use the 6× figure.** The 5.89× in REPORT_v0 is specific to the GPT-2
  tokenizer, a non-content-constant word denominator, and a macro-average on a
  10-line, non-parallel toy set.
- On identical, sentence-aligned content, the cost to encode the same request
  (tokens/parallel sentence, ratio vs English) is:

| tokenizer class | Hindi | Kannada | Tamil | Telugu | Mal. |
|---|---|---|---|---|---|
| GPT-2 (current assumption) | 7.4× | 13.6× | 15.5× | 13.0× | 15.2× |
| Modern multilingual (o200k-class) | **1.6×** | **2.0×** | 2.0× | 1.9× | 2.0× |
| Indic-specialised (IndicBERTv2-class) | 1.2× | 1.1× | 1.05× | 1.06× | 1.1× |

- Root cause is the tokenizer, not the script: tokenizer choice alone moves the
  Hindi ratio 7.4× → 1.2× on the same text. "Any tokenizer will struggle" is
  false.

## Routing recommendation

1. **Choose the model by its tokenizer.** A deployed model's tokenizer is fixed,
   so this is a model-selection/migration decision (or a one-time retrain), not a
   config flag: pick an o200k-class or Indic-vocab model. This removes ~70–85% of
   the asserted Indic penalty at **zero marginal inference cost**, but carries a
   one-time migration/retraining cost that this audit does not estimate.
2. **Budget 1.6× (Hindi) and ~2.0× (Dravidian)** per equivalent request relative
   to English for a modern tokenizer — not 6×.
3. Keep a **separate route only if it wins on quality-per-unit-cost**, i.e. only
   after multiplying the token ratio by the candidate model's quality and price;
   a language-only routing rule is unjustified by token counts.

## Biggest caveat

FLORES is **formal, professionally translated prose and input-only**. It does
not measure (a) generated-output token ratios, which are often the larger cost
term for Indic, or (b) casual/code-mixed Hinglish — the actual product register.
So the 2× budget is a **lower bound on the input side**; validate on real
traffic before committing spend.

## One production metric to catch this being wrong

Track **tokens per request, p50 and p95, split by detected language, on live
traffic** (and the derived cost/token per language). Expected under this
analysis: Hindi ≈1.6×, Dravidian ≈2.0× the English token count per request. If
production shows ≫2× (or a different ranking), the offline corpus is
unrepresentative — most likely output tokens or code-mixing dominate — and the
routing budget must be re-derived. This metric directly falsifies the analysis
at the point where money is spent, unlike a static offline re-run.
