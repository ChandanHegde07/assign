# A3 — Corrected multilingual comparison

Method: FLORES-200 `devtest`, 1012 parallel sentences, 6 languages, sentences
tokenized independently, no lowercasing (see A1). Five tokenizers; four
denominators. Reproduction: `partA/corrected_analysis.py` →
`results/corrected_devtest.txt`, `results/summary_table.md`.

## Headline results — tokens per parallel sentence (ratio vs English)

| tokenizer | English | Hindi | Kannada | Tamil | Telugu | Malayalam |
|---|---|---|---|---|---|---|
| gpt2 (tiktoken, English-centric) | 26.7 | 198.3 (**7.42×**) | 363.0 (**13.58×**) | 415.2 (**15.54×**) | 346.6 (12.97×) | 405.1 (15.16×) |
| cl100k_base (tiktoken) | 26.9 | 128.0 (4.77×) | 237.9 (8.86×) | 205.3 (7.64×) | 222.6 (8.29×) | 240.0 (8.94×) |
| o200k_base (tiktoken, modern multilingual) | 26.6 | 41.7 (**1.57×**) | 52.3 (**1.97×**) | 52.6 (1.98×) | 51.2 (1.93×) | 52.0 (1.96×) |
| xlm-roberta-base (multilingual SP) | 30.3 | 37.8 (1.25×) | 41.0 (1.35×) | 40.9 (1.35×) | 39.9 (1.32×) | 41.7 (1.38×) |
| ai4bharat/IndicBERTv2-MLM-only (Indic) | 26.8 | 31.3 (**1.17×**) | 29.4 (**1.10×**) | 28.1 (1.05×) | 28.4 (1.06×) | 29.5 (1.10×) |

The `dev` split reproduces every ratio within ~1% (`results/corrected_dev.txt`),
and per-sentence paired mean/median ratios agree with the aggregate ratios
(e.g. gpt2 Hindi: aggregate 7.42×, paired mean 7.51×, median 7.40×) — the
result is not an artifact of averaging.

**The report's magnitude and its root cause are both wrong.** The "6×" is a
property of the *GPT-2 tokenizer*, not of Hindi/the script: on identical text
the Hindi penalty is 7.42× (gpt2) but 1.17× (IndicBERTv2). Tokenizer choice
moves the number by 6.3×, which falsifies REPORT_v0's "any tokenizer will
struggle … not the tokenizer".

## Denominator reasoning — what is each denominator holding constant?

The denominator must be the thing that is the *same* in every language; only
then does the ratio compare tokenizer cost rather than measure unrelated
surface differences.

| denominator | what it holds constant | why it is wrong for routing/cost |
|---|---|---|
| whitespace word | token count per space-delimited token | word boundaries and morphological density differ (agglutinative Dravidian packs a phrase into one word). The "same" word is not the same amount of meaning. |
| unicode code point (`len()`, report's `tok/char`) | code points | not even a consistent surface unit: Devanagari/Tamil combining marks are separate code points. Hindi has 131,180 code points vs 85,957 grapheme clusters (+52.6%). It applies a different unit to each script. |
| grapheme cluster | user-perceived characters | closer, but still surface: one English word ≈ many graphemes of meaning-bearing Indic text. It also ignores economy of writing systems. |
| UTF-8 byte | raw storage bytes | tokenizer-dependent and can invert the ranking: with strong tokenizers Indic drops below 1× (o200k Hindi 0.62×, IndicBERTv2 0.46×) because Indic characters need ~3 bytes, while gpt2 is still 2.90×. It measures script byte-width, not content; using it would conclude Indic is *cheaper*, which is false for per-request token cost. |
| **parallel sentence** | **the same meaning, by construction** | **correct**: each of the 1012 lines is the professional translation of the same source sentence, so `tokens_lang / tokens_eng` is exactly "tokens needed to process the same request". |

Measured denominator spread for the *same* corpus and *same* tokenizer (gpt2,
FLORES devtest): Hindi/English = **2.90×** (byte) → 6.34× (word) → 7.42×
(parallel sentence) → **11.39×** (grapheme). A **3.9× spread** from the
denominator alone. REPORT_v0's two "agreeing" metrics (word and code-point-char)
are both non-content-constant, so their agreement is not confirmation; both are
biased in the same direction.

## The single metric to drive routing and cost

**Tokens per sentence, measured on a sentence-aligned parallel corpus, reported
as a ratio to English** (equivalently: mean tokens to encode the same request,
relative to English). Rationale:

1. It is the only denominator under which "equal value" means "equal meaning",
   so the ratio isolates tokenizer/language cost rather than script encoding.
2. Tokens are the billing/throughput unit: cost ≈ tokens × price; KV-cache and
   prefill cost are per token (see Part B). A content-constant token ratio is
   therefore directly a per-equivalent-request cost multiplier.
3. It is robust: aggregate vs paired mean vs median agree, and dev vs devtest
   agree within ~1%.

Numerically, for a production model, the routing decision should use the ratio
of **that model's own tokenizer**, not GPT-2's. Corrected headline, if the
candidate model uses a modern multilingual tokenizer (o200k-class): budget
**≈1.6× English cost for Hindi and ≈2.0× for the four Dravidian languages**;
with an Indic-specialised tokenizer (IndicBERTv2-class), **≈1.05–1.18×**. The
6× figure over-budgets by roughly 3–6×.

Two things this metric deliberately does **not** capture and must be multiplied
in for a real routing decision: (i) generation-length ratios (FLORES is
input-only), and (ii) the target model's quality and per-token price. A token
ratio of 2× is only decisive if the quality/price of the routed model is at
least as good.
