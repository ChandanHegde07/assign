# A2 — Audit of `fertility.py` and the metric

All numbers below are produced by `partA/audit_fertility.py` and
`partA/flaw_impacts.py`. Raw output: `partA/results/audit_v0.txt`,
`partA/results/flaw_impacts.txt`. Baseline reproduction:

```
cd starter_kit
python fertility.py --corpus eng=corpus_sample/eng_sample.txt \
                    --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
# eng 1.27 / 0.226   hin 7.45 / 1.579   "hin is 5.89x the fertility of eng"
```

This reproduces REPORT_v0 exactly (1.2652 / 7.4485 / 5.887 at full precision).

---

## Summary of findings

| # | Item | Category | Measured effect on the reported ratio (5.887) |
|---|---|---|---|
| 1 | `line.split(" ")` makes empty "words" | **code bug** | 5.887 → 5.922 |
| 2 | per-line mean of ratios (macro) instead of aggregate (micro) | **code/statistics bug** | 5.922 → 5.928 |
| 3 | `tokens per whitespace word` is not content-constant | **conceptual problem** | ratio spans 2.90×–11.39× depending only on denominator |
| 4 | `line.lower()` is an English-only transformation | measurement-validity issue (not a crash) | 5.887 → 6.059 |
| 5 | `add_special_tokens=False` | **suspicious but correct** | would be +~18% on HF tokenizers if wrong |
| — | `random.seed(1337)` unused; NFC normalization | harmless / correct | 0% |

---

## Finding 1 (code bug): `words = line.split(" ")`

`fertility.py:62` splits on the single ASCII space. Consecutive spaces (common
in real text) produce empty-string tokens that inflate the word count and
therefore *deflate* fertility.

Evidence — the toy corpus already contains this:

```
eng line 7 : 'Please keep the books  in the cupboard.'  split(' ')=8  split()=7
hin line 10: 'किताबें  अलमारी में रखी हैं।'              split(' ')=6  split()=5
```

Effect of changing only `split(" ")` → `split()` (everything else at v0):

| corpus | eng macro | hin macro | ratio |
|---|---|---|---|
| toy, v0 | 1.2652 | 7.4485 | **5.887** |
| toy, fixed | 1.2831 (+1.41%) | 7.5985 (+2.01%) | **5.922** |
| FLORES devtest, v0 | 1.2874 | 7.8651 | 6.109 |
| FLORES devtest, fixed | 1.2874 (+0.00%) | 7.8669 (+0.02%) | 6.111 |

Interpretation: a real bug, but small — it moves the headline ratio by +0.6%
on the toy and ~0% on clean parallel text. It is **not** the reason the report
is wrong. (`str.split()` is also the only correct choice for scripts/whitespace
other than U+0020.)

## Finding 2 (aggregation bug): macro-average of per-line ratios

`fertility.py:66-67` returns the *mean of per-line ratios*. The corpus-level
quantity is `Σtokens / Σwords`. Averaging ratios weights a 2-word line the same
as a 50-word line, so short lines dominate. The docstring says "averaged over
lines", so the code does what it says — but it computes a per-line average,
not the corpus ratio.

| corpus | macro (v0) | micro (aggregate) | effect |
|---|---|---|---|
| toy | 1.2652 / 7.4485 → 5.887 | 1.2692 / 7.5246 → 5.928 | ratio +0.7% |
| FLORES devtest | 1.2874 / 7.8651 → 6.109 | 1.2782 / 7.8265 → **6.123** | ratio +0.2% |

Interpretation: real but second-order here. On FLORES it is small because
sentences are homogeneous; on a corpus mixing one-word and long sentences it
would be larger. Not the main error.

## Finding 3 (conceptual problem): "tokens per word" is not content-constant

The code computes exactly what the docstring promises: tokens per whitespace
word. The problem is that **a word is not a constant unit of content across
languages**, so the ratio is not a ratio of like quantities. The same is true
of graphemes, code points and bytes. Only a *parallel sentence* (identical
meaning, professionally translated) holds content constant.

Evidence — FLORES devtest, same corpus, same tokenizer (gpt2), only the
denominator changes:

| denominator | eng | hin | hin/eng |
|---|---|---|---|
| whitespace word | 1.235 | 7.826 | 6.34× |
| grapheme cluster (`\X`) | 0.205 | 2.335 | **11.39×** |
| UTF-8 byte | 0.205 | 0.595 | **2.90×** |
| parallel sentence | 26.72 | 198.31 | **7.42×** |

A **3.9× spread** in the "Hindi penalty" from the denominator alone. The byte
denominator is the most misleading: because Indic scripts cost ~3 UTF-8 bytes
per character, a stronger tokenizer pushes it *below* 1× (o200k Hindi =
0.62×, IndicBERTv2 = 0.46×), i.e. per-byte Indic would look *cheaper* — an
encoding artifact, not content.

The report's `tok/char` column (line 63: `chars = len(line)`) is the
**code-point** denominator, explicitly not a script-neutral unit. Measure:
Hindi has 131,180 code points but only 85,957 grapheme clusters on FLORES
devtest — a **+52.6% inflation** (Devanagari combining matras are separate code
points). English has 131,966 of each. So `tok/char` silently applies a
different unit to each language. REPORT_v0's "the two metrics agree, so the
result is robust" is exactly backwards: both share a denominator that is not
comparable across scripts.

## Finding 4 (validity issue, *not* a bug): `line.lower()`

`fertility.py:60` lowercases before tokenizing. It is intentional
("casing doesn't add noise"), and it cannot crash, so calling it a bug would be
overreach. But it is applied to *case-bearing English only* (Indic scripts have
no case), so it is an asymmetric transformation whose size is corpus- and
tokenizer-dependent:

| corpus | effect of lowercasing on eng | on hin | ratio |
|---|---|---|---|
| toy | 1.2293 → 1.2652 (+2.9%) | 0% | 6.059 → 5.887 |
| FLORES devtest | 1.2444 → 1.2874 (+3.3%) | 0% | 6.320 → 6.109 |

The direction here happens to be *conservative* (lowercasing makes English
look slightly worse, i.e. shrinks the Hindi penalty), but the mechanism is a
corpus artifact: on the toy corpus it is driven almost entirely by the acronym
`NASA` (1 GPT-2 token) vs `nasa` (2 tokens). For production cost modelling,
tokenize raw text; do not lowercase.

## Finding 5 (suspicious but correct): `add_special_tokens=False`

`fertility.py:33` passes `add_special_tokens=False` for HF tokenizers. This
*looks* like it undercounts, but it is correct: content-token fertility must
exclude wrapper tokens that the serving stack adds, otherwise every line pays a
fixed +1/+2 penalty that varies only by line count.

Evidence (`xlm-roberta-base`):

```
'Bengaluru International Airport handled...'  False=11  True=13  (+<s>,</s>)
'The Quarterly Review meeting moved to Thu...' False=10  True=12
'I bought this book yesterday from a small...' False=13  True=15
```

Turning it on would inflate every per-line fertility by `2/words` (~+18% on
short lines) with no relation to language. The only nuance: a real request does
pay for ~1 BOS token, so absolute cost accounting must add the serving
wrapper — a constant ≈ +1 token/request, negligible against the multi-×
cross-language deltas. Also correct: `NFC` normalization (`fertility.py:49`)
and the unused `random.seed(1337)` (dead, harmless).

## What the report should have concluded

The 5.89× headline is a **GPT-2-specific, word-denominator, macro-averaged**
number on a 10-line non-parallel toy set. Each of those four choices moves the
number, and the tokenizer choice moves it far more (Finding 3, A3). REPORT_v0's
root-cause claim — "Hindi simply has more Unicode characters per word, so any
tokenizer will struggle; this is a property of the script, not the tokenizer" —
is **falsified** in A3: on identical text the Hindi penalty is 7.42× for GPT-2
but 1.17× for an Indic tokenizer.
