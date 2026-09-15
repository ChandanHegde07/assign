# A1 — Evaluation corpus

## Source & provenance

**FLORES-200** (Facebook Low Resource / `facebookresearch/flores`), the official
distribution tarball:

```
https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz   (25,585,843 bytes)
sha256: b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6
```

Downloaded and extracted only the needed files; copies live in
`partA/corpus/flores200/{devtest,dev}/<lang>.txt`.

## Size, languages, domain

| property | value |
|---|---|
| languages used | `eng_Latn`, `hin_Deva`, `kan_Knda`, `tam_Taml`, `tel_Telu`, `mal_Mlym` |
| Dravidian languages | Kannada, Tamil, Telugu, Malayalam (≥2 required) |
| `devtest` | 1012 sentence-aligned lines per language |
| `dev` | 997 sentence-aligned lines per language (disjoint; the primary run is `devtest`) |
| parallelism | verified sentence-aligned by construction and asserted in `corrected_analysis.py` |
| domain | professionally translated web/Wikipedia/news-style prose; formal register |
| encoding | UTF-8 |

The starter corpora were **not** used: they are 10 lines each and, contrary to
the PDF's "parallel line-by-line" description, are **not translation-parallel**
(e.g. eng line 1 is about airport traffic while hin line 1 is "I like morning
tea"). REPORT_v0 compares unrelated sentences, so it cannot support any
content-relative claim.

## Preprocessing

- strip leading/trailing whitespace; drop empty lines;
- Unicode **NFC** normalization;
- **no lowercasing** (production text is not lowercased; see A2 Finding 4);
- tokenize each sentence independently (no cross-sentence merges), matching how
  requests are served.

## Reproduce

```
cd partA
python corrected_analysis.py devtest   # -> results/corrected_devtest.json
python corrected_analysis.py dev       # robustness check
```

## What this corpus cannot establish

FLORES-200 measures **input encoding only**, on **formal, professionally
translated prose**. It cannot tell us: (1) how many *output* tokens a model
needs to *generate* the same reply per language — for a causal LM this can
differ from input ratios and is often the larger cost term; (2) behaviour on
**casual/conversational** Hindi or Dravidian text (the actual product
register) or on code-mixed Hinglish, which FLORES does not contain; (3) per
request cost, which also depends on the routed model's per-token price, quality
and prompt template; (4) anything about production session lengths or
sentiment. The 1012-sentence sample gives tight confidence intervals on the
*mean* token ratio for this domain (dev vs devtest agree within ~1%), but the
domain caveat, not sample size, is the binding limitation.
