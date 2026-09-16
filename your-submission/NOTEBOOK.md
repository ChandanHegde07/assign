# NOTEBOOK — chronological lab log

Format: **Hypothesis → Experiment → Result → Interpretation → Decision.**
Includes dead ends and corrections. Times are effort order, not wall-clock.

---

## 0. Environment and baseline reproduction

**H:** The starter kit should reproduce REPORT_v0 exactly before I change
anything.

**E:**
```
python3 -m venv .venv
.venv/bin/pip install tiktoken transformers datasets regex pandas pypdf
cd starter_kit
../.venv/bin/python fertility.py --corpus eng=corpus_sample/eng_sample.txt \
    --corpus hin=corpus_sample/hin_sample.txt --tokenizer gpt2
```

**R:** `eng 1.27 / 0.226`, `hin 7.45 / 1.579`, `hin is 5.89x the fertility of eng`.
Full precision: 1.2652 / 7.4485 → 5.887. Matches REPORT_v0.

**I:** Baseline is exactly reproducible. Good.

**D:** Preserve the original as `partA/fertility_v0_original.py`; build separate
instrumented scripts rather than editing it, so every claim has a
toggle-one-variable comparison.

---

## 1. Are the sample corpora actually parallel?

**H:** The PDF calls `corpus_sample/` "parallel line-by-line"; if so it is a
usable content-constant check.

**E:** `paste -d'|' eng_sample.txt hin_sample.txt | cat -n`.

**R:** Not parallel. eng line 1 = "Bengaluru International Airport…"; hin line 1
= "मुझे सुबह की चाय बहुत पसंद है।" ("I like morning tea"). Every line is a
different sentence.

**I:** REPORT_v0 compares unrelated sentences. Any per-word/per-char ratio is a
ratio of non-comparable content. Also spotted double spaces in lines 7 and 10.

**D:** Build a genuinely parallel corpus (A1). Flag the PDF's "parallel" claim
as inaccurate about the shipped files.

---

## 2. Corpus acquisition (dead ends)

**H:** Download FLORES-200.

**E/Dead ends:**
- `datasets.load_dataset("facebook/flores", …)` → `DatasetNotFoundError: gated`.
- `Muennighoff/flores200`, `gsarti/flores_101` → `RuntimeError: Dataset scripts
  are no longer supported`.
- `openlanguagedata/flores_plus` → gated.

**E (worked):** official tarball
`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (25,585,843 B,
sha256 `b8b0b767…11f6`); extracted only the six needed languages.

**R:** 1012 parallel `devtest` + 997 parallel `dev` lines per language.

**D:** Use `devtest` as the primary eval, `dev` as a disjoint robustness check.

---

## 3. A2 — audit of `fertility.py`

### 3a. `split(" ")`

**H:** `line.split(" ")` creates empty "words" on repeated spaces, deflating
fertility.

**E:** `partA/audit_fertility.py` (toggles one variable at a time).

**R:** eng line 7 → `<8 vs 7>` tokens; hin line 10 → `<6 vs 5>`. Fixing it:
toy ratio 5.887 → 5.922 (+0.6%); FLORES ratio 6.109 → 6.111 (~0%).

**I:** Real bug, small effect. Not the headline cause.

**D:** Report as a bug with its measured (small) effect; do not overstate.

### 3b. Macro vs micro

**H:** The docstring says "averaged over lines"; that weights short lines
equally, not by length.

**R:** toy 5.887 → 5.928 (macro→micro); FLORES 6.109 → 6.123.

**I:** Real statistical defect, second-order on homogeneous text.

### 3c. `lower()` — hypothesis *reversed by measurement*

**H (initial):** Lowercasing English should *reduce* its token count, inflating
the Hindi ratio — i.e. the report is biased high.

**E:** Recompute with `lower=False`.

**R:** toy eng 1.2652 → 1.2293 (raw is *lower*); ratio 5.887 → **6.059**, i.e.
the opposite direction. Root cause confirmed by direct tokenization:
`NASA` = 1 GPT-2 token but `nasa` = 2.

**I:** My hypothesis was wrong. The effect is real but corpus-specific (driven by
one acronym), and on these corpora lowercasing is conservatively *shrinking* the
ratio. Calling it a "bug" would have been an unverified flaw.

**D:** Downgrade to a measurement-validity issue, not a bug; state direction
honestly. This is the main "AI-sounded-confident-and-was-wrong" correction.

### 3d. Denominator (the conceptual problem)

**H:** The A3 hint ("what is the denominator supposed to hold constant?") points
at `tokens/word`. Test denominator sensitivity directly.

**R:** Same corpus/tokenizer (gpt2, FLORES devtest), Hindi/English changes with
denominator: byte 2.90×, word 6.34×, parallel sentence 7.42×, grapheme
**11.39×**. Hindi has 131,180 code points vs 85,957 graphemes (+52.6%) on
FLORES; English equal.

**I:** `tok/char` (code points) is not script-neutral; the whole fertility
family is non-content-constant. This is the conceptual defect. Byte's inversion
(Indic <1×) appears only with strong tokenizers (o200k 0.62×, IndicBERTv2
0.46×); gpt2's byte ratio is still 2.90×.

**D:** Adopt tokens per **parallel sentence** as the routing metric (A3).

### 3e. Suspicious but correct

**H:** `add_special_tokens=False` looks like it undercounts.

**R:** With `xlm-roberta-base`, True adds `+2` (`<s>`, `</s>`) to every line
(11→13, 10→12, 13→15). False is correct for content-token comparison. NFC
normalization and unused `random.seed(1337)` are also fine.

**D:** Classify as "suspicious but actually fine", with evidence.

### 3f. Tooling self-check (failed experiment)

Early `flaw_impacts.py` printed `graphemes eng=0 hin=0` — I had written
`r'\\X'` (literal backslash) instead of `r'\X'`. Caught by cross-checking the
grapheme totals against the code-point totals; fixed. Logged because it is
exactly the kind of silent number error the evidence rule is meant to catch.

---

## 4. A3 — corrected comparison

**H:** The "6×" is tokenizer- and denominator-specific.

**E:** `partA/corrected_analysis.py` — 5 tokenizers (
gpt2, cl100k_base, o200k_base, xlm-roberta-base,
ai4bharat/IndicBERTv2-MLM-only) × tokens per sentence / word / grapheme / byte /
code point, on FLORES devtest, then dev.

**R:** Hindi/English tokens per parallel sentence: **gpt2 7.42×, cl100k 4.77×,
o200k 1.57×, xlm-r 1.25×, IndicBERTv2 1.17×**. Dravidian under o200k ≈1.9–2.0×,
under IndicBERTv2 ≈1.05–1.10×. dev reproduces every ratio within ~1%. Paired
per-sentence mean/median agree with aggregate (not an averaging artifact).

**I:** Tokenizer choice moves the number 6.3×; the report's root-cause claim
("property of the script, not the tokenizer") is falsified.

**D:** Recommend routing by tokenizer, budget ~1.6× Hindi / ~2.0× Dravidian, with
an input-only/formal-domain caveat.

---

## 5. Part B — capacity reconciliation

**H1:** KV bytes/token = 2 × kv_heads × head_dim × 2 B × layers.

**R:** `2×8×128×2 = 4096` B/layer × 28 = **114,688 B (112 KiB)/token**.

**H2 (the trap):** Does capacity ignore weights? Test both.

**R:** Subtracting weights (8.4e9 B) and 1.6 GB overhead from 0.92×24e9 B gives a
KV budget of 12.08e9 B → **105,329 tokens → 25.72 concurrent 4096-token
sequences**. Predicted `kv_cache_util = batch×seq/105,329` matches the log on
**all 11 non-preempted rows in the CSV** (0.16/0.31/0.62/0.93 for batches 4/8/16/24). If
weights were ignored the prediction would be ~44 seqs and would not fit the log;
if "24 GB" were GiB it would be 29 seqs (batch 24 → 0.82, not 0.93). The log
resolves both ambiguities in favour of "decimal GB, weights subtracted".

**H3 (B2 anomaly):** Throughput should scale with batch; instead it peaks at 24.

**R:** batch 32 needs 131,072 KV tokens > 105,329 capacity → 7 preemptions
(23 at batch 48), util pins at 0.97, e2e p95 rises 69→105 s, throughput falls
1607→1384→1299. Short-prompt sweep never hits the limit and never preempts.

**I:** KV-capacity-bound preemption/recompute thrashing, not compute-bound.

**D:** Propose `max_num_seqs=24`; **prediction** (unverified): preemptions → 0,
48-request burst recovers to ~1,607 reported / ~201 goodput tok/s (+23.8%).

**H4 (B3):** The report misread a column.

**R:** `reported_tok_s = num_requests × (prompt_len + gen_len) / wall_clock_s`
reproduces all 13 rows (max err 0.19, within CSV rounding). It **includes
prefill**. Honest goodput for batch-24 long row = **200.9 tok/s** by three
independent derivations (D1 200.92, D2 200.93, D3 200.99). At batch 16 the
honest goodput is *higher* for short prompts (294.5) than long (163.9), reversing
the report's conclusion. Batch 48 actual goodput = 162.3, not ~3200.

**D:** State the misread column, the two derivations, and the corrected sentence.

---

## 6. Part C — decision

**H:** Under one A100, a 3-week deadline and a Hindi+Kannada-only reviewer,
training-based paths may be under-resourced relative to the validation bottleneck.

**I:** (a) and (b) can only be validated on 2 of 6 languages; (b) is dominated by
(a)/(c); (c) is testable on both reviewable languages within one reviewer-week
and is reversible via config.

**D:** Recommend (c) with a Day-7 numeric kill gate and a scoped (a) fallback for
Hindi+Kannada only. Full arithmetic and thresholds in `partC/memo.md`.

## 7. Revisions after self-review (strict pass)

**H:** Re-read the submission as an adversarial reviewer and check numbers
against the submission's own evidence.

**R / D — four fixes:**
1. Part C assumption #5 said "latency is dominated by prefill". Part B's own data
   contradicts it: ITL 96 ms × 512 ≈ 49 s of decode vs TTFT ≈ 0.5 s. Reframed to
   "throughput is dominated by prefill tokens; latency is decode-dominated".
2. A4 routing advice implied deploying an Indic tokenizer to a fixed model.
   A tokenizer cannot be swapped without retraining/relaunching the model;
   reworded to model selection/migration and flagged the un-estimated one-time
   cost.
3. B2 mechanism now labels the preemption evidence as **MEASURED** and the
   re-prefill/recompute step as **INFERRED**, with B4 as the confirming counter.
4. A3's byte-row statement "reverses the ranking" qualified: true for
   o200k/IndicBERTv2 (0.62×/0.46×), not for gpt2 (2.90×).

**Progress-note correction:** §4 says the denominator spread is "4×"; exact is
11.39/2.90 = 3.9×.

## Dead ends / things that did not survive

1. Gated/script-based FLORES mirrors (see §2).
2. The lowercasing hypothesis was refuted by measurement (§3c).
3. Grapheme-count regex bug produced zeros until cross-checked (§3f).
4. Considered framing macro-vs-micro as the headline A2 conceptual flaw;
   denominator sensitivity (3.9× spread) was the larger and more
   decision-relevant defect, so it took that slot.
5. Considered a byte-denominator comparison as "script-neutral"; it is not —
   it measures UTF-8 width and, for strong tokenizers, inverts the ranking.
6. Part C's "latency dominated by prefill" claim (§7): killed by the
   submission's own Part B numbers.
