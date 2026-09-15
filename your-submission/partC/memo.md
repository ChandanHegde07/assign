# Part C — Decision memo: making six Indic languages sound casual

**Decision: ship path (c) prompt engineering as the primary path, on a
pre-registered 7-day readout; hold (a) SFT as a scoped fallback for Hindi +
Kannada only; reject (b).** Rationale below is driven by the two hard
constraints — a 2-language reviewer and a 3-week launch — not by which method
is most attractive.

## Assumptions (explicit; falsify these first)

1. `FLM-4B-Instruct` (4.2B) is the model to be casualised; one A100-80GB is free
   for 2 weeks.
2. "No external API budget" means synthetic data / judging must run on the local
   A100 with an open multilingual instruct model as teacher (assumed available).
3. Main model already supports the 6 languages reasonably; the problem is
   *register*, not comprehension.
4. Reviewer = 10 h/week × 3 weeks = **30 h total**, Hindi + Kannada only. Tamil,
   Telugu, Bengali, Marathi have **no reviewer** → nothing shipped for them can
   be quality-validated.
5. Cost of a request ≈ tokens × price. Token **throughput** is dominated by
   prefill (Part B: 88% of long-prompt tokens are prefill), but per-request
   **latency** is dominated by decode (ITL 96 ms × 512 ≈ 49 s vs TTFT ≈ 0.5 s).

## Why not (a) or (b) first

- **(a) SFT** requires synthetic pairs for **6** languages but can only be
  validated on **2**. Any register/meaning error baked into the other 4 is
  invisible until launch. It also spends the A100 (teacher generation +
  training) for the whole 2 weeks before producing a shippable artifact.
- **(b) ≤1B rewriter** is dominated: zero-shot it is prompt engineering with an
  extra model and extra latency; fine-tuned it inherits (a)'s data/validation
  bottleneck *and* adds a second serving path. It is the most technically
  attractive and the worst expected-value under these constraints.
- **(c)** needs no training data, no GPU training, ships in days, and its failure
  mode is *immediately visible* on the two languages we can review.

## Back-of-envelope arithmetic

**Reviewer throughput.** Side-by-side blind preference at ~2–3 min/item →
**20–30 items/h → 200–300 items/week**. A 200-item eval (100 Hindi + 100
Kannada) costs ~7–10 h, i.e. one reviewer-week. This is the binding resource;
design the eval to fit it.

**Data volume if (a) is triggered.** 10k pairs/language × 2 validatable
languages = **20k pairs** (not 60k, because more is unverifiable). At ~400
tok/pair that is ~8M teacher-generated tokens; on the A100 at a conservative
~1,500 tok/s batched → **~1.5 h generation**. LoRA/SFT a 4B on 20k pairs × 2
epochs ≈ 16M tokens → **~3 h**. Training is not the bottleneck; review is.

**(c) serving cost.** Prompt pack (~500 tokens) added per request. Using the
Part B measured prefill rate (~1,406 tok/s on L4; A100 is faster) → **+≈0.36 s
TTFT estimate** (+500/1406), small next to the decode-dominated e2e (≈49 s at
gen_len 512); and ~500 × 112 KiB ≈ 57 MB extra KV per request (negligible
against the ~20 GB budget). No second model, no training.

## Success metric (numeric threshold)

On the frozen 200-item held-out set, blind, reviewer-judged:
**prompt-engineered output must be preferred as "casual and natural" in ≥70% of
side-by-side comparisons vs the current output, with meaning-drift/error flagged
in ≤3% of items.** Bands: <50% = fail; 50–70% = partial.

## Kill criterion (with deadline)

- **Day 7:** if the readout shows win-rate <50% **or** meaning-drift >5% on
  Hindi+Kannada → abandon (c) and start (a). If 50–70%, iterate prompts one more
  week with a hard stop.
- **Day 12 (if (a) started):** if the SFT model does not beat the best
  prompt-engineered baseline by **≥10 preference points** on the same eval →
  kill (a) and ship (c) best-effort, explicitly de-scoping Tamil/Telugu/Bengali/
  Marathi to "unvalidated".
- Launch gate (Day 15): do not ship (a) to any language without a reviewer pass;
  for unreviewed languages, ship only (c), which is reversible in a config flag.

## First experiment — Day 1

Build the frozen evaluation harness before building any solution:
1. 200 held-out prompts (100 Hindi, 100 Kannada) from real user traffic-like
   queries; reviewer writes the reference "casual" tone for a 50-item dev slice
   only (never the test set).
2. Generate three outputs per prompt: current system prompt, and three
   few-shot-casual system prompts (5 in-context exemplars each); freeze decoding
   params.
3. Blind, randomise, and have the reviewer do side-by-side preference + a
   meaning-preservation checkbox.
4. Read out win-rate and drift.

This is <10 h of the reviewer's week and decides the path by Day 7 with measured
evidence, not a preference for any technique. **Falsifier for the whole memo:**
if the base model cannot produce casual Hindi/Kannada even with few-shot
exemplars (win-rate <50%), the "no training needed" premise is false and the
plan must pivot to (a) — which is exactly the pre-registered trigger.
