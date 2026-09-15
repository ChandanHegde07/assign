# AI_USAGE.md

This document briefly describes how I used AI during the assignment.

I used an AI assistant mainly for code scaffolding, brainstorming possible
explanations, and checking calculations. I treated its suggestions as
hypotheses rather than verified results. Whenever a conclusion affected the
submission, I checked it against the code, corpus, or benchmark data.

## How I Used AI

### Exploration and implementation

AI helped me move faster when exploring the existing code and building small
measurement scripts. In particular, it helped with the initial scaffolding of:

- `audit_fertility.py`
- `corrected_analysis.py`
- `capacity.py`

I then ran and modified these scripts locally as needed.

The main benefit was speed: I could test an idea quickly instead of writing
repetitive Python from scratch.

### Reasoning and cross-checking

I also used AI to explore possible explanations and alternative ways of
calculating results.

Examples included:

- possible causes of tokenizer-ratio changes,
- denominator choices,
- reconstructing `reported_tok_s`,
- KV-cache arithmetic,
- independent goodput calculations.

These suggestions were useful for deciding what to test, but the measured
result was what I used in the final analysis.

## Where AI Was Wrong

A useful part of the process was discovering that some plausible AI
explanations were wrong.

### Lowercasing

An initial hypothesis was that lowercasing might inflate the Hindi/English
ratio by reducing English token counts.

The experiment showed the opposite. Raw-case English tokenized lower on both
the toy and FLORES corpora, so lowercasing increased the English-side token
count and reduced the ratio:

    Toy:    5.887 → 6.059
    FLORES: 6.109 → 6.320

I kept the measured result rather than the original hypothesis.

### Denominator choice

Initially, I treated code points versus grapheme clusters mainly as a
Unicode implementation detail.

The experiments showed that the denominator could materially change the
cross-language comparison. This became an important part of the final
analysis.

### Grapheme-counting experiment

An early experiment produced `graphemes = 0` because of an escaping mistake
in the regex:

    r'\\X'

instead of:

    r'\X'

A cross-check against the code-point counts showed that the result was
implausible. I fixed the script and recorded the dead end in `NOTEBOOK.md`.

### Dataset acquisition

Some initially suggested FLORES routes did not work in the environment. I
checked the failures rather than treating the suggested routes as confirmed
sources and eventually used the official FLORES-200 tarball.

### KV-cache capacity

An early capacity calculation that did not account for model weights gave a
higher theoretical concurrency than the benchmark supported.

Comparing the calculation with the actual log exposed the inconsistency. The
final calculation subtracts model weights before estimating the available
KV-cache capacity.

### Part C latency assumption

A drafted Part C assumption said latency was dominated by prefill. The Part B
data did not support that: decode time was approximately `96 ms × 512 ≈ 49 s`,
while TTFT was about 0.5 s.

I corrected the assumption so that it was consistent with the benchmark.

## What I Focused on Defending

Rather than trying to treat every number in the repository as equally
important, I focused my understanding on four main evidence chains.

### 1. Tokenizer sensitivity

The corrected analysis showed that the Hindi/English token ratio changes
substantially with tokenizer choice:

    GPT-2       → 7.42×
    o200k       → 1.57×
    IndicBERT   → 1.17×

The claim I defend is not that one tokenizer is universally better. The
experiment shows that the original language-cost headline cannot be treated
as an intrinsic property of Hindi; tokenizer choice materially affects the
measurement.

### 2. Denominator sensitivity

Using the same underlying comparison, changing the denominator produced a
large change in the apparent language penalty:

    UTF-8 bytes   → 2.90×
    words         → 6.34×
    sentences     → 7.42×
    graphemes     → 11.39×

This is why I consider the denominator a central metric-design issue rather
than a minor reporting detail. For routing and cost, the comparison should
use the tokenizer actually deployed and a denominator that keeps the
underlying content comparable.

### 3. KV-cache capacity

From the model specification, I derived the KV-cache footprint as:

    2 × 8 × 128 × 2 × 28
    = 114,688 bytes/token

After accounting for model weights and other memory overhead, the estimated
KV-cache budget is approximately:

    105,329 tokens

For 4096-token sequences:

    105,329 / 4096 ≈ 25.7

or approximately 25 complete sequences.

I then compared this theoretical estimate with the serving benchmark. The
observed increase in preemptions and near-saturated KV-cache utilization at
higher batch sizes is consistent with the predicted memory boundary.

### 4. Reported throughput versus honest goodput

I reconstructed the `reported_tok_s` column as approximately:

    N × (prompt_tokens + generated_tokens) / wall_time

This explained the unusually large throughput numbers in the report.

For the batch-24 long-prompt case, I independently derived honest goodput
at approximately:

    200.9 tok/s

The independent calculations agreed closely, supporting the conclusion that
the report's approximately 3200 tok/s interpretation was misleading.

## What I Should Be Able to Explain in the Defense

The defense is not something I expect to approach by memorizing every line of
the repository. I should be able to reproduce and explain the four evidence
chains above from the submitted code and data.

In particular, I should be able to:

- reproduce the tokenizer comparison and explain why the ratio changes,
- explain why the denominator changes the interpretation,
- derive the KV-cache calculation from the model specification,
- explain how 114,688 bytes/token leads to approximately 105,329 KV tokens
  and 25 full 4096-token sequences,
- reconstruct the reported throughput column,
- derive the approximately 200.9 tok/s goodput,
- distinguish measured observations from inferred mechanisms and predicted
  effects,
- and explain how these findings support the final recommendations.

I do not claim that every conclusion in the repository is equally deep or that
I can recall every implementation detail without looking at the code. My
focus is on being able to reproduce and defend the main claims that drive the
audit.

## Final Note

AI was useful as a coding and reasoning accelerator. It helped me generate
scaffolding, explore hypotheses, and cross-check calculations, but it was not
treated as an authority.

Several AI-generated hypotheses were contradicted by experiments. In those
cases, I followed the measured evidence and revised the analysis.

The main lesson from using AI on this assignment was that a plausible
explanation is not evidence. The final claims are the ones I was able to
connect to reproducible experiments and calculations.