# Part B — Capacity reconciliation

All arithmetic is reproduced by `partB/capacity.py` (output: `partB/results.txt`).

---

## B1 — KV bytes/token and max concurrent 4096-token sequences

### (a) Exact KV bytes per token

Spec: 28 layers, 8 KV heads (GQA), head_dim 128, fp16 cache (2 B), K and V
stored per layer.

```
per layer per token = 2 (K,V) × 8 kv_heads × 128 head_dim × 2 bytes = 4,096 B
total per token     = 4,096 B × 28 layers                          = 114,688 B
                    = 112 KiB/token  (0.109375 MiB)
```

Note the GQA saving: without GQA (24 Q heads) it would be `2×24×128×2×28 =
344,064 B/token`, 3× larger. The spec's `KV heads (GQA) = 8` is the relevant
number.

### (b) Max concurrent 4096-token sequences

The trap is forgetting model weights. They are not "overhead":

```
weights (fp16) = 4.2e9 params × 2 B                =  8.400e9 B
usable         = 0.92 × 24e9 B (L4, decimal GB)    = 22.080e9 B
KV budget      = usable − weights − 1.6e9 overhead = 12.080e9 B
token capacity = 12.080e9 / 114,688 B per token    = 105,329 tokens
4096-token seqs= 105,329 / 4096                     = 25.72  → 25 concurrent
```

**≈25.7 (integer-safe: 25) concurrent 4096-token sequences.**

Unit check: if "24 GB" were 24 GiB (25.77e9 B), capacity would be ~29.2 seqs —
but the log below rejects that. The log's utilization at batch 24 is 0.93;
`24/25.72 = 0.933` under decimal GB, versus `24/29.2 = 0.82` under GiB. The log
therefore resolves the GB/GiB ambiguity in favour of decimal GB.

### Reconciliation with the log

Predict `kv_cache_util = batch × seq_len / 105,329` and compare:

| batch | prompt | gen | seq_len | predicted util | logged util | preempted |
|---|---|---|---|---|---|---|
| 1 | 512 | 256 | 768 | 0.007 | 0.01 | 0 |
| 4 | 512 | 256 | 768 | 0.029 | 0.03 | 0 |
| 16 | 512 | 256 | 768 | 0.117 | 0.12 | 0 |
| 64 | 512 | 256 | 768 | 0.467 | 0.47 | 0 |
| 4 | 3584 | 512 | 4096 | 0.156 | 0.16 | 0 |
| 8 | 3584 | 512 | 4096 | 0.311 | 0.31 | 0 |
| 16 | 3584 | 512 | 4096 | 0.622 | 0.62 | 0 |
| 24 | 3584 | 512 | 4096 | 0.933 | 0.93 | 0 |
| 32 | 3584 | 512 | 4096 | 1.244 | 0.97 | 7 |
| 48 | 3584 | 512 | 4096 | 1.867 | 0.97 | 23 |

Predicted utilization matches **all 11 non-preempted rows in the CSV** to 2 dp. Rows 32/48
exceed 1.0, so the scheduler cannot fit them: utilization pins at 0.97 and
sequences are preempted (7 and 23). The capacity model is confirmed across both
prompt-length sweeps.

---

## B2 — The long-context (prompt=3584) throughput anomaly

### Observation

Reported throughput rises with batch through 24, then **falls** at 32 and 48,
even though more requests were submitted:

| batch | reported tok/s | change | goodput tok/s | ttft p50 ms | e2e p95 ms | kv util | preempted |
|---|---|---|---|---|---|---|---|
| 4 | 565.4 | | 70.7 | 483.2 | 32,673 | 0.16 | 0 |
| 8 | 902.6 | +59.6% | 112.8 | 519.0 | 39,983 | 0.31 | 0 |
| 16 | 1311.4 | +45.3% | 163.9 | 498.3 | 54,602 | 0.62 | 0 |
| 24 | **1607.4** | +22.6% | **200.9** | 500.5 | 69,221 | 0.93 | 0 |
| 32 | 1384.0 | **−13.9%** | 173.0 | 636.9 | 97,466 | 0.97 | 7 |
| 48 | 1298.5 | −6.2% | 162.3 | 955.4 | 105,428 | 0.97 | 23 |

The anomaly: throughput *peaks* at batch 24 and degrades. Naive "throughput
scales with batch" predicts batch 48 ≫ batch 24; the log shows the opposite,
and the short-prompt sweep (up to batch 64, kv 0.47) does **not** show it — so
it is specific to the long-context sweep.

### Mechanism

**MEASURED (from the log):** batch 32 needs `32 × 4096 = 131,072` KV tokens vs
a capacity of 105,329 (B1), so it cannot all be resident. `preempted_seqs` is
non-zero exactly when this happens (0 through batch 24, 7 at batch 32, 23 at
batch 48), `kv_cache_util` saturates at 0.97, `ttft`/`e2e_p95` inflate
(500→955 ms and 69→105 s), and throughput falls.

**INFERRED (standard vLLM behaviour, not directly logged in `bench_log.csv`):**
preempted sequences are re-scheduled and their prompts **re-prefilled**, and
that recompute consumes compute/bandwidth without producing output — i.e.
preemption thrashing. This is the causal step, and it is inferred, not measured
here; B4 specifies the counter (`vllm:num_preemptions_total`, and
`vllm:prompt_tokens_total` vs the expected `num_requests × prompt_len`) that
would confirm or falsify it.

The short-prompt rows never cross the capacity limit, so they never preempt —
which is why the anomaly appears only at long context. The system is
**KV-capacity-bound**, not compute-bound.

### Proposed change (config) and predicted effect — *prediction, unverified*

Cap admission so that `max_num_seqs ≤ floor(105,329 / max_model_len) = 25`
(the log's validated-safe point is batch 24). Concretely set `max_num_seqs = 24`
for `max_model_len = 4096`.

Prediction, using the measured no-preemption batch-24 cost of 61.16 s per
24-request wave:

```
48 requests as 2 waves ≈ 2 × 61.16 s = 122.3 s
predicted reported tok/s = 48 × 4096 / 122.3 = 1,607  (vs 1,298.5 observed, +23.8%)
predicted goodput        = 48 × 512  / 122.3 =   201  (vs 162.3, +23.8%)
predicted preempted_seqs = 0                          (vs 23)
```

Trade-off and falsifier: this trades tail latency under bursts (some requests
queue) for throughput and stability. The prediction is **falsified if** the
capped run still shows `preempted_seqs > 0` or does not recover ≥1,500 tok/s on
a 48-request burst. (An alternative that increases capacity rather than capping
it: fp8 KV cache halves bytes/token to 57,344, doubling capacity to ~51
sequences, so batch 48 fits at 0.93 util — but it changes KV precision and its
throughput effect is not yet quantified.)

---

## B3 — The misread column and honest goodput

### The misread column is `reported_tok_s`

`reported_tok_s` is **total processed tokens per second including prefill**, not
output tokens:

```
reported_tok_s = num_requests × (prompt_len + gen_len) / wall_clock_s
```

This reproduces every row of `bench_log.csv` (max abs error 0.19 tok/s across
13 rows, i.e. within the CSV's 1-decimal rounding). Example, batch 2 short:
`2×(512+256)/11.61 = 132.3`, exactly the logged 132.3.

Because longer prompts add many prefill tokens to the numerator, comparing
"long vs short" on this column **manufactures** the conclusion "longer prompts
give better throughput". The report's batch-16 comparison:

| batch 16 | reported (misread) | goodput = n×gen/wall |
|---|---|---|
| short (prompt 512) | 883.2 | **294.5 tok/s** |
| long (prompt 3584) | 1311.4 | **163.9 tok/s** |

The honest output metric **reverses** the conclusion: short prompts have ~1.8×
*higher* output goodput. The "batch 48 ≈ 3200 tok/s" claim compounds two errors:
it scales linearly from a best-observed 1607 (which is batch 24, not batch 16)
and it uses the inflated reported column; the actual batch-48 row is 1298.5
reported / **162.3 goodput**.

### Honest goodput, batch-24 long-prompt row — two independent derivations

Row: `num_requests=24, prompt_len=3584, gen_len=512, wall_clock_s=61.16,
reported_tok_s=1607.4`.

```
D1  output tokens / wall      = 24 × 512 / 61.16            = 200.92 tok/s
D2  reported × gen/(prompt+gen)= 1607.4 × 512/4096 = 1607.4 × 0.125 = 200.93 tok/s
D3  reported − prefill rate    = 1607.4 − (24×3584/61.16) = 1607.4 − 1406.41 = 200.99 tok/s
```

D1 and D2 agree to 0.01 tok/s (D3 to 0.07); **honest goodput ≈ 200.9 output
tok/s**, 8× lower than the reported 1607.4.

### What the report should have said

"`reported_tok_s` includes prefill tokens and must not be read as output
throughput. Output goodput for the long-context sweep peaks at batch 24
(≈201 tok/s) and *falls* at batch 32 (173) and 48 (162) because the KV cache is
exhausted and sequences are preempted. Short prompts actually deliver higher
output goodput at every batch. Capacity planning must use goodput and the
KV-capacity limit, not the reported counter."

---

## B4 — One serving-stack counter to confirm the B2 mechanism

Pull **`vllm:num_preemptions_total`** (preemptions/sec) together with
`vllm:gpu_cache_usage_perc`. Expected observation: the preemption rate is **flat
at zero** for every run with `batch × seq_len ≤ 105,329` (all short-prompt rows
and the long-prompt rows up to batch 24, where `gpu_cache_usage_perc < ~0.93`),
then becomes **non-zero and step-increases** the moment utilization saturates
(0.97) at batch 32 and 48 — temporally aligned with the throughput drop. A
sharper confirmation of *recomputation* is to compare engine-side
`vllm:prompt_tokens_total` against the harness's expected
`num_requests × prompt_len`: if preempted prompts are re-prefilled, measured
prompt tokens will exceed the expected 172,032 for the batch-48 row. If
preemptions saturate but `prompt_tokens_total` matches expectations, the
recompute part of the mechanism is falsified.
