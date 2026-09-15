import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "starter_kit", "bench", "bench_log.csv")

LAYERS = 28
KV_HEADS = 8          
HEAD_DIM = 128
KV_BYTES = 2          
PARAMS = 4.2e9
W_BYTES = 2           
GPU_BYTES = 24e9      
UTIL = 0.92
NON_KV_OVERHEAD = 1.6e9

D = 114688.0  


def kv_bytes_per_token():
    per_layer = 2 * KV_HEADS * HEAD_DIM * KV_BYTES   # K and V
    return per_layer * LAYERS, per_layer


def main():
    kv_per_tok, per_layer = kv_bytes_per_token()
    print("=" * 74)
    print("B1  KV-CACHE ARITHMETIC")
    print("=" * 74)
    print(f"  per layer per token = 2(K,V) x {KV_HEADS} kv_heads x {HEAD_DIM} dim"
          f" x {KV_BYTES} B = {per_layer} B")
    print(f"  x {LAYERS} layers                    = {kv_per_tok} B/token"
          f"  ({kv_per_tok/1024:.0f} KiB/token)")

    weights = PARAMS * W_BYTES
    budget = UTIL * GPU_BYTES - weights - NON_KV_OVERHEAD
    print(f"\n  weights (fp16)        = {PARAMS:.1e} x {W_BYTES} B = {weights:.3e} B")
    print(f"  usable @ util={UTIL}   = {UTIL} x {GPU_BYTES:.3e} = {UTIL*GPU_BYTES:.3e} B")
    print(f"  KV budget = usable - weights - overhead({NON_KV_OVERHEAD:.1e})"
          f" = {budget:.3e} B")
    tok_cap = budget / kv_per_tok
    print(f"  KV token capacity     = {budget:.3e} / {kv_per_tok} = {tok_cap:,.0f} tokens")
    print(f"  concurrent 4096-token seqs = {tok_cap:,.0f} / 4096 = "
          f"{tok_cap/4096:.2f}  -> floor {int(tok_cap//4096)}")
    print("  (if one instead treated '24 GB' as 24 GiB = 25.77e9 B, capacity"
          " would be 29.2 seqs;")
    print("   the log below rejects that: it matches 25.7 seqs, i.e. decimal GB.)")

    print("\n  RECONCILIATION: predicted kv_util = batch*seq_len / token_capacity")
    rows = list(csv.DictReader(open(CSV)))
    hdr = (f"  {'b':>3}{'prompt':>7}{'gen':>5}{'tot/seq':>8}{'pred_util':>10}"
           f"{'logged_util':>12}{'reported':>9}{'recomputed':>11}{'preempt':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        b = int(r["batch_size"])
        p = int(r["prompt_len"]); g = int(r["gen_len"])
        wall = float(r["wall_clock_s"])
        n = int(r["num_requests"])
        seq = p + g
        pred = b * seq / tok_cap
        rep = float(r["reported_tok_s"])
        rec = n * (p + g) / wall
        print(f"  {b:>3}{p:>7}{g:>5}{seq:>8}{pred:>10.3f}"
              f"{float(r['kv_cache_util']):>12.2f}{rep:>9.1f}{rec:>11.1f}"
              f"{int(r['preempted_seqs']):>8}")

    print("\n  -> reported_tok_s == num_requests*(prompt_len+gen_len)/wall_clock"
          "  (max abs err shown above)")
    err = max(abs(n * (int(r["prompt_len"]) + int(r["gen_len"])) / float(r["wall_clock_s"])
                  - float(r["reported_tok_s"]))
              for r in rows for n in [int(r["num_requests"])])
    print(f"     max abs error across all 13 rows = {err:.3f} tok/s")
    print("  -> reported_tok_s COUNTS PREFILL (prompt) TOKENS, not just output.")

    print("\n" + "=" * 74)
    print("B3  HONEST GOODPUT = generated tokens / wall clock")
    print("=" * 74)
    print(f"  {'b':>3}{'prompt':>7}{'gen':>5}{'reported':>10}{'goodput':>9}"
          f"{'prompt_share':>14}")
    for r in rows:
        b = int(r["batch_size"]); p = int(r["prompt_len"]); g = int(r["gen_len"])
        n = int(r["num_requests"]); wall = float(r["wall_clock_s"])
        rep = float(r["reported_tok_s"])
        good = n * g / wall
        print(f"  {b:>3}{p:>7}{g:>5}{rep:>10.1f}{good:>9.1f}"
              f"{p/(p+g):>13.0%}")

    r = [x for x in rows if x["batch_size"] == "24" and x["prompt_len"] == "3584"][0]
    b = int(r["batch_size"]); p = int(r["prompt_len"]); g = int(r["gen_len"])
    n = int(r["num_requests"]); wall = float(r["wall_clock_s"])
    rep = float(r["reported_tok_s"])
    d1 = n * g / wall
    d2 = rep * g / (p + g)
    prefill_rate = n * p / wall
    d3 = rep - prefill_rate
    print(f"\n  batch-24 / prompt-3584 / gen-512 row (wall={wall}s):")
    print(f"    D1  n*gen/wall            = {n}*{g}/{wall} = {d1:.2f} tok/s")
    print(f"    D2  reported*gen/(p+gen)  = {rep}*{g}/{p+g} = {d2:.2f} tok/s")
    print(f"    D3  reported - n*prompt/wall = {rep} - {prefill_rate:.2f}"
          f" = {d3:.2f} tok/s")
    print(f"    agree within {max(d1,d2,d3)-min(d1,d2,d3):.2f} tok/s"
          f"  -> honest goodput = {d1:.1f} tok/s (vs reported {rep:.1f})")

    print("\n" + "=" * 74)
    print("B2  LONG-CONTEXT ANOMALY (prompt=3584)")
    print("=" * 74)
    print(f"  {'b':>3}{'report':>9}{'goodput':>9}{'ttft':>8}{'itl':>7}"
          f"{'e2e_p95':>9}{'kv':>6}{'preempt':>8}{'gen/s':>8}")
    prev = None
    for x in rows:
        if int(x["prompt_len"]) != 3584:
            continue
        b = int(x["batch_size"]); wall = float(x["wall_clock_s"])
        good = int(x["num_requests"]) * int(x["gen_len"]) / wall
        rep = float(x["reported_tok_s"])
        delta = ""
        if prev is not None:
            delta = f"  ({100*(rep-prev)/prev:+.1f}% vs b={prevb})"
        prev, prevb = rep, b
        print(f"  {b:>3}{rep:>9.1f}{good:>9.1f}{float(x['ttft_ms_p50']):>8.1f}"
              f"{float(x['itl_ms_p50']):>7.1f}{float(x['e2e_ms_p95']):>9.1f}"
              f"{float(x['kv_cache_util']):>6.2f}{int(x['preempted_seqs']):>8}"
              f"{int(x['num_requests'])*int(x['gen_len'])/wall:>8.1f}{delta}")


if __name__ == "__main__":
    main()
