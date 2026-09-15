import os
import unicodedata
import regex  

try:
    import tiktoken

    ENC = tiktoken.get_encoding("gpt2")
    ENCODE = ENC.encode
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"need tiktoken: {exc}")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOY = os.path.join(_HERE, "..", "..", "starter_kit", "corpus_sample")
ENG = os.path.join(_TOY, "eng_sample.txt")
HIN = os.path.join(_TOY, "hin_sample.txt")


def read_lines(path, normalize="NFC"):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if normalize:
                line = unicodedata.normalize(normalize, line)
            lines.append(line)
    return lines


PER_LINE = {}  # scratch diagnostics


def analyze_orig(lines, encode):
    fl, tc = [], []
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        words = line.split(" ")
        chars = len(line)
        fl.append(len(tokens) / len(words))
        tc.append(len(tokens) / chars)
    return sum(fl) / len(fl), sum(tc) / len(tc)


def metrics(lines, encode, *, lower=True, split_mode="space", denominator="tok/word"):
    tot_tok = tot_word = tot_gc = tot_cp = 0
    macro_den = 0.0
    n = len(lines)
    for line in lines:
        if lower:
            line = line.lower()
        tokens = encode(line)
        if split_mode == "space":
            words = line.split(" ")
            nw = len(words)
        else:  # generic whitespace
            nw = len(line.split())
        gc = len(regex.findall(r"\X", line))       # user-perceived chars
        cp = len(line)                              # unicode code points
        macro_den += (len(tokens) / nw) if nw else 0.0
        tot_tok += len(tokens)
        tot_word += nw
        tot_gc += gc
        tot_cp += cp
    micro = {
        "tok/word": tot_tok / tot_word,
        "tok/grapheme": tot_tok / tot_gc,
        "tok/codepoint": tot_tok / tot_cp,
    }[denominator]
    return {
        "micro": micro,
        "macro_perline_word": macro_den / n,
        "tot_tok": tot_tok,
        "tot_word": tot_word,
        "tot_gc": tot_gc,
        "tot_cp": tot_cp,
    }


def report(lang, path):
    lines = read_lines(path)
    orig_f, orig_c = analyze_orig(lines, ENCODE)
    print(f"\n===== {lang} ({path}) : {len(lines)} lines =====")
    print(f"v0 baseline: fertility={orig_f:.4f} tok/word   tok/char={orig_c:.4f}")

    # --- Flaw 1: split(' ') vs split() --------------------------------
    empty = sum(len(l.split(" ")) - len(l.split()) for l in lines)
    m_space = metrics(lines, ENCODE, split_mode="space", denominator="tok/word")
    m_gen = metrics(lines, ENCODE, split_mode="generic", denominator="tok/word")
    print("\n[F1] split(' ') vs split()")
    print(f"     spurious empty 'words' from split(' '): {empty}")
    print(f"     micro tok/word  split(' ') = {m_space['micro']:.4f}"
          f"   split() = {m_gen['micro']:.4f}"
          f"   delta = {100*(m_gen['micro']-m_space['micro'])/m_space['micro']:+.2f}%")

    # --- Flaw 2: macro (per-line mean) vs micro (aggregate) -----------
    print("\n[F2] macro-average (per-line mean) vs micro-average (sum/sum)")
    print(f"     macro (reported style) = {m_space['macro_perline_word']:.4f}")
    print(f"     micro (aggregate)      = {m_space['micro']:.4f}"
          f"   delta = {100*(m_space['macro_perline_word']-m_space['micro'])/m_space['micro']:+.2f}%")

    # --- Flaw 3: lower() asymmetry ------------------------------------
    m_low = metrics(lines, ENCODE, lower=True, denominator="tok/word")
    m_raw = metrics(lines, ENCODE, lower=False, denominator="tok/word")
    print("\n[F3] lower() vs original case")
    print(f"     tok/word lowered = {m_low['micro']:.4f}"
          f"   original = {m_raw['micro']:.4f}"
          f"   delta = {100*(m_raw['micro']-m_low['micro'])/m_low['micro']:+.2f}%")

    # --- Denom: grapheme vs codepoint ---------------------------------
    print("\n[F4] denominator: codepoint len() vs grapheme clusters")
    print(f"     tot_codepoints = {m_space['tot_cp']}   tot_graphemes = {m_space['tot_gc']}"
          f"   inflation = {100*(m_space['tot_cp']-m_space['tot_gc'])/m_space['tot_gc']:+.2f}%")
    print(f"     tok/codepoint = {m_space['tot_tok']/m_space['tot_cp']:.4f}"
          f"   tok/grapheme = {m_space['tot_tok']/m_space['tot_gc']:.4f}")

    return {"fert": orig_f, "tpc": orig_c}


if __name__ == "__main__":
    # ---- diagnostic: show exact lines with double spaces
    print("LINES WITH MULTIPLE CONSECUTIVE SPACES")
    for lang, path in (("eng", ENG), ("hin", HIN)):
        for i, l in enumerate(read_lines(path), 1):
            if "  " in l:
                print(f"  {lang} line {i}: {l!r}  -> split(' ')={len(l.split(' '))} tokens,"
                      f" split()={len(l.split())}")
    e = report("eng", ENG)
    h = report("hin", HIN)
    print("\n===== RATIOS (baseline vs each correction) =====")
    print(f"v0 reported hindi/eng fertility ratio = {h['fert']/e['fert']:.3f}")
    els = read_lines(ENG)
    hls = read_lines(HIN)
    for sm, label in (("space", "split(' ')"), ("generic", "split()")):
        for low, lab2 in ((True, "lower"), (False, "rawcase")):
            em = metrics(els, ENCODE, lower=low, split_mode=sm)
            hm = metrics(hls, ENCODE, lower=low, split_mode=sm)
            print(f"  micro ratio [{label}, {lab2}]"
                  f"= {hm['micro']/em['micro']:.3f}")
    for low in (True, False):
        em = metrics(els, ENCODE, lower=low)
        hm = metrics(hls, ENCODE, lower=low)
        print(f"  macro ratio [{'lower' if low else 'rawcase'}] ="
              f" {hm['macro_perline_word']/em['macro_perline_word']:.3f}")
