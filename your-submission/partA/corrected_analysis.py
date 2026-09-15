import json
import sys
import unicodedata

import regex
import tiktoken
from transformers import AutoTokenizer

CORPUS = "corpus/flores200"  # run from partA/
LANGS = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu", "mal_Mlym"]
LANG_SHORT = {
    "eng_Latn": "eng", "hin_Deva": "hin", "kan_Knda": "kan",
    "tam_Taml": "tam", "tel_Telu": "tel", "mal_Mlym": "mal",
}


def load_tokenizers():
    tiks = {}
    for name in ("gpt2", "cl100k_base", "o200k_base"):
        try:
            enc = tiktoken.get_encoding(name)
            tiks[name] = lambda s, e=enc: e.encode(s)
        except Exception as exc:
            print(f"  [skip tiktoken {name}: {exc}]")
    for name in ("xlm-roberta-base", "ai4bharat/IndicBERTv2-MLM-only"):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            tiks[name] = lambda s, t=tok: t.encode(s, add_special_tokens=False)
        except Exception as exc:
            print(f"  [skip hf {name}: {exc}]")
    return tiks


def load_corpus(split):
    data = {}
    for lang in LANGS:
        path = f"{CORPUS}/{split}/{lang}.txt"
        with open(path, encoding="utf-8") as f:
            lines = [unicodedata.normalize("NFC", ln.strip()) for ln in f]
        data[lang] = [ln for ln in lines if ln]
    n = [len(v) for v in data.values()]
    assert len(set(n)) == 1, f"corpus not parallel: {n}"
    return data, n[0]


def analyze(data, n, encode):
    out = {}
    for lang, lines in data.items():
        tot = {"tokens": 0, "words": 0, "graphemes": 0, "bytes": 0,
               "codepoints": 0, "per_sentence_tokens": []}
        for s in lines:
            t = len(encode(s))
            tot["tokens"] += t
            tot["words"] += len(s.split())
            tot["graphemes"] += len(regex.findall(r"\X", s))
            tot["bytes"] += len(s.encode("utf-8"))
            tot["codepoints"] += len(s)
            tot["per_sentence_tokens"].append(t)
        out[lang] = tot
    return out


def fmt_ratio(x):
    return f"{x:.2f}x"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "devtest"
    data, n = load_corpus(split)
    print(f"corpus: FLORES-200 {split}, {n} parallel sentences, {len(data)} languages")
    toks = load_tokenizers()
    print("tokenizers:", list(toks))

    results = {}
    for tname, enc in toks.items():
        r = analyze(data, n, enc)
        results[tname] = r
        print(f"\n=================== tokenizer: {tname} ===================")
        hdr = (f"{'lang':<5}{'tok/sent':>10}{'tok/word':>10}{'tok/grapheme':>14}"
               f"{'tok/byte':>10}{'tok/codepoint':>14}")
        print(hdr)
        print("-" * len(hdr))
        base_tps = r["eng_Latn"]["tokens"] / n
        for lang in LANGS:
            d = r[lang]
            tps = d["tokens"] / n
            print(f"{LANG_SHORT[lang]:<5}{tps:>10.2f}"
                  f"{d['tokens']/d['words']:>10.3f}"
                  f"{d['tokens']/d['graphemes']:>14.3f}"
                  f"{d['tokens']/d['bytes']:>10.3f}"
                  f"{d['tokens']/d['codepoints']:>14.3f}")
        print("  ratios vs English (same parallel sentences):")
        for lang in LANGS[1:]:
            d = r[lang]
            ratios = {
                "tok/sentence": (d["tokens"] / n) / base_tps,
                "tok/word": (d["tokens"] / d["words"]) / (r["eng_Latn"]["tokens"] / r["eng_Latn"]["words"]),
                "tok/grapheme": (d["tokens"] / d["graphemes"]) / (r["eng_Latn"]["tokens"] / r["eng_Latn"]["graphemes"]),
                "tok/byte": (d["tokens"] / d["bytes"]) / (r["eng_Latn"]["tokens"] / r["eng_Latn"]["bytes"]),
                "tok/codepoint": (d["tokens"] / d["codepoints"]) / (r["eng_Latn"]["tokens"] / r["eng_Latn"]["codepoints"]),
            }
            # per-sentence paired ratio vs English
            import statistics
            paired = [a / b for a, b in zip(d["per_sentence_tokens"],
                                            r["eng_Latn"]["per_sentence_tokens"]) if b]
            print(f"    {LANG_SHORT[lang]:<4} " +
                  "  ".join(f"{k}={v:.2f}x" for k, v in ratios.items()) +
                  f"   | paired/sent mean={statistics.mean(paired):.2f}x"
                  f" median={statistics.median(paired):.2f}x")

    with open(f"results/corrected_{split}.json", "w") as f:
        json.dump({t: {l: {k: v for k, v in d.items() if k != "per_sentence_tokens"}
                       for l, d in r.items()} for t, r in results.items()}, f, indent=2)
    print(f"\nwrote results/corrected_{split}.json")


if __name__ == "__main__":
    main()
