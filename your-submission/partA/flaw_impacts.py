import statistics
import unicodedata

import regex
import tiktoken
from transformers import AutoTokenizer


def read_corpus(path):
    with open(path, encoding="utf-8") as f:
        return [unicodedata.normalize("NFC", ln.strip())
                for ln in f if ln.strip()]


TOY = {"eng": read_corpus("../../starter_kit/corpus_sample/eng_sample.txt"),
       "hin": read_corpus("../../starter_kit/corpus_sample/hin_sample.txt")}
FL = {"eng": read_corpus("corpus/flores200/devtest/eng_Latn.txt"),
      "hin": read_corpus("corpus/flores200/devtest/hin_Deva.txt")}

ENC = tiktoken.get_encoding("gpt2").encode


def macro(lines, *, lower=True, split_space=True):
    vals = []
    for ln in lines:
        s = ln.lower() if lower else ln
        toks = len(ENC(s))
        w = len(s.split(" ") if split_space else s.split())
        vals.append(toks / w)
    return statistics.mean(vals)


def micro(lines, *, lower=True, split_space=True, denom="word"):
    tk = w = g = b = cp = 0
    for ln in lines:
        s = ln.lower() if lower else ln
        tk += len(ENC(s))
        w += len(s.split(" ") if split_space else s.split())
        g += len(regex.findall(r"\X", s))
        b += len(s.encode("utf-8"))
        cp += len(s)
    d = {"word": w, "grapheme": g, "byte": b, "codepoint": cp}[denom]
    return tk / d


def line(title):
    print(f"\n### {title}")


for corpus_name, corpus in (("TOY (10 lines/lang)", TOY), ("FLORES devtest (1012)", FL)):
    print("=" * 72)
    print(f"CORPUS: {corpus_name}")
    print("=" * 72)
    e, h = corpus["eng"], corpus["hin"]

    v0e, v0h = macro(e), macro(h)
    line("v0 baseline (lower, split(' '), per-line mean)")
    print(f"  eng={v0e:.4f}  hin={v0h:.4f}  ratio={v0h/v0e:.3f}")

    line("FIX-1  split(' ') -> split()  [code bug: empty tokens]")
    f1e, f1h = macro(e, split_space=False), macro(h, split_space=False)
    print(f"  eng={f1e:.4f} ({100*(f1e-v0e)/v0e:+.2f}%)  "
          f"hin={f1h:.4f} ({100*(f1h-v0h)/v0h:+.2f}%)  "
          f"ratio={f1h/f1e:.3f} (was {v0h/v0e:.3f})")

    line("FIX-2  per-line mean -> aggregate token/word  [macro vs micro]")
    m2e, m2h = micro(e, split_space=False), micro(h, split_space=False)
    print(f"  eng={m2e:.4f} ({100*(m2e-f1e)/f1e:+.2f}% vs fixed-macro)  "
          f"hin={m2h:.4f} ({100*(m2h-f1h)/f1h:+.2f}%)  ratio={m2h/m2e:.3f}")

    line("FLAW-3  lower() removed  [English-only, corpus-dependent]")
    f3e, f3h = macro(e, lower=False), macro(h, lower=False)
    print(f"  eng={f3e:.4f} ({100*(f3e-v0e)/v0e:+.2f}%)  "
          f"hin={f3h:.4f} ({100*(f3h-v0h)/v0h:+.2f}%)  ratio={f3h/f3e:.3f}")

    line("FLAW-4  denominator: tok/word vs tok/grapheme vs tok/codepoint")
    print(f"  tok/word      eng={micro(e,denom='word'):.4f} "
          f"hin={micro(h,denom='word'):.4f} "
          f"ratio={micro(h,denom='word')/micro(e,denom='word'):.3f}")
    print(f"  tok/grapheme  eng={micro(e,denom='grapheme'):.4f} "
          f"hin={micro(h,denom='grapheme'):.4f} "
          f"ratio={micro(h,denom='grapheme')/micro(e,denom='grapheme'):.3f}")
    cp_e = sum(len(ln) for ln in e); cp_h = sum(len(ln) for ln in h)
    gr_e = sum(len(regex.findall(r"\X", ln)) for ln in e)
    gr_h = sum(len(regex.findall(r"\X", ln)) for ln in h)
    by_e = sum(len(ln.encode("utf-8")) for ln in e)
    by_h = sum(len(ln.encode("utf-8")) for ln in h)
    print(f"  tok/byte      eng={micro(e,denom='byte'):.4f} "
          f"hin={micro(h,denom='byte'):.4f} "
          f"ratio={micro(h,denom='byte')/micro(e,denom='byte'):.3f}")
    print(f"  (v0 tok/char uses code points: eng_cp={cp_e} hin_cp={cp_h}; "
          f"graphemes eng={gr_e} hin={gr_h}; utf8_bytes eng={by_e} hin={by_h}; "
          f"hin codepoint/grapheme inflation={100*(cp_h-gr_h)/gr_h:.1f}%)")

# ---- suspicious-but-correct: add_special_tokens=False --------------------
print("\n" + "=" * 72)
print("SUSPICIOUS-BUT-CORRECT: HF tokenizer add_special_tokens=False")
print("=" * 72)
tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
for ln in TOY["eng"][:3]:
    a = tok.encode(ln, add_special_tokens=False)
    b = tok.encode(ln, add_special_tokens=True)
    print(f"  {ln[:40]!r:44} False={len(a)} True={len(b)} "
          f"(+{len(b)-len(a)} special: {tok.convert_ids_to_tokens(b[:1]+b[-1:])})")
print("  -> add_special_tokens=False is CORRECT: special tokens are added by the")
print("     serving pipeline, not part of the text's token cost.")
