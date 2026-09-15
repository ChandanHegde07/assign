import json

SHORT = {"eng_Latn": "English", "hin_Deva": "Hindi", "kan_Knda": "Kannada",
         "tam_Taml": "Tamil", "tel_Telu": "Telugu", "mal_Mlym": "Malayalam"}
LANG = list(SHORT)

for split in ("devtest", "dev"):
    with open(f"results/corrected_{split}.json") as f:
        R = json.load(f)
    toks = list(R)
    print(f"\n## FLORES-200 {split}: tokens per parallel sentence"
          f" (ratio vs English in parentheses)\n")
    print("| tokenizer | " + " | ".join(SHORT[l] for l in LANG) + " |")
    print("|---|" + "---|" * len(LANG))
    for t in toks:
        base = R[t]["eng_Latn"]["tokens"]
        n = 1012 if split == "devtest" else 997
        cells = []
        for l in LANG:
            d = R[t][l]
            tps = d["tokens"] / n
            cells.append(f"{tps:.1f} ({d['tokens']/base:.2f}x)")
        print(f"| {t} | " + " | ".join(cells) + " |")
