import json
nv, hf = json.load(open("nv.json")), json.load(open("hf.json"))
for i, (a, b) in enumerate(zip(nv, hf)):
    n = min(len(a), len(b))
    div = next((j for j in range(n) if a[j] != b[j]), None)
    if div is None:
        print(f"prompt {i}: MATCH for all {n} compared tokens (nv={len(a)}, hf={len(b)})")
    else:
        print(f"prompt {i}: diverges at index {div}: nv={a[div]} hf={b[div]}")
        print(f"  shared prefix ids: {a[:div]}")
