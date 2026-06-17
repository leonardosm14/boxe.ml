"""Avaliacao contra um gabarito (start_frame,end_frame,class).

Duas medidas:
  1. CLASSIFICACAO (por golpe do GT): para cada golpe anotado, classifica o span e compara
     -> acuracia 6-classes, TIPO (reto/hook/upp) e MAO (lead/rear) + matriz de confusao.
  2. DETECCAO (pico): quantos golpes do GT foram pegos por >=1 pico e quantos picos extras.
"""
import argparse
import csv
import numpy as np

import detect
import classify as C
import stance as st
import mechanics as mc

TYPE_OF = {"Jab": 0, "Cross": 0, "Lead Hook": 1, "Rear Hook": 1, "Lead Uppercut": 2, "Rear Uppercut": 2}
LEAD = {"Jab", "Lead Hook", "Lead Uppercut"}
TN = ["reto", "hook", "upp"]


def load_gt(path):
    return [(int(r["start_frame"]), int(r["end_frame"]), r["class"].strip().title())
            for r in csv.DictReader(open(path))]


def evaluate(skeletons, gt):
    # stance global (dos golpes do GT)
    W = [np.zeros((25, 17, 2)) for _ in gt]
    for i, (s, e, _) in enumerate(gt):
        seg = skeletons[s:min(s + 25, e + 1)]; W[i][:len(seg)] = seg
    stance = st.clip_stance(np.array(W))

    okT = okH = ok6 = 0
    cT = np.zeros((3, 3), int)
    for s, e, cls in gt:
        pred = C.classify_punch(skeletons, (s, e), stance)   # span do GT (qualidade do metodo)
        ptype = TYPE_OF[pred]
        pside = "lead" if pred in LEAD else "rear"
        okT += ptype == TYPE_OF[cls]
        okH += (pside == "lead") == (cls in LEAD)
        ok6 += pred == cls
        cT[TYPE_OF[cls], ptype] += 1
    n = len(gt)

    # deteccao + ponta-a-ponta (span detectado casado ao golpe do GT por sobreposicao)
    spans = detect.detect_punches(skeletons)
    matched = e2e = 0
    for s, e, cls in gt:
        best = None
        for on, pk, off in spans:
            ov = min(e, off) - max(s, on)
            if ov > 0 and (best is None or ov > best[0]):
                best = (ov, (on, off))
        if best is None:
            continue
        matched += 1
        pred = C.classify_punch(skeletons, best[1], stance)
        e2e += pred == cls

    print(f"  CLASSIFICACAO por golpe do GT (n={n}, stance={'orthodox' if stance > 0 else 'southpaw'}):")
    print(f"    6-classes={ok6/n:.3f}  TIPO={okT/n:.3f}  MAO={okH/n:.3f}")
    print(f"    TIPO confusao (verdade x pred {TN}):")
    for t in range(3):
        print(f"      {TN[t]:<5} {cT[t].tolist()}")
    print(f"  DETECCAO: {matched}/{n} golpes pegos | {len(spans)} spans detectados")
    print(f"  PONTA-A-PONTA (deteccao + classificacao): {e2e}/{n} = {e2e/n:.3f}")
    return dict(acc6=ok6/n, accT=okT/n, accH=okH/n, matched=matched, e2e=e2e/n, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--skeleton", required=True)
    ap.add_argument("-g", "--gt", required=True)
    args = ap.parse_args()
    sk = np.load(args.skeleton).astype(np.float64)
    print(f"=== {args.skeleton} vs {args.gt} ===")
    evaluate(sk, load_gt(args.gt))


if __name__ == "__main__":
    main()
