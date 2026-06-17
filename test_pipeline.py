"""Checagem runnable da pipeline (sem framework): invariantes + regressao no adam.
Rodar: python test_pipeline.py  (precisa de _sk_yolo11x.npy + adam_gt.csv)."""
import os
import numpy as np

import mechanics as mc
import detect
import classify as C
import stance as st
import eval as ev


def test_mechanics_shape():
    win = np.random.RandomState(0).rand(25, 17, 2)
    f = mc.features(win, hand=0)
    assert f.shape == (len(mc.FEATURE_NAMES),), "features com tamanho errado"
    assert np.isfinite(f).all(), "features com NaN/inf"


def test_detect_invariants():
    # 3 "golpes" sinteticos: punho estende e volta 3x
    T = 200
    sk = np.zeros((T, 17, 2))
    sk[:, 5] = [0.4, 0.3]; sk[:, 6] = [0.6, 0.3]            # ombros
    sk[:, 11] = [0.42, 0.6]; sk[:, 12] = [0.58, 0.6]        # quadris
    sk[:, 7] = [0.38, 0.4]; sk[:, 8] = [0.62, 0.4]          # cotovelos (guarda)
    sk[:, 9] = [0.4, 0.35]                                  # punho esq (guarda)
    sk[:, 10] = [0.6, 0.35]                                 # punho dir (guarda, perto do ombro)
    for c in (40, 100, 160):                                # extensoes do punho esq
        for t in range(c - 8, c + 9):
            sk[t, 9] = [0.4 - 0.30 * (1 - abs(t - c) / 8), 0.35]   # estende p/ longe do ombro
    spans = detect.detect_punches(sk)
    assert len(spans) == 3, f"esperado 3 golpes, deu {len(spans)}"
    for on, pk, off in spans:
        assert on <= pk <= off, "span mal-formado (onset<=peak<=offset)"
    for i in range(1, len(spans)):
        assert spans[i][0] > spans[i - 1][2], "spans sobrepostos"


def test_classify_valid():
    win = np.random.RandomState(1).rand(30, 17, 2)
    cls = C.classify_punch(win, (0, 29), stance=1)
    assert cls in C.MAP6.values(), f"classe invalida: {cls}"


def test_adam_regression():
    if not os.path.exists("_sk_yolo11x.npy"):
        print("  (pula regressao adam: _sk_yolo11x.npy ausente)")
        return
    sk = np.load("_sk_yolo11x.npy").astype(np.float64)
    r = ev.evaluate(sk, ev.load_gt("adam_gt.csv"))
    assert r["acc6"] >= 0.65, f"regressao 6-classes adam caiu: {r['acc6']}"
    assert r["matched"] == 18, f"deteccao adam != 18/18: {r['matched']}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("OK")
