"""Avaliação end-to-end da pipeline (tipo 3 classes + inferência lead/rear)
contra o ground truth anotado manualmente do vídeo adam (adam_gt.csv).

Usa o cache de esqueletos (tdets_adam.npy) — não roda YOLO. Roda o MESMO caminho
de código de boxe.py (assign_boxers -> dense -> smooth -> classify_events) e
compara cada evento detectado com o segmento do GT em que o pico dele cai.

Métricas:
  - tipo (3):      acurácia de Straight/Hook/Uppercut nos eventos casados
  - lead/rear:     acurácia do lado nos eventos casados
  - classe final:  acurácia das 6 classes (tipo E lado certos)
  - cobertura:     golpes do GT com pelo menos um evento detectado
  - falsos positivos: eventos cujo pico não cai em nenhum segmento do GT

Uso: python eval_leadrear.py  (ou jp run eval_leadrear.py no vlab)
"""
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

CACHE = sys.argv[1] if len(sys.argv) > 1 else "tdets_adam_clean.npy"

from boxe import classify_events, load_norm_stats, smooth_dense
from tracking import assign_boxers, build_dense_skeletons
from stance_utils import collapse_class

LEAD = {"Jab", "Lead Hook", "Lead Uppercut"}

# --- pipeline idêntica ao boxe.py, a partir do cache ---
# allow_pickle: cache gerado pelo nosso próprio boxe.py (mesmo padrão de lá) — confiável.
cached = np.load(CACHE, allow_pickle=True).item()
slots = assign_boxers(cached["frames_dets"], cached["total_frames"], cached["video_width"])
slot = max(slots, key=lambda s: s["present"].sum())          # adam: 1 boxeador
dense_raw = build_dense_skeletons(slot, cached["total_frames"])
dense = smooth_dense(dense_raw.copy())

model = load_model("modelo_boxe.keras")
mean, std = load_norm_stats()
# (on, pk, off, label, conf, side)
_, events = classify_events(dense, model, mean, std, skeletons_raw=dense_raw)

# --- ground truth ---
gt = pd.read_csv("adam_gt.csv")

matched, fp = [], []
used_rows = set()
for on, pk, off, label, conf, side in events:
    row = gt[(gt.start_frame <= pk) & (pk <= gt.end_frame)]
    if len(row) == 0:
        fp.append((pk, label))
        continue
    r = row.iloc[0]
    used_rows.add(int(r.name))
    matched.append({
        "gt": r["class"], "pred": label,
        "tipo_ok": collapse_class(r["class"]) == collapse_class(label),
        "lado_ok": (r["class"] in LEAD) == (label in LEAD),
        "full_ok": r["class"] == label,
    })

n = len(matched)
print("\n================ EVAL adam (n_gt=%d golpes) ================" % len(gt))
print(f"eventos detectados: {len(events)} | casados: {n} | falsos positivos: {len(fp)}")
print(f"cobertura do GT:    {len(used_rows)}/{len(gt)}")
if n:
    print(f"acc tipo (3):       {sum(m['tipo_ok'] for m in matched) / n:.3f}")
    print(f"acc lead/rear:      {sum(m['lado_ok'] for m in matched) / n:.3f}")
    print(f"acc classe final:   {sum(m['full_ok'] for m in matched) / n:.3f}")
print("\n--- detalhe ---")
for m in matched:
    flag = "OK " if m["full_ok"] else ("tipo" if m["tipo_ok"] else ("lado" if m["lado_ok"] else "ERR"))
    print(f"  [{flag}] gt={m['gt']:<14} pred={m['pred']}")
for pk, label in fp:
    print(f"  [FP ] pico {pk} -> {label}")

# --- Parte 2: lead/rear ISOLADO, sobre os segmentos do GT -------------------
# Mede SÓ a geometria lead/rear (stance.lead_rear), sem depender do detector de
# eventos nem do modelo de tipo: janela = frames anotados de cada golpe do GT.
from stance import lead_rear  # noqa: E402

ok = 0
print("\n--- lead/rear por segmento do GT (geometria isolada, esqueleto cru) ---")
for _, r in gt.iterrows():
    win = dense_raw[int(r.start_frame):int(r.end_frame) + 1]
    side = lead_rear(win)
    truth = "lead" if r["class"] in LEAD else "rear"
    hit = side == truth
    ok += hit
    print(f"  [{'OK' if hit else 'X '}] {r['class']:<14} gt={truth:<4} pred={side}")
print(f"lead/rear nos segmentos GT: {ok}/{len(gt)} = {ok / len(gt):.3f}")

# --- Parte 3: variante "tudo cru" (detector+modelo no esqueleto sem suavizar) --
# O modelo foi treinado nas janelas CRUAS do dataset; a suavização só existe na
# pipeline de inferência. Mede se remover a suavização melhora o TIPO no adam.
print("\n--- variante: detector+modelo no esqueleto CRU ---")
_, events_raw = classify_events(dense_raw, model, mean, std, skeletons_raw=dense_raw)
m2 = []
for on, pk, off, label, conf, side in events_raw:
    row = gt[(gt.start_frame <= pk) & (pk <= gt.end_frame)]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    m2.append({
        "tipo_ok": collapse_class(r["class"]) == collapse_class(label),
        "lado_ok": (r["class"] in LEAD) == (label in LEAD),
        "full_ok": r["class"] == label,
    })
if m2:
    n2 = len(m2)
    print(f"eventos: {len(events_raw)} | casados: {n2}")
    print(f"acc tipo (3): {sum(m['tipo_ok'] for m in m2) / n2:.3f} | "
          f"acc lead/rear: {sum(m['lado_ok'] for m in m2) / n2:.3f} | "
          f"acc final: {sum(m['full_ok'] for m in m2) / n2:.3f}")
