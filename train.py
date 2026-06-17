"""Treino do classificador de golpes (gera modelo_boxe.keras + norm_stats.npz).

Pipeline nova (ver PLANO_IMPLEMENTACAO.md), tudo medido em cross-video/LOVO:
  - features view-invariantes (de-roll + ângulos internos + razão de direção) — boxe_utils
  - máscara do padding de verdade (pooling só nos frames reais) — model.py
  - logit-adjusted softmax (balanced softmax) p/ a Rear Hook faminta — model.py
  - mirror augmentation com troca de mão (Lead<->Rear) — boxe_utils.mirror_windows
  - deep ensemble (média de softmax) p/ calibração sob shift
A pipeline de features é a MESMA do boxe.py (boxe_utils.feature_pipeline): treino == inferência.
"""
from gpupick import pick_and_set_gpu
pick_and_set_gpu()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter

from boxe_utils import (load_video, axis_stats, feature_pipeline, mirror_windows, augment_pose,
                        WINDOW_LEN)
from model import build_model, compile_model, predict_proba, predict_proba_ensemble, log_prior
import eval_utils

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

CLASSES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]
LID = {c: i for i, c in enumerate(CLASSES)}
# espelho troca a MÃO (mesmo TIPO): Cross<->Jab, Lead Hook<->Rear Hook, Lead Upp<->Rear Upp
SWAP_L = [("Cross", "Jab"), ("Lead Hook", "Rear Hook"), ("Lead Uppercut", "Rear Uppercut")]
SWAP_ID = np.arange(len(CLASSES))
for a, b in SWAP_L:
    SWAP_ID[LID[a]] = LID[b]; SWAP_ID[LID[b]] = LID[a]

# colapso para o TIPO (3 classes): reto / hook / uppercut
TYPE_OF = {"Cross": 0, "Jab": 0, "Lead Hook": 1, "Rear Hook": 1, "Lead Uppercut": 2, "Rear Uppercut": 2}
TYPE_NAMES = ["reto", "hook", "uppercut"]

TRAIN_VIDS = ["V1", "V2", "V7", "V8"]
CV_VIDS = ["V3", "V4"]       # cross-video (held-out) — generalização OOD
TEST_VIDS = ["V5", "V9"]     # test (held-out, same-source)


def load_concat(vids):
    Xs, ys = [], []
    for v in vids:
        sk, lb = load_video(v)
        Xs.append(sk.reshape(-1, WINDOW_LEN, 17, 2).astype(np.float32))
        ys.append(lb)
    return np.concatenate(Xs), np.concatenate(ys)


def labels_to_ids(raw, type3=False):
    if type3:
        return np.array([TYPE_OF[x] for x in raw])
    return np.array([LID[x] for x in raw])


def make_train_set(Xtr_raw, ytr_id, mean, std, type3, aug_cfg, seed, feat_cfg):
    """Conjunto de treino = original + espelho (troca de mão no 6-classes; rótulo
    preservado no TIPO) [+ cópias com pose-noise se aug_cfg ativo]. Tudo passa pela
    MESMA feature_pipeline."""
    rng = np.random.default_rng(seed)
    swap = (np.arange(3) if type3 else SWAP_ID)   # TIPO: espelho preserva o rótulo

    blocks_X = [feature_pipeline(Xtr_raw, mean, std, **feat_cfg)]
    blocks_y = [ytr_id]
    # espelho
    Xm = mirror_windows(Xtr_raw)
    blocks_X.append(feature_pipeline(Xm, mean, std, **feat_cfg))
    blocks_y.append(swap[ytr_id])
    # pose-noise (opcional): augmenta original e espelho
    if aug_cfg:
        Xa = augment_pose(Xtr_raw, rng, **aug_cfg)
        blocks_X.append(feature_pipeline(Xa, mean, std, **feat_cfg)); blocks_y.append(ytr_id)
        Xma = augment_pose(Xm, rng, **aug_cfg)
        blocks_X.append(feature_pipeline(Xma, mean, std, **feat_cfg)); blocks_y.append(swap[ytr_id])
    return np.concatenate(blocks_X), np.concatenate(blocks_y)


def train_ensemble(Xtr_raw, ytr_id, mean, std, n_classes, *, type3=False, tau=1.0,
                   label_smoothing=0.05, seeds=(42,), aug_cfg=None, feat_cfg=None,
                   epochs=80, verbose=0):
    """Treina um ensemble (1 modelo por seed). Cada modelo: split estratificado interno
    p/ early-stop, features original+espelho(+aug), loss logit-adjusted. Retorna a lista."""
    feat_cfg = feat_cfg or {}
    models = []
    for seed in seeds:
        tf.keras.utils.set_random_seed(int(seed))
        tr_i, val_i = train_test_split(np.arange(len(ytr_id)), test_size=0.15,
                                       stratify=ytr_id, random_state=int(seed))
        Xtr_b, ytr_b = make_train_set(Xtr_raw[tr_i], ytr_id[tr_i], mean, std, type3, aug_cfg, seed, feat_cfg)
        Xval = feature_pipeline(Xtr_raw[val_i], mean, std, **feat_cfg)
        yval = ytr_id[val_i]
        log_pi = log_prior(ytr_b, n_classes)
        mdl = build_model((WINDOW_LEN, Xtr_b.shape[-1]), n_classes)
        compile_model(mdl, log_pi, tau=tau, label_smoothing=label_smoothing)
        cbs = [EarlyStopping(monitor="val_acc", patience=15, restore_best_weights=True, mode="max"),
               ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min")]
        h = mdl.fit(Xtr_b, tf.keras.utils.to_categorical(ytr_b, n_classes),
                    validation_data=(Xval, tf.keras.utils.to_categorical(yval, n_classes)),
                    epochs=epochs, batch_size=32, callbacks=cbs, verbose=verbose)
        # print por seed: mantém a sessão jp run viva (silêncio longo a desconecta) + informa
        va = max(h.history.get("val_acc", [0]))
        print(f"   seed {seed}: val_acc={va:.3f} ({len(h.history['loss'])} épocas)", flush=True)
        models.append(mdl)
    return models


def report(name, models, Xraw, y_ids, mean, std, n_classes, target_names, feat_cfg=None):
    """Imprime acc (com IC bootstrap), macro-F1, ECE e pisos para um split."""
    feat_cfg = feat_cfg or {}
    keep = y_ids >= 0
    X = feature_pipeline(Xraw[keep], mean, std, **feat_cfg)
    probs = predict_proba_ensemble(models, X)
    yp = probs.argmax(1); yt = y_ids[keep]
    acc, lo, hi = eval_utils.bootstrap_ci(yt, yp)
    mf1 = f1_score(yt, yp, average="macro")
    e = eval_utils.ece(probs, yt)
    maj, ale = eval_utils.floors(yt)
    print(f"=== {name}: acc={acc:.3f} [IC95 {lo:.3f},{hi:.3f}] macroF1={mf1:.3f} ECE={e:.3f} "
          f"| pisos maj={maj:.3f} aleat={ale:.3f} | n={len(yt)}")
    return yp, yt, probs


def main():
    Xtr_raw, ytr_raw = load_concat(TRAIN_VIDS)
    Xcv_raw, ycv_raw = load_concat(CV_VIDS)
    Xte_raw, yte_raw = load_concat(TEST_VIDS)
    mean, std = axis_stats(Xtr_raw)
    np.savez("norm_stats.npz", mean=mean, std=std)

    # ----- 6 classes -----
    ytr6 = labels_to_ids(ytr_raw); ycv6 = labels_to_ids(ycv_raw); yte6 = labels_to_ids(yte_raw)
    seeds = (42, 7, 123, 2024, 99)
    print("\n########## 6-CLASSES (features novas + logit-adjust + mirror + ensemble) ##########")
    m6 = train_ensemble(Xtr_raw, ytr6, mean, std, 6, type3=False, tau=1.0, seeds=seeds)
    report("TEST (V5,V9)", m6, Xte_raw, yte6, mean, std, 6, CLASSES)
    yp, yt, _ = report("CROSS-VIDEO (V3,V4)", m6, Xcv_raw, ycv6, mean, std, 6, CLASSES)
    print(classification_report(yt, yp, labels=list(range(6)), target_names=CLASSES, zero_division=0))

    # ----- 3 classes (TIPO) -----
    ytr3 = labels_to_ids(ytr_raw, True); ycv3 = labels_to_ids(ycv_raw, True); yte3 = labels_to_ids(yte_raw, True)
    print("\n########## 3-CLASSES (TIPO: reto/hook/uppercut) ##########")
    m3 = train_ensemble(Xtr_raw, ytr3, mean, std, 3, type3=True, tau=1.0, seeds=seeds)
    report("TEST tipo (V5,V9)", m3, Xte_raw, yte3, mean, std, 3, TYPE_NAMES)
    report("CROSS-VIDEO tipo (V3,V4)", m3, Xcv_raw, ycv3, mean, std, 3, TYPE_NAMES)

    # salva o modelo 6-classes (ensemble: salva todos)
    for k, m in enumerate(m6):
        m.save(f"modelo_boxe_e{k}.keras")
    m6[0].save("modelo_boxe.keras")
    print("\nsalvo: modelo_boxe.keras (+ ensemble e0..e4) + norm_stats.npz")


if __name__ == "__main__":
    main()
