"""
Treino do classificador de golpes (gera modelo_boxe.keras + norm_stats.npz).

Diferenças vs a versão inicial (ver MUDANCAS.md):
  - Normalização RELATIVA AO CORPO (recentro no quadril + escala de ombro) — features
    invariantes a posição/escala/câmera. Foi o que melhorou same-source E cross-video.
  - label smoothing (calibra a confiança, mata o "crava 100%").
  - mirror augmentation no referencial do corpo (espelha x + troca juntas e rótulos Lead<->Rear).
  - ReduceLROnPlateau corrigido; seed fixo; validação estratificada do treino.
  - Avaliação reportando TEST (V5,V9) E CROSS-VIDEO (V3,V4) com pisos majoritário/aleatório.

A pipeline de features (corpo -> padronização por eixo -> vel/acc) é a mesma de boxe.py
(boxe_utils.preprocess_windows), garantindo treino == inferência.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Bidirectional, LSTM, Dense, Dropout, MultiHeadAttention,
                                      GlobalAveragePooling1D, Input, Add, LayerNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
from boxe_utils import (load_video, build_label_mapping, body_frame_windows,
                        add_velocity_and_acceleration, WINDOW_LEN)

tf.keras.utils.set_random_seed(42)
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

TRAIN_VIDS = ["V1", "V2", "V7", "V8"]
CV_VIDS = ["V3", "V4"]       # cross-video (held-out) — só para reportar generalização
TEST_VIDS = ["V5", "V9"]     # test (held-out, same-source)


def load_concat(vids):
    Xs, ys = [], []
    for v in vids:
        sk, lb = load_video(v)
        Xs.append(sk.astype(np.float32)); ys.append(lb)
    return np.concatenate(Xs), np.concatenate(ys)


X_tr_all, y_tr_all = load_concat(TRAIN_VIDS)
X_cv, y_cv_raw = load_concat(CV_VIDS)
X_te, y_te_raw = load_concat(TEST_VIDS)

classes, label_to_id, id_to_label = build_label_mapping(y_tr_all)
num_classes = len(classes)
y_tr_all_id = np.array([label_to_id[x] for x in y_tr_all])
y_cv = np.array([label_to_id.get(x, -1) for x in y_cv_raw])
y_te = np.array([label_to_id.get(x, -1) for x in y_te_raw])

tr_i, val_i = train_test_split(np.arange(len(y_tr_all_id)), test_size=0.15,
                               stratify=y_tr_all_id, random_state=42)
X_tr, y_tr = X_tr_all[tr_i], y_tr_all_id[tr_i]
X_val, y_val = X_tr_all[val_i], y_tr_all_id[val_i]


def axis_stats(X):
    """Média/desvio por eixo (x,y) no referencial do corpo, só nos valores não-nulos."""
    B = body_frame_windows(X); x, y = B[..., 0], B[..., 1]
    return (np.array([x[x != 0].mean(), y[y != 0].mean()], np.float32),
            np.array([x[x != 0].std() + 1e-6, y[y != 0].std() + 1e-6], np.float32))


mean, std = axis_stats(X_tr)
np.savez("norm_stats.npz", mean=mean, std=std)   # carregado por boxe.py (treino == inferência)

SWAP_J = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
SWAP_L = [("Cross", "Jab"), ("Lead Hook", "Rear Hook"), ("Lead Uppercut", "Rear Uppercut")]
swap_id = np.arange(num_classes)
for a, b in SWAP_L:
    swap_id[label_to_id[a]] = label_to_id[b]; swap_id[label_to_id[b]] = label_to_id[a]


def flip_body(B):
    """Espelho horizontal no referencial do corpo: nega x + troca juntas esquerda<->direita."""
    F = B.copy(); F[..., 0] *= -1
    for a, b in SWAP_J:
        t = F[:, :, a, :].copy(); F[:, :, a, :] = F[:, :, b, :]; F[:, :, b, :] = t
    return F


def std_va(B):
    X = B.reshape(len(B), WINDOW_LEN, 34).copy()
    X[:, :, 0::2] = (X[:, :, 0::2] - mean[0]) / std[0]
    X[:, :, 1::2] = (X[:, :, 1::2] - mean[1]) / std[1]
    return add_velocity_and_acceleration(X)


def feats(X):
    return std_va(body_frame_windows(X))


B_tr = body_frame_windows(X_tr)
X_tr_feat = np.concatenate([std_va(B_tr), std_va(flip_body(B_tr))])   # original + espelhado
y_tr_final = np.concatenate([y_tr, swap_id[y_tr]])
X_val_feat, X_cv_feat, X_te_feat = feats(X_val), feats(X_cv), feats(X_te)


def build_model(input_shape, nc):
    i = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(i)
    x = Dropout(0.3)(x)
    a = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
    x = Add()([x, a]); x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(0.0005))(x); x = Dropout(0.3)(x)
    return Model(i, Dense(nc, activation="softmax")(x))


model = build_model((25, 102), num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
weights = compute_class_weight("balanced", classes=np.unique(y_tr_final), y=y_tr_final)
class_weights = dict(enumerate(weights))
y_tr_oh = tf.keras.utils.to_categorical(y_tr_final, num_classes)
y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, mode="min", verbose=0),
    ModelCheckpoint("best_model.keras", monitor="val_accuracy", mode="max", save_best_only=True, verbose=0),
]
model.fit(X_tr_feat, y_tr_oh, validation_data=(X_val_feat, y_val_oh), epochs=100, batch_size=32,
          class_weight=class_weights, callbacks=callbacks, verbose=2)
best = load_model("best_model.keras")


def report(name, Xf, y):
    keep = y >= 0
    yp = np.argmax(best.predict(Xf[keep], batch_size=512, verbose=0), axis=1); yt = y[keep]
    acc = float((yp == yt).mean()); mf1 = float(f1_score(yt, yp, average="macro"))
    maj = Counter(yt).most_common(1)[0][1] / len(yt)
    print(f"=== {name}: acc={acc:.4f} macroF1={mf1:.4f} | piso majoritário={maj:.3f} aleatório={1/num_classes:.3f} | n={len(yt)}")
    return yp, yt


print("\n########## RESULTADOS ##########")
report("TEST (V5,V9, same-source)", X_te_feat, y_te)
report("CROSS-VIDEO (V3,V4, held-out)", X_cv_feat, y_cv)
yp, yt = report("TEST detalhe", X_te_feat, y_te)
print(classification_report(yt, yp, labels=list(range(num_classes)), target_names=classes, zero_division=0))
best.save("modelo_boxe.keras")
print("saved modelo_boxe.keras + norm_stats.npz")
