from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATION_DIR = Path("dataset/clean_annotation_data")
SKELETON_DIR = Path("dataset/skeleton_data")
DEFAULT_VIDEOS = ["V1", "V2", "V3", "V4", "V5", "V7", "V8", "V9", "V10"]


def normalize_label(label: str) -> str:
    return label.strip().title()


def load_video(video: str):
    csv_path = ANNOTATION_DIR / f"{video}.csv"
    npy_path = SKELETON_DIR / f"{video}.npy"
    labels = pd.read_csv(csv_path)
    labels = labels.dropna(axis=1, how="all")
    skeleton = np.load(npy_path, allow_pickle=True)
    normalized = np.array([normalize_label(l) for l in labels["class"].values])
    return skeleton, normalized


def add_velocity_and_acceleration(X):
    vel = np.diff(X, axis=1, prepend=np.zeros_like(X[:, :1, :]))
    acc = np.diff(vel, axis=1, prepend=np.zeros_like(vel[:, :1, :]))
    return np.concatenate([X, vel, acc], axis=-1)


WINDOW_LEN = 25
REAL_LEN = 10  # frames reais por janela (média do dataset); o resto é padding de zeros

# COCO: ombros 5,6 · cotovelos 7,8 · punhos 9,10 · quadris 11,12
SHO_L, SHO_R, HIP_L, HIP_R = 5, 6, 11, 12


def body_frame_windows(windows):
    """Normaliza para o referencial do CORPO (invariante a posição/escala/câmera):
    recentra cada frame no quadril médio (11,12) e divide pela escala do corpo
    (largura de ombro mediana na janela). Frames de padding (tudo zero) continuam zero.
    É a correção central do colapso cross-video — antes as coords absolutas codificavam
    lado da câmera e zoom. Usada IGUAL no treino e na inferência. windows: (N,25,17,2)."""
    W = np.asarray(windows, dtype=np.float32)
    real = (W != 0).any(axis=(2, 3))                              # (N,25) frame real?
    mid_hip = (W[:, :, HIP_L, :] + W[:, :, HIP_R, :]) / 2.0       # (N,25,2)
    sho_w = np.linalg.norm(W[:, :, SHO_L, :] - W[:, :, SHO_R, :], axis=2)  # (N,25)
    out = np.zeros_like(W)
    for i in range(len(W)):
        m = real[i]
        if not m.any():
            continue
        s = np.median(sho_w[i][m])
        s = s if s > 1e-3 else 1e-3
        fr = W[i, m]                                   # (n,17,2) frames reais
        jm = (fr != 0).any(axis=2, keepdims=True)      # (n,17,1) junta detectada?
        out[i, m] = np.where(jm, (fr - mid_hip[i, m][:, None, :]) / s, 0.0)
    # clipa para range corporal sã (mata outliers de coords ruins, ex. V6 cru em pixels)
    return np.clip(out, -6.0, 6.0)


def make_window(skeletons, end_idx, real_len=REAL_LEN):
    """Monta a janela no MESMO layout do treino: os frames reais ficam no INÍCIO e
    o resto é zero. O .npy do BoxingVI usa esse formato (golpe curto ~10 frames +
    padding até 25). Antes a inferência usava 25 frames densos, que o modelo nunca
    viu no treino (out-of-distribution) — principal causa dos golpes falsos.

    `skeletons` tem shape (frames, 17, 2); `end_idx` é o frame atual.
    Retorna (25, 34) ou None se ainda não há frames reais suficientes.
    """
    start = end_idx - real_len + 1
    # Guarda de borda: None se não há frames reais suficientes ANTES (start<0) OU
    # se a janela passa do ÚLTIMO frame (end_idx fora do range). Sem o segundo
    # cheque, um golpe cujo pico cai perto do fim do vídeo faz a fatia
    # skeletons[start:end_idx+1] vir mais curta que real_len -> broadcast error
    # (ex.: (9,17,2) em (10,17,2)). classify_events ignora janelas None.
    if start < 0 or end_idx >= len(skeletons):
        return None
    window = np.zeros((WINDOW_LEN, 17, 2), dtype=np.float32)
    window[:real_len] = skeletons[start:end_idx + 1]
    return window


def preprocess_windows(windows, mean, std):
    """Pipeline COMPARTILHADA treino == inferência: referencial do corpo -> padronização
    por EIXO (x,y separados) -> velocidade/aceleração. windows: (N,25,17,2) coords cruas;
    mean,std: arrays (2,) [x,y] ajustados no treino. Retorna (N,25,102)."""
    B = body_frame_windows(windows).reshape(len(windows), WINDOW_LEN, 34)
    B[:, :, 0::2] = (B[:, :, 0::2] - mean[0]) / std[0]   # colunas pares = x
    B[:, :, 1::2] = (B[:, :, 1::2] - mean[1]) / std[1]   # colunas ímpares = y
    return add_velocity_and_acceleration(B)


def load_all_videos(videos=DEFAULT_VIDEOS):
    X_all, y_all_raw = [], []
    for video in videos:
        skeleton, labels_raw = load_video(video)
        X_all.append(skeleton)
        y_all_raw.append(labels_raw)
    return np.concatenate(X_all), np.concatenate(y_all_raw)


def build_label_mapping(y_all_raw: np.ndarray):
    classes = sorted(np.unique(y_all_raw))
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return classes, label_to_id, id_to_label


def build_label_mapping_from_videos(videos=DEFAULT_VIDEOS):
    all_labels = []
    for video in videos:
        _, labels = load_video(video)
        all_labels.extend(labels)
    return build_label_mapping(np.array(all_labels))