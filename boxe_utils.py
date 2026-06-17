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
    skeleton = np.load(npy_path, allow_pickle=True)   # dados próprios do grupo (BoxingVI)
    normalized = np.array([normalize_label(l) for l in labels["class"].values])
    return skeleton, normalized


def add_velocity_and_acceleration(X):
    vel = np.diff(X, axis=1, prepend=np.zeros_like(X[:, :1, :]))
    acc = np.diff(vel, axis=1, prepend=np.zeros_like(vel[:, :1, :]))
    return np.concatenate([X, vel, acc], axis=-1)


WINDOW_LEN = 25
REAL_LEN = 10  # frames reais por janela (média do dataset); o resto é padding de zeros

# COCO: nariz 0 · olhos 1,2 · orelhas 3,4 · ombros 5,6 · cotovelos 7,8 · punhos 9,10 ·
# quadris 11,12 · joelhos 13,14 · tornozelos 15,16  (E=esquerda, D=direita)
SHO_L, SHO_R, HIP_L, HIP_R = 5, 6, 11, 12
ELB_L, ELB_R, WRI_L, WRI_R = 7, 8, 9, 10


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
    # clipa para range corporal são (mata outliers de coords ruins, ex. V6 cru em pixels)
    return np.clip(out, -6.0, 6.0)


def window_mask(windows):
    """(N,25) booleano: True onde o frame é real (alguma junta != 0), False no padding.
    Fonte única da máscara — o modelo usa a mesma lógica (Masking) e os features zeram o
    padding, então treino e inferência veem o mesmo recorte de frames válidos."""
    return (np.asarray(windows) != 0).any(axis=(2, 3))


def deroll(coords):
    """De-roll no plano: nivela a linha de quadril (rotaciona cada frame por -alpha em torno
    da origem, onde alpha = atan2(quadrilD - quadrilE)). Tira roll/tilt de câmera — é a parte
    BEM-POSTA da canonicalização em 2D (frontalização total seria mal-posta e destruiria o
    arco do hook). coords já no referencial do corpo (N,25,17,2), padding em zero permanece zero."""
    C = np.asarray(coords, dtype=np.float32)
    real = (C != 0).any(axis=(2, 3))                      # (N,25)
    hip_vec = C[:, :, HIP_R, :] - C[:, :, HIP_L, :]       # (N,25,2)
    alpha = np.arctan2(hip_vec[..., 1], hip_vec[..., 0])  # (N,25)
    cos_a, sin_a = np.cos(-alpha), np.sin(-alpha)         # rotação por -alpha
    out = C.copy()
    x, y = C[..., 0], C[..., 1]                           # (N,25,17)
    rx = cos_a[..., None] * x - sin_a[..., None] * y
    ry = sin_a[..., None] * x + cos_a[..., None] * y
    rot = np.stack([rx, ry], axis=-1)                     # (N,25,17,2)
    det = (C != 0).any(axis=3)                            # (N,25,17) junta detectada?
    out = np.where(det[..., None] & real[:, :, None, None], rot, 0.0)
    return out.astype(np.float32)


def _interior_angle(coords, j, a, b):
    """Ângulo interno na junta j entre os vetores (a-j) e (b-j), por frame. (N,25) em [0,pi].
    Zera onde alguma das 3 juntas não foi detectada."""
    va = coords[:, :, a, :] - coords[:, :, j, :]
    vb = coords[:, :, b, :] - coords[:, :, j, :]
    dot = (va * vb).sum(-1)
    cross = va[..., 0] * vb[..., 1] - va[..., 1] * vb[..., 0]
    ang = np.arctan2(np.abs(cross), dot)                  # (N,25) em [0,pi]
    det = ((coords[:, :, j, :] != 0).any(-1) &
           (coords[:, :, a, :] != 0).any(-1) &
           (coords[:, :, b, :] != 0).any(-1))
    return np.where(det, ang, 0.0).astype(np.float32)


def engineered_features(coords):
    """Features view-invariantes (ângulos internos do corpo), calculadas no referencial do
    corpo de-rolled. São o que separa o TIPO (reto/hook/uppercut) e generaliza entre câmeras
    melhor que coordenadas cruas:
      - ângulo de cotovelo E/D (extensão = reto; ~90° = hook; agudo = uppercut)
      - ângulo ombro (cotovelo-ombro-quadril) E/D (abertura do braço)
      - torção do tronco (linha de ombro pós-deroll, = ombro-quadril): sin e cos
    Retorna (N,25,6). Padding/juntas faltando = 0. Ângulos escalados p/ ~[-1,1] (÷pi)."""
    C = np.asarray(coords, dtype=np.float32)
    elb_l = _interior_angle(C, ELB_L, SHO_L, WRI_L) / np.pi    # cotovelo esquerdo
    elb_r = _interior_angle(C, ELB_R, SHO_R, WRI_R) / np.pi    # cotovelo direito
    sho_l = _interior_angle(C, SHO_L, ELB_L, HIP_L) / np.pi    # ombro esquerdo
    sho_r = _interior_angle(C, SHO_R, ELB_R, HIP_R) / np.pi    # ombro direito
    sl = C[:, :, SHO_R, :] - C[:, :, SHO_L, :]                 # linha de ombro (pós-deroll = torção)
    ang = np.arctan2(sl[..., 1], sl[..., 0])                  # (N,25)
    det_sl = ((C[:, :, SHO_L, :] != 0).any(-1) & (C[:, :, SHO_R, :] != 0).any(-1))
    sin_t = np.where(det_sl, np.sin(ang), 0.0).astype(np.float32)
    cos_t = np.where(det_sl, np.cos(ang), 0.0).astype(np.float32)
    return np.stack([elb_l, elb_r, sho_l, sho_r, sin_t, cos_t], axis=-1)   # (N,25,6)


def make_window(skeletons, end_idx, real_len=REAL_LEN):
    """Monta a janela no MESMO layout do treino: os frames reais ficam no INÍCIO e
    o resto é zero. O .npy do BoxingVI usa esse formato (golpe curto ~10 frames +
    padding até 25). Antes a inferência usava 25 frames densos, que o modelo nunca
    viu no treino (out-of-distribution) — principal causa dos golpes falsos.

    `skeletons` tem shape (frames, 17, 2); `end_idx` é o frame atual.
    Retorna (25, 17, 2) ou None se ainda não há frames reais suficientes.
    """
    start = end_idx - real_len + 1
    if start < 0:
        return None
    window = np.zeros((WINDOW_LEN, 17, 2), dtype=np.float32)
    window[:real_len] = skeletons[start:end_idx + 1]
    return window


def feature_pipeline(windows, mean, std, use_deroll=True, use_eng=True):
    """Pipeline COMPARTILHADA treino == inferência. Etapas:
      1. referencial do corpo (recentro quadril + escala ombro)
      2. de-roll (nivela linha de quadril) — opcional (use_deroll)
      3. padronização por EIXO (x,y) das coords só nos frames reais (padding fica 0)
      4. velocidade + aceleração (102 canais)
      5. (use_eng) features de ângulo view-invariantes (6) + razão de direção do punho (2)
      6. ZERA todos os canais nos frames de padding -> o modelo mascara só no pooling
    windows: (N,25,17,2) coords cruas; mean,std: arrays (2,) [x,y]. Retorna (N,25,102 ou 110)."""
    coords = body_frame_windows(windows)                 # (N,25,17,2)
    if use_deroll:
        coords = deroll(coords)
    real = window_mask(windows)                          # (N,25)

    flat = coords.reshape(len(coords), WINDOW_LEN, 34).copy()
    flat[:, :, 0::2] = (flat[:, :, 0::2] - mean[0]) / std[0]   # x padronizado
    flat[:, :, 1::2] = (flat[:, :, 1::2] - mean[1]) / std[1]   # y padronizado
    va = add_velocity_and_acceleration(flat)             # (N,25,102)

    if use_eng:
        eng = engineered_features(coords)                # (N,25,6)
        vel = va[:, :, 34:68]                             # bloco de velocidade

        def vshare(jx, jy):
            vx, vy = vel[:, :, jx], vel[:, :, jy]
            return (np.abs(vy) / (np.abs(vx) + np.abs(vy) + 1e-6)).astype(np.float32)
        vsh = np.stack([vshare(2 * WRI_L, 2 * WRI_L + 1),
                        vshare(2 * WRI_R, 2 * WRI_R + 1)], axis=-1)   # (N,25,2)
        feats = np.concatenate([va, eng, vsh], axis=-1)  # (N,25,110)
    else:
        feats = va                                       # (N,25,102)

    feats[~real] = 0.0                                   # padding zerado em TODOS os canais
    return feats.astype(np.float32)


# alias de compatibilidade: boxe.py importa preprocess_windows
def preprocess_windows(windows, mean, std):
    return feature_pipeline(windows, mean, std)


def axis_stats(windows):
    """Média/desvio por eixo (x,y) no referencial do corpo DE-ROLLED, só nos valores não-nulos.
    Fonte única salva em norm_stats.npz (treino == inferência)."""
    C = deroll(body_frame_windows(windows))
    x, y = C[..., 0], C[..., 1]
    return (np.array([x[x != 0].mean(), y[y != 0].mean()], np.float32),
            np.array([x[x != 0].std() + 1e-6, y[y != 0].std() + 1e-6], np.float32))


# ----------------------------------------------------------------------------- #
# Augmentation de pose (só no treino) — aplicada nas coords CRUS antes da pipeline.
# Robustez estilo PoseC3D (tolerar ruído do YOLO) + simular ângulos de câmera não vistos.
# Mantida MILD: LSTM é sensível a ruído forte. Cada componente é ablacionado em LOVO.
# ----------------------------------------------------------------------------- #
def augment_pose(windows, rng, jitter=0.0, drop_p=0.0, rot_deg=0.0,
                 scale=(1.0, 1.0), twarp=0.0):
    """Augmenta um lote de janelas cruas (N,25,17,2). Tudo só nos frames/juntas reais.
    jitter: desvio do ruído gaussiano por junta, em fração da largura de ombro.
    drop_p: prob. de zerar (ocluir) uma junta. rot_deg: rotação no plano (±) em torno do
    quadril médio. scale: (lo,hi) fator de escala. twarp: prob. de dropar/duplicar 1 frame real."""
    W = np.asarray(windows, dtype=np.float32).copy()
    N = len(W)
    real = (W != 0).any(axis=(2, 3))                     # (N,25)
    for i in range(N):
        m = real[i]
        n = int(m.sum())
        if n == 0:
            continue
        fr = W[i, m]                                     # (n,17,2)
        det = (fr != 0).any(axis=2)                      # (n,17)
        mid_hip = (fr[:, HIP_L, :] + fr[:, HIP_R, :]) / 2.0
        sho_w = np.median(np.linalg.norm(fr[:, SHO_L, :] - fr[:, SHO_R, :], axis=1))
        sho_w = sho_w if sho_w > 1e-3 else 1e-3
        # escala
        sc = rng.uniform(scale[0], scale[1])
        # rotação no plano em torno do quadril médio
        th = np.deg2rad(rng.uniform(-rot_deg, rot_deg)) if rot_deg > 0 else 0.0
        c, s = np.cos(th), np.sin(th)
        rel = fr - mid_hip[:, None, :]
        rx = c * rel[..., 0] - s * rel[..., 1]
        ry = s * rel[..., 0] + c * rel[..., 1]
        rel = np.stack([rx, ry], axis=-1) * sc
        fr2 = rel + mid_hip[:, None, :]
        # jitter gaussiano (em unidades de largura de ombro)
        if jitter > 0:
            fr2 = fr2 + rng.normal(0.0, jitter * sho_w, size=fr2.shape).astype(np.float32)
        # dropout de junta
        if drop_p > 0:
            keep = rng.random(det.shape) >= drop_p
            det = det & keep
        fr2 = np.where(det[..., None], fr2, 0.0).astype(np.float32)
        # warp temporal simples: dropar ou duplicar 1 frame real (mantém início+padding)
        if twarp > 0 and n >= 4 and rng.random() < twarp:
            if rng.random() < 0.5:
                k = rng.integers(0, n)
                fr2 = np.delete(fr2, k, axis=0)
            else:
                k = rng.integers(0, n)
                fr2 = np.insert(fr2, k, fr2[k], axis=0)[:n]
        buf = np.zeros((WINDOW_LEN, 17, 2), dtype=np.float32)
        buf[:len(fr2)] = fr2[:WINDOW_LEN]
        W[i] = buf
    return W


# pares COCO esquerda<->direita (para espelhar) — olhos, orelhas, ombros, cotovelos,
# punhos, quadris, joelhos, tornozelos. Nariz (0) fica.
SWAP_JOINTS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def mirror_windows(windows):
    """Espelha horizontalmente as janelas CRUAS (N,25,17,2): nega x em torno do x do
    quadril médio (espelha o esqueleto no lugar) e troca as juntas esquerda<->direita.
    Mantém juntas/frames não detectados em zero. Aplicado ANTES da feature_pipeline para
    que de-roll/ângulos recalculem corretos. No 6-classes o rótulo também troca a mão
    (Lead<->Rear); no TIPO (3-classes) o rótulo é preservado (hook espelhado = hook)."""
    W = np.asarray(windows, dtype=np.float32).copy()
    det = (W != 0).any(axis=3)                                  # (N,25,17) junta detectada?
    mid_hip_x = (W[:, :, HIP_L, 0] + W[:, :, HIP_R, 0]) / 2.0   # (N,25)
    x = W[..., 0]
    W[..., 0] = np.where(det, 2.0 * mid_hip_x[..., None] - x, 0.0)   # nega x detectado
    out = W.copy()
    for a, b in SWAP_JOINTS:                                    # troca lados E<->D
        out[:, :, a, :] = W[:, :, b, :]
        out[:, :, b, :] = W[:, :, a, :]
    return out


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
