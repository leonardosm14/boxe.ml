"""Extracao de pose (YOLO11x-pose) -> esqueleto COCO-17 por frame.

Correcao central vs a extracao antiga: NAO zerar juntas de baixa confianca. A antiga
zerava o punho quando conf<0.5; no adam (perfil lateral) isso apagava o punho de TRAS
em ~48% dos frames -> geometria de mao virava lixo. Aqui mantemos a posicao ESTIMADA do
YOLO (continua, mesmo ocluida) e guardamos a confianca a parte, pra quem quiser ponderar.
Medido: cobertura do punho de tras 52% -> 99%.
"""
import os
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11x-pose.pt"   # x = mais preciso; recupera melhor o braco de tras (conf 0.49->0.68)


def extract(video_path, model_path=MODEL_PATH, smooth=True):
    """Roda YOLO11x-pose no video e retorna (skeletons, conf).
    skeletons: (T,17,2) em coords normalizadas [0,1] (x/W, y/H), y cresce p/ baixo.
    conf: (T,17) confianca por junta. Mantem a pessoa de maior confianca media por frame;
    se o frame perde a pessoa, segura o frame anterior (sem buraco)."""
    cap = cv2.VideoCapture(video_path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(model_path)
    sk = np.zeros((T, 17, 2), np.float64)
    cf = np.zeros((T, 17), np.float64)
    prev = None
    results = model.track(source=video_path, stream=True, conf=0.25, verbose=False)
    for i, r in enumerate(results):
        if i >= T:
            break
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            data = r.keypoints.data.cpu().numpy()            # (P,17,3)
            best = int(np.argmax(data[:, :, 2].mean(axis=1)))  # pessoa de interesse = maior conf media
            kp = data[best]
            xy = kp[:, :2].copy(); xy[:, 0] /= W; xy[:, 1] /= H
            sk[i], cf[i] = xy, kp[:, 2]
            prev = (xy, kp[:, 2])
        elif prev is not None:
            sk[i], cf[i] = prev
    if smooth:
        sk = smooth_keypoints(sk)
    return sk, cf


def smooth_keypoints(sk, k=5):
    """Suavizacao temporal leve (media movel k-tap) por junta/canal. Tira o jitter do
    detector sem achatar o golpe."""
    out = sk.copy()
    pad = k // 2
    for j in range(17):
        for c in range(2):
            s = sk[:, j, c]
            out[:, j, c] = np.convolve(np.pad(s, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")
    return out


def load_or_extract(video_path, cache_path=None, model_path=MODEL_PATH):
    """Usa o cache .npy se existir, senao extrai e salva. (skeletons, conf)."""
    cache_path = cache_path or f"skeletons_{os.path.splitext(os.path.basename(video_path))[0]}.npy"
    conf_path = cache_path.replace("skeletons_", "conf_")
    if os.path.exists(cache_path):
        cf = np.load(conf_path) if os.path.exists(conf_path) else np.ones((len(np.load(cache_path)), 17))
        return np.load(cache_path), cf
    sk, cf = extract(video_path, model_path)
    np.save(cache_path, sk); np.save(conf_path, cf)
    return sk, cf
