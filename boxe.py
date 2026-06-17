from gpupick import pick_and_set_gpu
pick_and_set_gpu()
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import subprocess
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from boxe_utils import make_window, feature_pipeline, body_frame_windows, WINDOW_LEN
from model import predict_proba_ensemble
import decode
import stance as st

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU error: {e}")

BOXING_CLASSES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]
WRIST = [9, 10]
# TIPO -> (classe lead, classe rear). Usado na correção adaptativa da mão.
TYPE_OF = {0: 0, 1: 0, 2: 1, 4: 1, 3: 2, 5: 2}           # id 6-classes -> tipo
TYPE_PAIR = {0: (1, 0), 1: (2, 4), 2: (3, 5)}            # tipo -> (lead_id, rear_id)

# Detecção POR PICO da velocidade do punho (body-frame). Histerese/Schmitt funde combos e
# perde golpes leves; cada golpe é um pico local. Medido no adam: ~25 picos (find_peaks
# prominence~0.4). Sinal body-frame: mediana ~0.39, picos >1.5. Ver _debug2.py.
PEAK_PROM = 0.3      # proeminência mínima do pico (golpe) — pega golpes mais leves também
PEAK_DIST = 7        # distância mínima entre golpes (frames)
PEAK_FLOOR = 0.25    # span do golpe = região contígua acima desse piso ao redor do pico
CONF_FLOOR = 0.25    # rejeita golpe ambíguo (conf no pico < 0.25)
HAND_MARGIN = 0.15   # se |P(lead)-P(rear)| do mesmo tipo < margem -> usa geometria (adaptativo)


def load_norm_stats():
    if os.path.exists("norm_stats.npz"):
        s = np.load("norm_stats.npz")
        return np.asarray(s["mean"], np.float32), np.asarray(s["std"], np.float32)
    print("--> Aviso: norm_stats.npz não encontrado; usando fallback")
    return np.array([0.0, 0.0], np.float32), np.array([1.0, 1.0], np.float32)


def load_models(model_arg):
    """Carrega o ensemble (modelo_boxe_e*.keras) se existir; senão o modelo único."""
    ens = sorted(glob.glob("modelo_boxe_e*.keras"))
    if ens:
        print(f"--> Ensemble: {len(ens)} modelos ({', '.join(ens)})")
        return [load_model(m) for m in ens]
    print(f"--> Modelo único: {model_arg}")
    return [load_model(model_arg)]


def ensure_25fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if abs(fps - 25.0) < 0.1:
        print(f"--> Video already at {fps:.2f} FPS")
        return video_path
    output_dir = "25fps"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    if os.path.exists(output_path):
        return output_path
    print(f"--> Converting {fps:.2f} FPS -> 25 FPS...")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vf", "fps=25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", output_path], check=True)
    return output_path


def extract_skeletons(video_path):
    print(f"--> [1/3] Running YOLO-Pose: {video_path}")
    model_yolo = YOLO("yolov8m-pose.pt")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    skeletons = np.zeros((total_frames, 17, 2))
    results = model_yolo.track(source=video_path, stream=True, device="cuda", conf=0.3)
    for i, r in enumerate(results):
        if i >= total_frames:
            break
        if r.boxes is not None and r.keypoints is not None and len(r.keypoints.data) > 0:
            kp = r.keypoints.data[0].cpu().numpy()
            if kp.shape[0] == 17 and kp[:, 2].mean() >= 0.5:
                xy = kp[:, :2].copy()
                xy[:, 0] /= vw; xy[:, 1] /= vh
                skeletons[i] = xy
                continue
        if i > 0:
            skeletons[i] = skeletons[i - 1]
    print("--> [2/3] Skeleton extraction complete.")
    return skeletons


def wrist_speed_bodyframe(skeletons):
    """Velocidade do punho dominante NORMALIZADA pelo corpo (não em pixels). Recentra cada
    frame no quadril médio e escala pela largura de ombro mediana do clipe -> comparável
    entre câmeras/ângulos diferentes. Suavizada (5-tap)."""
    sk = skeletons.astype(np.float32)
    det = (sk != 0).any(axis=2)                          # (T,17)
    mid_hip = (sk[:, 11, :] + sk[:, 12, :]) / 2.0
    sw = np.linalg.norm(sk[:, 5, :] - sk[:, 6, :], axis=1)
    scale = np.median(sw[sw > 1e-3]) if (sw > 1e-3).any() else 1.0
    norm = (sk - mid_hip[:, None, :]) / max(scale, 1e-3)
    norm = np.where(det[..., None], norm, 0.0)
    d = np.diff(norm[:, WRIST, :], axis=0)
    spd = np.linalg.norm(d, axis=2).max(axis=1)
    spd = np.concatenate([[0.0], spd])
    k = 5
    spd = np.convolve(np.pad(spd, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return spd


def per_frame_probs(skeletons, models, mean, std):
    """Probabilidade 6-classes por frame. Para cada frame t monta a janela (real no início)
    e classifica; frames sem janela ainda viram quase-uniforme (baixa confiança)."""
    T = len(skeletons)
    windows, idx = [], []
    for t in range(T):
        w = make_window(skeletons, t)
        if w is not None:
            windows.append(w); idx.append(t)
    probs = np.full((T, 6), 1.0 / 6, dtype=np.float32)
    if windows:
        X = feature_pipeline(np.array(windows), mean, std)
        p = predict_proba_ensemble(models, X)
        for j, t in enumerate(idx):
            probs[t] = p[j]
    return probs


def label_punches(skeletons, probs, speed):
    """Rotula POR PICO: cada pico da velocidade do punho = 1 golpe. Classifica no pico
    (média do softmax de algumas janelas ancoradas), corrige a MÃO adaptativamente por
    geometria quando o modelo titubeia, e pinta o span inteiro (onset->offset) com 1
    rótulo contíguo. Repouso entre golpes = -1 (fundo)."""
    spans = decode.detect_punches(speed, PEAK_PROM, PEAK_DIST, PEAK_FLOOR)
    labels = np.full(len(skeletons), -1, dtype=np.int64)
    for on, pk, off in spans:
        avg = probs[max(0, pk - 2):pk + 3].mean(axis=0)     # softmax médio no pico
        ci, conf = int(np.argmax(avg)), float(avg.max())
        if conf < CONF_FLOOR:                               # golpe ambíguo -> ignora
            continue
        typ = TYPE_OF[ci]; lead_id, rear_id = TYPE_PAIR[typ]
        if abs(avg[lead_id] - avg[rear_id]) < HAND_MARGIN:  # modelo incerto na mão
            window = make_window(skeletons, pk)
            if window is not None:
                side = st.lead_rear(window)
                ci = lead_id if side == "lead" else rear_id
        labels[on:off + 1] = ci
    return labels, spans


def _spans(labels):
    """Spans contíguos com rótulo >= 0 (golpes), ignorando o fundo (-1)."""
    spans, on = [], None
    for i, v in enumerate(labels):
        if v >= 0 and on is None:
            on = i
        elif v < 0 and on is not None:
            spans.append((on, i - 1)); on = None
    if on is not None:
        spans.append((on, len(labels) - 1))
    return spans


def render(video_path, output_path, skeletons, labels):
    cap = cv2.VideoCapture(video_path)
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("temp_raw_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw, vh))
    i, punch_frames, transitions, prev_on, switches = 0, 0, 0, False, 0
    prev_label = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lab = int(labels[i]) if i < len(labels) else -1
        on = lab >= 0
        if on:
            txt, color = f"{BOXING_CLASSES[lab]}", (0, 255, 0)
            punch_frames += 1
            if prev_on and lab != prev_label:       # troca de classe DENTRO de um golpe
                switches += 1
        else:
            txt, color = "...", (255, 255, 255)
        if on != prev_on:
            transitions += 1
        prev_on, prev_label = on, lab
        cv2.rectangle(frame, (0, 0), (vw, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {txt}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        ft = f"Frame: {i}"
        (tw, th), _ = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(frame, ft, (vw - tw - 20, vh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        out.write(frame)
        i += 1
    cap.release(); out.release()
    pct = 100.0 * punch_frames / max(i, 1)
    print(f"--> Frames com golpe: {punch_frames}/{i} ({pct:.1f}%) | transições={transitions} "
          f"| TROCAS de classe dentro de golpe={switches} (ideal 0)")
    try:
        subprocess.run(['ffmpeg', '-y', '-i', "temp_raw_output.mp4", '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p', output_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        os.remove("temp_raw_output.mp4")
        print(f"--> Final video saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Boxing Punch Classifier")
    parser.add_argument("-v", "--video", required=True)
    parser.add_argument("-m", "--model", default="modelo_boxe.keras")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.video):
        print(f"Error: video '{args.video}' not found."); exit(1)

    video_path = ensure_25fps(args.video)
    output_path = args.output or os.path.join("outputs", os.path.basename(args.video))
    os.makedirs("outputs", exist_ok=True)

    print("--> Loading models...")
    models = load_models(args.model)
    mean, std = load_norm_stats()

    base = os.path.splitext(os.path.basename(video_path))[0]
    cache = f"skeletons_{base}.npy"
    if args.clear_cache and os.path.exists(cache):
        os.remove(cache)
    skeletons = np.load(cache) if os.path.exists(cache) else extract_skeletons(video_path)
    if not os.path.exists(cache):
        np.save(cache, skeletons)

    # suavização leve dos keypoints
    for j in range(17):
        for c in range(2):
            sig = skeletons[:, j, c]
            pad = np.pad(sig, (2, 2), mode="edge")
            skeletons[:, j, c] = np.convolve(pad, np.ones(5) / 5, mode="valid")

    print("--> [3/3] Classificando por PICO da velocidade do punho...")
    probs = per_frame_probs(skeletons, models, mean, std)
    speed = wrist_speed_bodyframe(skeletons)
    labels, spans = label_punches(skeletons, probs, speed)
    n_ev = len(_spans(labels))
    print(f"--> {len(spans)} picos detectados, {n_ev} golpes rotulados (1 rótulo contíguo cada)")
    np.save(f"labels_{base}.npy", labels)   # p/ auditoria frame-a-frame (_verify_adam.py)
    print("--> SPANS (golpe: frames [on-off] len classe):")
    for k, (on, off) in enumerate(_spans(labels)):
        ci = int(labels[on])
        print(f"    #{k+1}: [{on}-{off}] len={off-on+1} {BOXING_CLASSES[ci]}")
    render(video_path, output_path, skeletons, labels)


if __name__ == "__main__":
    main()
