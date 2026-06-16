import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# GPU configurável via ambiente (antes era fixo em "3", causava conflito no lab).
# Definir antes de importar tensorflow/ultralytics para a visibilidade valer.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import subprocess
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from boxe_utils import make_window, preprocess_windows

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU error: {e}")

BOXING_CLASSES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]

# Fallback caso norm_stats.npz não exista (mean/std por eixo, no referencial do corpo).
X_MEAN_FALLBACK = np.array([0.0, 0.0], dtype=np.float32)
X_STD_FALLBACK  = np.array([1.0, 1.0], dtype=np.float32)

WRIST = [9, 10]      # COCO: punho esquerdo e direito

# Inferência por SEGMENTO DE MOVIMENTO (não frame a frame, nem por pico). O dataset é
# segmentado por golpe (start/end por linha), então a inferência também é: um golpe = uma
# região contígua onde a velocidade do punho passa de SEG_THI e só termina quando cai
# abaixo de SEG_TLO (Schmitt trigger / histerese). A histerese mantém o golpe inteiro
# (windup -> impacto -> retração) como UM segmento -> 1 classificação -> 1 rótulo segurado
# do início ao fim do golpe. Mata por construção: troca de classe no meio do golpe,
# rótulo "sumindo", múltiplos rótulos por golpe, e o golpe-em-repouso.
SEG_THI    = 0.070   # entra em "movimento" (limiar alto)
SEG_TLO    = 0.040   # sai de "movimento" (limiar baixo) — a histerese segura o golpe contíguo
SEG_GAP    = 5       # funde segmentos separados por <= GAP frames (dip de detecção no mesmo golpe)
SEG_MINLEN = 5       # descarta segmentos < MINLEN frames (jitter, não é golpe)
CONF_FLOOR = 0.30    # confiança média mínima do evento p/ exibir
SPAN_POST  = 10      # frames após o impacto que o rótulo CONTINUA (cobre extensão + retração);
                     # a velocidade do punho cai a ~0 na extensão, mas o golpe ainda é visível


def load_norm_stats():
    """Carrega média/desvio salvos no treino (norm_stats.npz). Mantém treino e
    inferência sincronizados — nunca hardcode."""
    if os.path.exists("norm_stats.npz"):
        s = np.load("norm_stats.npz")
        mean, std = np.asarray(s["mean"], np.float32), np.asarray(s["std"], np.float32)
        print(f"--> Norm stats (corpo, por eixo): mean={mean} std={std} (norm_stats.npz)")
        return mean, std
    print("--> Aviso: norm_stats.npz não encontrado; usando fallback")
    return X_MEAN_FALLBACK, X_STD_FALLBACK


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
        print(f"--> 25 FPS version already exists: {output_path}")
        return output_path

    print(f"--> Converting {fps:.2f} FPS video to 25 FPS...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "fps=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"--> Saved 25 FPS video: {output_path}")
    return output_path


def extract_skeletons(video_path):
    print(f"--> [1/3] Running YOLO-Pose: {video_path}")
    model_yolo = YOLO("yolov8m-pose.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    skeletons_matrix = np.zeros((total_frames, 17, 2))
    results = model_yolo.track(source=video_path, stream=True, device="cuda", conf=0.3)

    for frame_idx, r in enumerate(results):
        if frame_idx >= total_frames:
            break
        if r.boxes is not None and r.keypoints is not None and len(r.keypoints.data) > 0:
            kp = r.keypoints.data[0].cpu().numpy()
            if kp.shape[0] == 17:
                confidences = kp[:, 2]
                if confidences.mean() >= 0.5:
                    coords_xy = kp[:, :2]
                    coords_xy[:, 0] /= video_width
                    coords_xy[:, 1] /= video_height
                    skeletons_matrix[frame_idx] = coords_xy
                    continue
        if frame_idx > 0:
            skeletons_matrix[frame_idx] = skeletons_matrix[frame_idx - 1]

    print("--> [2/3] Skeleton extraction complete.")
    return skeletons_matrix


def preprocess_window(window, mean, std):
    """Normaliza UMA janela (25,17,2) pela pipeline compartilhada do boxe_utils
    (referencial do corpo -> padronização por eixo -> vel/acc). Treino == inferência."""
    return preprocess_windows(window[None], mean, std)


def wrist_speed_signal(skeletons):
    """Velocidade do punho dominante por frame (máx dos dois punhos), suavizada (5-tap)."""
    d = np.diff(skeletons[:, WRIST, :], axis=0)
    spd = np.linalg.norm(d, axis=2).max(axis=1)
    spd = np.concatenate([[0.0], spd])
    k = 5
    spd = np.convolve(np.pad(spd, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return spd


def detect_events(skeletons):
    """Segmenta por MOVIMENTO contínuo do punho (Schmitt trigger): entra em 'movimento'
    quando a velocidade passa de SEG_THI e só sai quando cai abaixo de SEG_TLO. A histerese
    captura o golpe inteiro (windup -> impacto -> retração) como UM segmento contíguo, então
    1 rótulo cobre o golpe todo sem cortes. Funde dips curtos (mesmo golpe) e descarta
    segmentos minúsculos (jitter). Retorna [(onset, peak, offset), ...] e o sinal."""
    spd = wrist_speed_signal(skeletons)

    active, segs, start = False, [], 0
    for i, v in enumerate(spd):
        if not active and v >= SEG_THI:
            active, start = True, i
        elif active and v < SEG_TLO:
            active = False
            segs.append([start, i - 1])
    if active:
        segs.append([start, len(spd) - 1])

    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= SEG_GAP:   # mesmo golpe (dip curto)
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))

    events = []
    for on, off in merged:
        if off - on + 1 < SEG_MINLEN:                    # jitter, não é golpe
            continue
        peak = on + int(np.argmax(spd[on:off + 1]))      # frame de impacto (máx velocidade)
        events.append((on, peak, off))
    return events, spd


def save_video_with_predictions(video_path, output_path, skeletons, model, mean, std):
    print("--> [3/3] Segmenting punch motions...")
    events, spd = detect_events(skeletons)
    print(f"--> {len(events)} segmentos de movimento detectados (Schmitt trigger)")

    # Classifica cada evento UMA vez: média do softmax de 5 janelas ancoradas no pico
    # (o pico é o instante de impacto/extensão, a fase que o modelo viu no treino).
    X_batch, owner = [], []
    for ei, (on, pk, off) in enumerate(events):
        for p in range(pk - 2, pk + 3):
            window = make_window(skeletons, p)
            if window is None:
                continue
            X_batch.append(preprocess_window(window, mean, std))
            owner.append(ei)

    frame_label = [None] * len(skeletons)
    n_shown = 0
    n = len(skeletons)
    if X_batch:
        preds = model.predict(np.concatenate(X_batch), batch_size=512, verbose=1)
        owner = np.array(owner)
        for ei, (on, pk, off) in enumerate(events):
            sel = preds[owner == ei]
            if len(sel) == 0:
                continue
            avg = sel.mean(axis=0)                       # média das probs = 1 decisão por golpe
            ci, conf = int(np.argmax(avg)), float(avg.max())
            if conf < CONF_FLOOR:                        # rejeita evento ambíguo
                continue
            n_shown += 1
            # rótulo do snap (onset) até o fim do follow-through (impacto + retração),
            # sem invadir o próximo golpe -> o rótulo fica até o golpe acabar
            end = max(off, pk + SPAN_POST)
            if ei + 1 < len(events):
                end = min(end, events[ei + 1][0] - 1)
            end = min(end, n - 1)
            for f in range(on, end + 1):
                frame_label[f] = (ci, conf)               # 1 rótulo fixo no span do golpe
    print(f"--> {n_shown} eventos exibidos após rejeição")

    temp_output_path = "temp_raw_output.mp4"
    cap = cv2.VideoCapture(video_path)
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (video_width, video_height))

    frame_idx    = 0
    punch_frames = 0
    transitions  = 0
    prev_on      = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # rótulo vem do EVENTO ao qual o frame pertence (1 só por golpe); fora = "..."
        lab = frame_label[frame_idx] if frame_idx < len(frame_label) else None
        on = lab is not None
        if on:
            ci, conf = lab
            label_text, label_color = f"{BOXING_CLASSES[ci]} ({conf*100:.1f}%)", (0, 255, 0)
            punch_frames += 1
        else:
            label_text, label_color = "...", (255, 255, 255)

        if on != prev_on:
            transitions += 1
        prev_on = on

        cv2.rectangle(frame, (0, 0), (video_width, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {label_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)

        frame_text = f"Frame: {frame_idx}"
        (text_width, text_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x = video_width - text_width - 20
        y = video_height - 20
        cv2.putText(frame, frame_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    pct = 100.0 * punch_frames / max(frame_idx, 1)
    print(f"--> Frames com golpe exibido: {punch_frames}/{frame_idx} ({pct:.1f}%)")
    print(f"--> Eventos exibidos: {n_shown} | transições liga/desliga: {transitions} (ideal ~2x eventos)")

    print("--> Converting codec to H.264...")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', temp_output_path,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        print(f"--> Final video saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxing Punch Classifier")
    parser.add_argument("-v", "--video",  required=True,               help="Input video path")
    parser.add_argument("-m", "--model",  default="modelo_boxe.keras", help="Model path")
    parser.add_argument("-o", "--output", default=None,                help="Custom output path")
    parser.add_argument("--clear-cache",  action="store_true",         help="Force skeleton cache reset")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video '{args.video}' not found.")
        exit(1)

    video_path = ensure_25fps(args.video)

    if args.output is None:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(args.video))
    else:
        output_path = args.output

    print("--> Loading TensorFlow model...")
    loaded_model = load_model(args.model)
    mean, std = load_norm_stats()

    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = f"skeletons_{video_base_name}.npy"

    if args.clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print("--> Previous cache cleared.")

    if os.path.exists(cache_path):
        print(f"--> Cache found: {cache_path}")
        skeletons = np.load(cache_path)
    else:
        skeletons = extract_skeletons(video_path)
        if skeletons is not None:
            np.save(cache_path, skeletons)

    if skeletons is not None:
        print("--> Applying skeleton smoothing...")
        window_size = 5
        for joint in range(17):
            for coord in range(2):
                signal = skeletons[:, joint, coord]
                padded = np.pad(signal, (window_size // 2, window_size // 2), mode="edge")
                smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode="valid")
                skeletons[:, joint, coord] = smoothed

        save_video_with_predictions(video_path, output_path, skeletons, loaded_model, mean, std)
