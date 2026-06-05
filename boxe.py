import os
import argparse
import subprocess
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU error: {e}")

BOXING_CLASSES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]
X_MEAN_GLOBAL = 0.1763427519365391
X_STD_GLOBAL  = 0.24370889809812205


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


def preprocess_window(window_skeletons, mean=X_MEAN_GLOBAL, std=X_STD_GLOBAL):
    X = window_skeletons.reshape(1, 25, 34)
    X = (X - mean) / std
    vel = np.diff(X, axis=1, prepend=np.zeros_like(X[:, :1, :]))
    acc = np.diff(vel, axis=1, prepend=np.zeros_like(vel[:, :1, :]))
    return np.concatenate([X, vel, acc], axis=-1)


def adaptive_ema_smoothing(current_probs, smoothed_history, last_class):
    current_class = np.argmax(current_probs)
    alpha = 0.7 if current_class != last_class else 0.2
    if smoothed_history is None:
        return current_probs, current_class
    smoothed = alpha * current_probs + (1 - alpha) * smoothed_history
    return smoothed, current_class


def is_confident_prediction(probs, threshold_conf=0.60, threshold_entropy=0.50):
    class_idx  = np.argmax(probs)
    confidence = probs[class_idx]
    entropy      = -np.sum(probs * np.log(probs + 1e-9))
    entropy_norm = entropy / np.log(len(probs))
    valid = confidence > threshold_conf and entropy_norm < threshold_entropy
    return class_idx, confidence, valid


def is_valid_skeleton(skeleton_frame, threshold=0.05):
    zero_ratio = np.mean(skeleton_frame == 0)
    return zero_ratio < 0.5 and skeleton_frame.mean() > threshold


def save_video_with_predictions(video_path, output_path, skeletons, model):
    print("--> [3/3] Preparing batch data...")
    total_frames = len(skeletons)

    valid_indices = []
    X_batch = []

    for frame_idx in range(total_frames):
        if frame_idx >= 25:
            window = skeletons[frame_idx - 25 : frame_idx]
            X_batch.append(preprocess_window(window))
            valid_indices.append(frame_idx)
        else:
            available_frames = skeletons[0:frame_idx]
            if len(available_frames) >= 5:
                window = np.zeros((25, 17, 2))
                window[-len(available_frames):] = available_frames
                X_batch.append(preprocess_window(window))
                valid_indices.append(frame_idx)

    X_batch = np.concatenate(X_batch, axis=0)

    print(f"--> Running inference on {len(valid_indices)} valid frames...")
    valid_predictions = model.predict(X_batch, batch_size=512, verbose=1)

    all_predictions = [None] * total_frames
    for i, frame_idx in enumerate(valid_indices):
        all_predictions[frame_idx] = valid_predictions[i]

    temp_output_path = "temp_raw_output.mp4"
    cap = cv2.VideoCapture(video_path)
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (video_width, video_height))

    frame_idx      = 0
    smoothed_probs = None
    last_class     = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pred = all_predictions[frame_idx]

        if pred is None or not is_valid_skeleton(skeletons[frame_idx]):
            label_text  = "..."
            label_color = (255, 255, 255)
        else:
            smoothed_probs, last_class = adaptive_ema_smoothing(pred, smoothed_probs, last_class)
            class_idx, confidence, valid = is_confident_prediction(smoothed_probs)

            if valid:
                label_text  = f"{BOXING_CLASSES[class_idx]} ({confidence*100:.1f}%)"
                label_color = (0, 255, 0)
            else:
                label_text  = "..."
                label_color = (255, 255, 255)

        cv2.rectangle(frame, (0, 0), (video_width, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {label_text}",  (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)

        frame_text = f"Frame: {frame_idx}"
        (text_width, text_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6,1)
        x = video_width - text_width - 20
        y = video_height - 20
        cv2.putText(
            frame,
            frame_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print("--> Converting codec to H.264...")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', temp_output_path,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        print(f"--> Final video saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxing Punch Classifier")
    parser.add_argument("-v", "--video",       required=True,                  help="Input video path")
    parser.add_argument("-m", "--model",       default="modelo_boxe.keras",    help="Model path")
    parser.add_argument("-o", "--output",      default=None,                   help="Custom output path")
    parser.add_argument("--clear-cache",       action="store_true",            help="Force skeleton cache reset")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video '{args.video}' not found.")
        exit(1)

    if args.output is None:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        original_name = os.path.basename(args.video)
        output_path = os.path.join(output_dir, original_name)
    else:
        output_path = args.output

    print("--> Loading TensorFlow model...")
    loaded_model = load_model(args.model)

    video_base_name = os.path.splitext(os.path.basename(args.video))[0]
    cache_path = f"skeletons_{video_base_name}.npy"

    if args.clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print("--> Previous cache cleared.")

    if os.path.exists(cache_path):
        print(f"--> Cache found: {cache_path}")
        skeletons = np.load(cache_path)
    else:
        skeletons = extract_skeletons(args.video)
        if skeletons is not None:
            np.save(cache_path, skeletons)

    if skeletons is not None:
        print("--> Applying skeleton smoothing...")
        window_size = 5
        for joint in range(17):
            for coord in range(2):
                signal = skeletons[:, joint, coord]
                padded = np.pad(
                    signal,
                    (window_size // 2, window_size // 2),
                    mode="edge"
                )
                smoothed = np.convolve(
                    padded,
                    np.ones(window_size) / window_size,
                    mode="valid"
                )
                skeletons[:, joint, coord] = smoothed
    
        save_video_with_predictions(
            args.video,
            output_path,
            skeletons,
            loaded_model
        )