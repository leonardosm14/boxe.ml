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

def ensure_25fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if abs(fps - 25.0) < 0.1:
        print(f"--> Video already at {fps:.2f} FPS")
        return video_path

    output_dir = "25fps"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        os.path.basename(video_path)
    )

    if os.path.exists(output_path):
        print(f"--> 25 FPS version already exists: {output_path}")
        return output_path

    print(f"--> Converting {fps:.2f} FPS video to 25 FPS...")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        "fps=25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path
    ]

    subprocess.run(cmd, check=True)

    print(f"--> Saved 25 FPS video: {output_path}")

    return output_path

# COCO-17 keypoint indices for the two wrists. These are the joints that move
# fastest when a punch is thrown, so they drive the "is this person boxing?"
# score in select_boxers(). Used to be irrelevant when we only ever kept one
# person; now they are how we tell boxers apart from the referee.
LEFT_WRIST  = 9
RIGHT_WRIST = 10


def extract_skeletons(video_path):
    # CHANGE 1 - MULTI-PERSON EXTRACTION
    # ----------------------------------
    # The OLD version called `r.keypoints.data[0]` and threw everything else
    # away, so the whole pipeline only ever saw a single skeleton per frame.
    # That is fine for a clip with one boxer, but in a real bout the frame
    # holds two boxers AND a referee, and YOLO's ordering of detections is not
    # stable - "[0]" could be any of them, frame to frame.
    #
    # The NEW version keeps EVERY detected person and groups their skeletons by
    # the tracker's ID (`r.boxes.id`), which YOLO's `.track()` already computes
    # for us (the old code requested tracking but ignored the IDs). The result
    # is a dictionary: one entry per tracked person, each holding that person's
    # skeleton over the whole video. Deciding which two of those people are the
    # boxers happens later, in select_boxers().
    print(f"--> [1/3] Running YOLO-Pose: {video_path}")
    model_yolo = YOLO("yolov8m-pose.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # tracks[track_id] = {
    #   "coords":    (total_frames, 17, 2) normalized keypoints, 0 where absent,
    #   "present":   (total_frames,)       bool, True on frames this id was seen,
    #   "centroids": (total_frames, 2)     normalized bbox center, for label placement
    # }
    # We lazily create an entry the first time we see a given track id.
    tracks = {}

    def _new_track():
        return {
            "coords":    np.zeros((total_frames, 17, 2)),
            "present":   np.zeros(total_frames, dtype=bool),
            "centroids": np.zeros((total_frames, 2)),
        }

    results = model_yolo.track(source=video_path, stream=True, device="cuda", conf=0.3)

    for frame_idx, r in enumerate(results):
        if frame_idx >= total_frames:
            break
        # Need both keypoints and boxes-with-ids to attribute a skeleton to a
        # person. If the tracker produced no IDs this frame, skip it (those
        # frames become gaps, filled later by build_dense_skeletons()).
        if r.boxes is None or r.keypoints is None or r.boxes.id is None:
            continue

        kp_all  = r.keypoints.data.cpu().numpy()   # (N persons, 17, 3): x, y, conf
        ids     = r.boxes.id.cpu().numpy().astype(int)  # (N,) track id per person
        xyxy    = r.boxes.xyxy.cpu().numpy()       # (N, 4) bbox corners, pixels

        # Loop over ALL people in the frame - this is the core of the rewrite.
        for kp, tid, box in zip(kp_all, ids, xyxy):
            if kp.shape[0] != 17:
                continue
            # Same per-person quality gate as the old code (mean keypoint
            # confidence >= 0.5) - just applied to each person instead of only
            # the first. Weak/occluded detections are dropped and become gaps.
            if kp[:, 2].mean() < 0.5:
                continue

            # Normalize x,y by frame size exactly like before, so the skeletons
            # remain in the same 0..1 space the classifier was trained on.
            coords_xy = kp[:, :2].copy()
            coords_xy[:, 0] /= video_width
            coords_xy[:, 1] /= video_height

            if tid not in tracks:
                tracks[tid] = _new_track()
            tracks[tid]["coords"][frame_idx]  = coords_xy
            tracks[tid]["present"][frame_idx] = True
            # bbox center, normalized - used only to position the on-screen
            # label near each boxer during rendering.
            cx = (box[0] + box[2]) / 2.0 / video_width
            cy = (box[1] + box[3]) / 2.0 / video_height
            tracks[tid]["centroids"][frame_idx] = (cx, cy)

    print(f"--> [2/3] Skeleton extraction complete. {len(tracks)} tracks found.")
    return tracks, total_frames


def select_boxers(tracks, total_frames):
    # CHANGE 2 - PICK THE TWO BOXERS, DROP THE REFEREE (and crowd)
    # ------------------------------------------------------------
    # extract_skeletons() hands us EVERY tracked person. Most frames also
    # contain a referee, and sometimes a stray person at the ropes gets a
    # track too. We need to keep the two actual boxers.
    #
    # Idea: boxers throw punches, so their wrists move a lot; the referee
    # mostly stands and walks, so his wrists move comparatively little. We
    # score each track with two simple signals and keep the top two:
    #
    #   wrist_motion = average frame-to-frame movement of the two wrists
    #                  (only over frames where the person was actually seen)
    #   presence     = fraction of the whole video the person appears in
    #                  (kills brief crowd/background detections)
    #
    #   score = wrist_motion * presence
    #
    # That's deliberately basic - no appearance model, no re-identification.
    # The trade-off (documented for honesty): we use YOLO's raw track IDs, so
    # if a boxer is fully occluded in a clinch and comes back with a NEW id,
    # that becomes a separate, lower-scoring track. Selection is global over
    # the clip, so a fragment may lose its slot until the original id resumes.
    # Short dropouts are hidden by the gap-fill in build_dense_skeletons().
    scored = []
    for tid, t in tracks.items():
        present_idx = np.where(t["present"])[0]
        if len(present_idx) < 2:
            continue  # need at least 2 sightings to measure any movement

        presence = len(present_idx) / total_frames

        # Wrist movement: take the wrist coordinates only on the frames where
        # this person was seen, then sum the step-to-step distance and average
        # it. Using present-only frames avoids counting the (empty) gaps.
        wrists = t["coords"][present_idx][:, [LEFT_WRIST, RIGHT_WRIST], :]  # (P, 2, 2)
        steps  = np.diff(wrists, axis=0)                                    # (P-1, 2, 2)
        wrist_motion = np.linalg.norm(steps, axis=2).sum() / len(present_idx)

        score = wrist_motion * presence
        scored.append((tid, score, wrist_motion, presence))

    # Highest score first, then keep up to two. (A single-boxer clip yields a
    # list of length 1, which the rest of the pipeline handles fine.)
    scored.sort(key=lambda s: s[1], reverse=True)

    print("--> Boxer selection (wrist_motion * presence):")
    for tid, score, wm, pres in scored:
        tag = "BOXER" if (tid, score, wm, pres) in scored[:2] else "drop "
        print(f"      [{tag}] id={tid}  score={score:.4f}  wrist={wm:.4f}  presence={pres:.2f}")

    return [tid for tid, _, _, _ in scored[:2]]


def build_dense_skeletons(track, total_frames):
    # CHANGE 3 - DENSE PER-BOXER MATRIX WITH GAP-FILL
    # -----------------------------------------------
    # The classifier needs ONE skeleton per frame with no holes, because
    # save_video_with_predictions() builds its 25-frame windows by slicing
    # `skeletons[idx-25:idx]`. A selected boxer is NOT present on every frame
    # (occlusion, blur, frame edge, or failing the confidence gate), so his
    # data is sparse: real skeletons on some frames, zeros on others.
    #
    # This turns that sparse track into the dense (total_frames, 17, 2) array
    # the old single-boxer code used to produce. The gap-fill rule is exactly
    # the one the OLD extract_skeletons used at the single-person level:
    # "if this frame is empty, copy the previous frame forward." That keeps the
    # array continuous so velocity/acceleration in preprocess_window() don't
    # spike on a zero row.
    coords  = track["coords"]
    present = track["present"]

    dense = np.zeros((total_frames, 17, 2))
    for idx in range(total_frames):
        if present[idx]:
            dense[idx] = coords[idx]
        elif idx > 0:
            dense[idx] = dense[idx - 1]   # carry the last known pose forward
    return dense


def smooth_skeletons(skeletons, window_size=5):
    # Moving-average temporal smoothing - lifted verbatim from the old main()
    # block so it can run once PER BOXER instead of once on the single global
    # skeleton. Behavior is unchanged; only the call site moved.
    smoothed = skeletons.copy()
    for joint in range(17):
        for coord in range(2):
            signal = smoothed[:, joint, coord]
            padded = np.pad(signal, (window_size // 2, window_size // 2), mode="edge")
            avg    = np.convolve(padded, np.ones(window_size) / window_size, mode="valid")
            smoothed[:, joint, coord] = avg
    return smoothed


def predict_skeletons(skeletons, model):
    # Batch inference for ONE boxer. This is the old save_video_with_predictions
    # batch-building code, extracted into a helper so it can be called once per
    # boxer. (The old code also accidentally ran model.predict twice on the same
    # batch - that duplicate is gone here.)
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

    all_predictions = [None] * total_frames
    if not X_batch:
        return all_predictions

    X_batch = np.concatenate(X_batch, axis=0)
    preds = model.predict(X_batch, batch_size=512, verbose=0)
    for i, frame_idx in enumerate(valid_indices):
        all_predictions[frame_idx] = preds[i]
    return all_predictions


def preprocess_window(window_skeletons, mean=X_MEAN_GLOBAL, std=X_STD_GLOBAL):
    X = window_skeletons.reshape(1, 25, 34)
    X = (X - mean) / std
    vel = np.diff(X, axis=1, prepend=np.zeros_like(X[:, :1, :]))
    acc = np.diff(vel, axis=1, prepend=np.zeros_like(vel[:, :1, :]))
    return np.concatenate([X, vel, acc], axis=-1)


def adaptive_ema_smoothing(current_probs, smoothed_history, last_class):
    current_class = np.argmax(current_probs)
    alpha = 0.65 if current_class != last_class else 0.55
    if smoothed_history is None:
        return current_probs, current_class
    smoothed = alpha * current_probs + (1 - alpha) * smoothed_history
    return smoothed, current_class


def is_confident_prediction(probs, threshold_conf=0.35, threshold_entropy=0.70):
    class_idx  = np.argmax(probs)
    confidence = probs[class_idx]
    entropy      = -np.sum(probs * np.log(probs + 1e-9))
    entropy_norm = entropy / np.log(len(probs))
    valid = confidence > threshold_conf and entropy_norm < threshold_entropy
    return class_idx, confidence, valid


def is_valid_skeleton(skeleton_frame, threshold=0.05):
    zero_ratio = np.mean(skeleton_frame == 0)
    return zero_ratio < 0.5 and skeleton_frame.mean() > threshold


# Per-boxer label colors (BGR). Boxer 1 = green, Boxer 2 = cyan. The order
# matches the order select_boxers() returns the ids in (highest score first).
BOXER_COLORS = [(0, 255, 0), (255, 255, 0)]


def save_video_with_predictions(video_path, output_path, boxers, model):
    # CHANGE 5 - PER-BOXER CLASSIFICATION + TWO ON-SCREEN LABELS
    # ---------------------------------------------------------
    # `boxers` is a list (length 1 or 2), one entry per selected boxer:
    #   { "skeletons": dense (total_frames,17,2), "centroids": (total_frames,2) }
    #
    # The OLD version ran the classifier on the one global skeleton and drew a
    # single STATUS bar at the top. The NEW version runs the SAME classifier
    # once per boxer (via predict_skeletons) and draws one color-coded label
    # anchored near each boxer, so two fighters can be read independently.
    print("--> [3/3] Running per-boxer inference...")

    # Classify every boxer up front. Each boxer also keeps its OWN EMA smoothing
    # state (smoothed_probs / last_class) - punches by boxer 1 must not bleed
    # into boxer 2's temporal smoothing, so the state cannot be shared.
    for i, b in enumerate(boxers):
        print(f"--> Boxer {i + 1}: classifying {len(b['skeletons'])} frames...")
        b["predictions"]    = predict_skeletons(b["skeletons"], model)
        b["smoothed_probs"] = None
        b["last_class"]     = None

    temp_output_path = "temp_raw_output.mp4"
    cap = cv2.VideoCapture(video_path)
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (video_width, video_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw a label for each boxer this frame.
        for i, b in enumerate(boxers):
            color = BOXER_COLORS[i % len(BOXER_COLORS)]
            pred  = b["predictions"][frame_idx]

            # Same validity gate as before, now per boxer. When the skeleton is
            # missing/garbage or the model is unsure, show "..." for that boxer.
            if pred is None or not is_valid_skeleton(b["skeletons"][frame_idx]):
                label_text = "..."
            else:
                b["smoothed_probs"], b["last_class"] = adaptive_ema_smoothing(
                    pred, b["smoothed_probs"], b["last_class"]
                )
                class_idx, confidence, valid = is_confident_prediction(b["smoothed_probs"])
                if valid:
                    label_text = f"{BOXING_CLASSES[class_idx]} ({confidence * 100:.1f}%)"
                else:
                    label_text = "..."

            # Anchor the label near the boxer's bbox center (replaces the old
            # single fixed top bar). centroids are normalized 0..1, so scale
            # back to pixels. Sit the text just above the center point.
            cx, cy = b["centroids"][frame_idx]
            px = int(np.clip(cx * video_width, 60, video_width - 220))
            py = int(np.clip(cy * video_height, 40, video_height - 20))

            text = f"Boxer {i + 1}: {label_text}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (px - 5, py - th - 8), (px + tw + 5, py + 4), (0, 0, 0), -1)
            cv2.putText(frame, text, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        frame_text = f"Frame: {frame_idx}"
        (text_width, text_height), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x = video_width - text_width - 20
        y = video_height - 20
        cv2.putText(frame, frame_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

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
        
    video_path = ensure_25fps(args.video)
    
    if args.output is None:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        original_name = os.path.basename(args.video)
        output_path = os.path.join(output_dir, original_name)
    else:
        output_path = args.output

    print("--> Loading TensorFlow model...")
    loaded_model = load_model(args.model)

    # Cache key bumped from "skeletons_" to "tracks_": the cached object is now
    # the multi-person dict {track_id: {...}} returned by extract_skeletons(),
    # NOT the old single (frames,17,2) matrix. The new name stops a stale
    # old-format cache from being loaded into the new code.
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = f"tracks_{video_base_name}.npy"

    if args.clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print("--> Previous cache cleared.")

    if os.path.exists(cache_path):
        print(f"--> Cache found: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True).item()
        tracks, total_frames = cached["tracks"], cached["total_frames"]
    else:
        tracks, total_frames = extract_skeletons(video_path)
        # Stored as a pickled dict (allow_pickle) - the per-track structure is
        # not a plain rectangular array, so wrap it in a 0-d object array.
        np.save(cache_path, np.array({"tracks": tracks, "total_frames": total_frames}, dtype=object))

    # CHANGE 2 + 3 wiring: choose the boxers, then build each one's dense,
    # smoothed skeleton matrix + a forward-filled centroid track for labels.
    boxer_ids = select_boxers(tracks, total_frames)
    if not boxer_ids:
        print("Error: no boxer-like tracks found in video.")
        exit(1)

    boxers = []
    for tid in boxer_ids:
        dense = build_dense_skeletons(tracks[tid], total_frames)   # gap-fill
        dense = smooth_skeletons(dense)                            # per-boxer smoothing

        # Forward-fill centroids the same way as skeletons, so the on-screen
        # label keeps sitting at the boxer's last known spot during short gaps
        # instead of snapping to the (0,0) corner.
        centroids = tracks[tid]["centroids"].copy()
        for idx in range(1, total_frames):
            if not tracks[tid]["present"][idx]:
                centroids[idx] = centroids[idx - 1]

        boxers.append({"skeletons": dense, "centroids": centroids})

    print(f"--> Tracking {len(boxers)} boxer(s).")
    save_video_with_predictions(video_path, output_path, boxers, loaded_model)
