"""Inferencia num video: pose -> deteccao por pico -> classificacao geometrica ->
rotulo continuo por golpe -> render. Saida: video anotado + labels_<base>.npy.

Pipeline decomposta (ver PLANO_V2.md): TIPO por geometria (gate uppercut + arco hook),
MAO por geometria de stance, 6 classes = MAP(TIPO, MAO). Cada golpe recebe UM rotulo
contiguo do onset ao offset -> sem piscar no video.
"""
import argparse
import os
import subprocess
import cv2
import numpy as np

import pose
import detect
import classify as C
import stance as st

CLASSES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]


def label_video(skeletons):
    """Roda a pipeline e devolve (labels (T,) com nome ou None, spans, stance)."""
    spans = detect.detect_punches(skeletons)
    # stance global do clipe a partir dos golpes detectados
    windows = []
    for on, pk, off in spans:
        w = np.zeros((25, 17, 2)); seg = skeletons[on:min(on + 25, off + 1)]; w[:len(seg)] = seg
        windows.append(w)
    stance = st.clip_stance(np.array(windows)) if windows else 1
    labels = np.full(len(skeletons), None, dtype=object)
    out_spans = []
    for on, pk, off in spans:
        cls = C.classify_punch(skeletons, (on, off), stance)
        labels[on:off + 1] = cls
        out_spans.append((on, off, cls))
    return labels, out_spans, stance


def render(video_path, output_path, skeletons, labels):
    """Desenha a classe por frame. Conta trocas de classe DENTRO de um golpe (ideal 0)."""
    cap = cv2.VideoCapture(video_path)
    vw, vh = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5) or 25.0
    tmp = "_tmp_render.mp4"
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    i, punch_frames, switches, prev = 0, 0, 0, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lab = labels[i] if i < len(labels) else None
        if lab is not None:
            txt, color = lab, (0, 255, 0); punch_frames += 1
            if prev is not None and prev != lab:
                switches += 1
        else:
            txt, color = "...", (200, 200, 200)
        prev = lab
        cv2.rectangle(frame, (0, 0), (vw, 58), (0, 0, 0), -1)
        cv2.putText(frame, txt, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"f{i}", (vw - 70, vh - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame); i += 1
    cap.release(); out.release()
    subprocess.run(["ffmpeg", "-y", "-i", tmp, "-vcodec", "libx264", "-pix_fmt", "yuv420p", output_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    os.remove(tmp)
    print(f"--> render: {punch_frames}/{i} frames com golpe | trocas dentro de golpe={switches} (ideal 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.video))[0]
    sk, _ = pose.load_or_extract(args.video, cache_path=f"skeletons_{base}.npy")
    labels, spans, stance = label_video(sk)
    print(f"--> {len(spans)} golpes (stance={'orthodox' if stance > 0 else 'southpaw'}):")
    for on, off, cls in spans:
        print(f"    [{on:>4}-{off:<4}] {cls}")
    np.save(f"labels_{base}.npy", np.array([x if x else "" for x in labels]))
    if not args.no_render:
        out = args.output or os.path.join("outputs", os.path.basename(args.video))
        os.makedirs("outputs", exist_ok=True)
        render(args.video, out, sk, labels)
        print(f"--> salvo: {out}")


if __name__ == "__main__":
    main()
