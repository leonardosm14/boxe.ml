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
    vel = np.diff(X, axis=1, prepend=X[:, :1, :])
    acc = np.diff(vel, axis=1, prepend=vel[:, :1, :])
    return np.concatenate([X, vel, acc], axis=-1)


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
