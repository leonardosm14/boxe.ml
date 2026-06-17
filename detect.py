"""Deteccao temporal dos golpes: cada golpe = um PICO de velocidade do punho.

Histerese/Schmitt funde combos rapidos e perde golpes leves; o pico capta cada golpe
isolado. O span do golpe vai do onset (velocidade subiu acima do piso) ate o offset
(caiu abaixo), limitado ao meio entre picos vizinhos. Sinal em referencial do CORPO
(recentrado no quadril, escalado pela largura de ombro) -> comparavel entre cameras.
"""
import numpy as np
from scipy.signal import find_peaks

WRIST = [9, 10]
SHO_L, SHO_R, HIP_L, HIP_R = 5, 6, 11, 12

# Deteccao por EXTENSAO: um golpe tem UM maximo de alcance punho-ombro (extensao total),
# enquanto a velocidade pica DUAS vezes (ida + volta) e super-segmenta golpes lentos. O
# pico de alcance da 1 golpe = 1 deteccao (medido no adam: 18/18, 0 falsos). O span (onset
# -> offset) vem da velocidade (movimento ativo ao redor da extensao).
PROMINENCE = 0.30
MIN_DISTANCE = 8
SPAN_FRAC = 0.20      # span = onde o alcance fica acima de base + SPAN_FRAC*(pico-base).
#                       menor = span mais largo (mais contexto p/ mao/guarda na classificacao)


def _bodyframe(skeletons):
    sk = np.asarray(skeletons, dtype=np.float64)
    det = (sk != 0).any(axis=2)
    mid_hip = (sk[:, HIP_L] + sk[:, HIP_R]) / 2.0
    sw = np.linalg.norm(sk[:, SHO_L] - sk[:, SHO_R], axis=1)
    scale = np.median(sw[sw > 1e-3]) if (sw > 1e-3).any() else 1.0
    return np.where(det[..., None], (sk - mid_hip[:, None, :]) / max(scale, 1e-3), 0.0), max(scale, 1e-3)


def _smooth(s, k=5):
    return np.convolve(np.pad(s, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def wrist_speed(skeletons):
    """Velocidade do punho dominante (referencial do corpo, suavizada). (T,) >= 0."""
    norm, _ = _bodyframe(skeletons)
    d = np.diff(norm[:, WRIST, :], axis=0)
    spd = np.concatenate([[0.0], np.linalg.norm(d, axis=2).max(axis=1)])
    return _smooth(spd)


def reach_signal(skeletons):
    """Maior alcance punho-ombro dos dois bracos (extensao), referencial do corpo, suavizado.
    Pica UMA vez por golpe (na extensao total). (T,)."""
    sk = np.asarray(skeletons, dtype=np.float64)
    _, sc = _bodyframe(sk)
    rl = np.linalg.norm(sk[:, 9] - sk[:, SHO_L], axis=1) / sc
    rr = np.linalg.norm(sk[:, 10] - sk[:, SHO_R], axis=1) / sc
    return _smooth(np.maximum(rl, rr))


def detect_punches(skeletons, prominence=PROMINENCE, distance=MIN_DISTANCE, span_frac=SPAN_FRAC):
    """Golpes por pico de EXTENSAO. Retorna spans [(onset, peak, offset)], 1 por golpe.

    O span segue o ALCANCE (braco estendido), nao a velocidade: na extensao maxima o punho
    esta momentaneamente lento, entao delimitar por velocidade colapsaria o span. Expande do
    pico enquanto o alcance fica acima de um limiar (base + span_frac*(pico-base)) -> cobre
    o golpe inteiro (ida + volta), 1 rotulo continuo, sem piscar."""
    reach = reach_signal(skeletons)
    peaks, _ = find_peaks(reach, prominence=prominence, distance=distance)
    spans = []
    for i, pk in enumerate(peaks):
        lo = (peaks[i - 1] + pk) // 2 if i > 0 else 0
        hi = (pk + peaks[i + 1]) // 2 if i < len(peaks) - 1 else len(reach) - 1
        base = min(reach[lo:pk + 1].min(), reach[pk:hi + 1].min())   # vale local
        thr = base + span_frac * (reach[pk] - base)                  # limiar do span
        on = pk
        while on > lo and reach[on - 1] >= thr:
            on -= 1
        off = pk
        while off < hi and reach[off + 1] >= thr:
            off += 1
        spans.append((on, pk, off))
    return spans
