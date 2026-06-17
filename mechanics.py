"""Features de MECANICA do golpe, view-invariantes, sobre esqueleto COCO-17 2D.

Ideia (medida): as coordenadas cruas que o BiLSTM aprende NAO generalizam pra outro
video/camera (OOD). Mas a mecanica do golpe — quanto o cotovelo estende, o quadril
sobe, o tronco gira, o antebraco varre um arco — e a MESMA fisica em qualquer camera.
Entao classificamos o TIPO (reto/hook/uppercut) por essas features, nao por pixel.

Tudo normalizado pela largura de ombro mediana do golpe (escala-invariante) e medido
numa janela centrada no PICO de velocidade do punho (alinha golpes de duracoes diferentes).
A MAO (lead/rear) sai da geometria de stance (ver stance.py); aqui so o TIPO.
"""
import numpy as np

WRI = {0: 9, 1: 10}      # punho esquerdo / direito
SHO = {0: 5, 1: 6}
ELB = {0: 7, 1: 8}
HIP_L, HIP_R, KNE_L, KNE_R = 11, 12, 13, 14

FEATURE_NAMES = [
    "vert_amp", "horz_amp", "reach_max", "reach_rng",
    "elb_pk", "elb_min", "elb_max", "elb_rng",
    "elbH_pk", "elbH_max", "elbH_min", "wristH_pk", "wristH_max",
    "straight", "elb_vel",
    "hip_vert", "sho_vert", "sho_rot", "knee_vert",
    "forearm_sweep", "wrist_sweep",
]


def _ang(j, a, b):
    """Angulo interno (graus) na junta j entre (a-j) e (b-j)."""
    va, vb = a - j, b - j
    return np.degrees(np.arctan2(abs(va[0] * vb[1] - va[1] * vb[0]), (va * vb).sum() + 1e-9))


def _sweep(angles_deg):
    """Amplitude de variacao angular (graus), corrigindo o salto +-180."""
    a = np.unwrap(np.radians(angles_deg))
    return float(np.degrees(a.max() - a.min()))


def active_window(frames, radius=10):
    """Recorta a janela ATIVA do golpe: frames REAIS centrados no pico de VELOCIDADE do
    punho (max dos dois punhos -> independente da mao). `frames` ja deve ser uma vizinhanca
    do golpe (o classificador corta +-radius ao redor do pico de deteccao, ver classify.py).
    Retorna (janela (m,17,2), indice do pico)."""
    f = np.asarray(frames, dtype=np.float64)
    real = (f != 0).any(axis=(1, 2))
    f = f[real]
    if len(f) < 3:
        return f, 0
    sl = np.concatenate([[0.0], np.linalg.norm(np.diff(f[:, WRI[0]], axis=0), axis=1)])
    sr = np.concatenate([[0.0], np.linalg.norm(np.diff(f[:, WRI[1]], axis=0), axis=1)])
    pk = int(np.argmax(np.maximum(sl, sr)))
    a, b = max(0, pk - radius), min(len(f), pk + radius + 1)
    return f[a:b], pk - a


def features(frames, hand, radius=10):
    """Vetor de features de mecanica do golpe (len = len(FEATURE_NAMES)).
    `frames`: sequencia (T,17,2) do golpe. `hand`: 0 esquerda / 1 direita (mao que golpeou).
    A janela ativa e centrada no pico de velocidade (independente da mao)."""
    win, pk = active_window(frames, radius)
    if len(win) < 3:
        return np.zeros(len(FEATURE_NAMES))
    wj, sj, ej = WRI[hand], SHO[hand], ELB[hand]
    sw = np.linalg.norm(win[:, 5] - win[:, 6], axis=1)
    sc = np.median(sw[sw > 1e-3]) if (sw > 1e-3).any() else 1.0
    sc = sc if sc > 1e-3 else 1.0

    w = win[:, wj] / sc                                   # punho
    s = win[:, sj] / sc                                   # ombro
    el = win[:, ej] / sc                                  # cotovelo
    reach = np.linalg.norm(w - s, axis=1)
    angs = np.array([_ang(win[t, ej], win[t, sj], win[t, wj]) for t in range(len(win))])
    elbH = s[:, 1] - el[:, 1]                              # cotovelo acima do ombro (+)
    wristH = s[:, 1] - w[:, 1]                             # punho acima do ombro (+)
    path = np.linalg.norm(np.diff(w, axis=0), axis=1).sum() + 1e-9
    net = np.linalg.norm(w - w[0], axis=1).max()
    forearm = np.degrees(np.arctan2(w[:, 1] - el[:, 1], w[:, 0] - el[:, 0]))
    wrist_a = np.degrees(np.arctan2(w[:, 1] - s[:, 1], w[:, 0] - s[:, 0]))

    midhip = (win[:, HIP_L] + win[:, HIP_R]) / 2 / sc
    midsho = (win[:, 5] + win[:, 6]) / 2 / sc
    sho_ang = np.degrees(np.arctan2(win[:, 6, 1] - win[:, 5, 1], win[:, 6, 0] - win[:, 5, 0]))
    knee = (win[:, KNE_L] + win[:, KNE_R]) / 2 / sc

    return np.array([
        np.ptp(w[:, 1]), np.ptp(w[:, 0]), reach.max(), np.ptp(reach),
        angs[pk], angs.min(), angs.max(), np.ptp(angs),
        elbH[pk], elbH.max(), elbH.min(), wristH[pk], wristH.max(),
        net / path, np.abs(np.diff(angs)).max(),
        np.ptp(midhip[:, 1]), np.ptp(midsho[:, 1]), np.ptp(sho_ang), np.ptp(knee),
        _sweep(forearm), _sweep(wrist_a),
    ])
