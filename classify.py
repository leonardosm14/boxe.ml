"""Classificacao do golpe: TIPO (reto/hook/uppercut) + MAO (lead/rear) -> 6 classes.

TIPO e hibrido:
  - UPPERCUT por GATE geometrico (fisica): o punho SOBE muito e o cotovelo CAI abaixo do
    ombro (movimento ascendente). Robusto a camera (vertical = gravidade em qualquer vista).
    O classificador treinado erra uppercut OOD (extrator/vista diferentes); o gate resolve.
  - reto vs hook pelo classificador de mecanica (reto e hook so diferem em sutilezas de
    trajetoria; e o caso dificil, ~0.55-0.65 cross-video — limite fisico em 2D).
MAO por geometria de stance (stance.py).
6-classes = MAP(TIPO, MAO).
"""
import numpy as np
import stance as st
import mechanics as mc

TYPE_NAMES = ["reto", "hook", "uppercut"]
MAP6 = {(0, "lead"): "Jab", (0, "rear"): "Cross",
        (1, "lead"): "Lead Hook", (1, "rear"): "Rear Hook",
        (2, "lead"): "Lead Uppercut", (2, "rear"): "Rear Uppercut"}

# indices das features de mecanica (ver mechanics.FEATURE_NAMES)
F_VERT = 0          # amplitude vertical do punho
F_ELBH_MIN = 10     # cotovelo mais baixo rel. ombro (negativo = caiu abaixo)
F_FOREARM_SWEEP = 19
F_WRIST_SWEEP = 20

# Limiares geometricos (vista lateral; calibrados no adam — ver _tune_gate.py).
# Tudo fisica do golpe (invariante a escala pela largura de ombro). NAO ha modelo treinado
# aqui: classificador treinado em outro extrator/vista nao transfere (medido).
UPP_VERT_MIN = 1.05     # uppercut: punho sobe > ~1 largura de ombro
UPP_ELBH_MAX = -0.15    # uppercut: cotovelo cai abaixo do ombro (drive ascendente)
HOOK_SWEEP_MIN = 95.0   # hook: antebraco+punho varrem um arco grande (graus); reto vai reto


def is_uppercut(feat):
    """Gate geometrico do uppercut: movimento ascendente forte com cotovelo baixo."""
    return feat[F_VERT] >= UPP_VERT_MIN and feat[F_ELBH_MIN] <= UPP_ELBH_MAX


def is_hook(feat):
    """reto vs hook: o hook curva (antebraco varre arco grande); o reto vai reto."""
    return (feat[F_FOREARM_SWEEP] + feat[F_WRIST_SWEEP]) >= HOOK_SWEEP_MIN


def punch_type(feat):
    """TIPO 0 reto / 1 hook / 2 uppercut por geometria."""
    if is_uppercut(feat):
        return 2
    return 1 if is_hook(feat) else 0


def classify_punch(skeletons, span, stance):
    """Classifica o golpe no intervalo span=(onset, offset) -> nome da classe (6).
    A mao/guarda usam o span inteiro (mais contexto = melhor); as features de TIPO usam a
    janela ativa (pico de velocidade) dentro dele."""
    s, e = span
    win = np.asarray(skeletons)[s:e + 1]
    hand = st.hand_of_punch(win)
    feat = mc.features(win, hand)
    typ = punch_type(feat)
    side = st.lead_rear(win, stance)
    return MAP6[(typ, side)]
