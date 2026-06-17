"""
Detecção de stance (qual perna está na frente) e mapeamento
3 classes → 6 classes baseado em stance.

COCO keypoints usados:
  11, 12: hips (esq, dir)
  15, 16: ankles (esq, dir)
"""

import numpy as np
from collections import deque

# COCO joints
HIP_L, HIP_R = 11, 12
ANKLE_L, ANKLE_R = 15, 16

# Mapeamento: 3 classes simplificadas → 6 classes finais
CLASS_MAPPING = {
    "Straight": ["Jab", "Cross"],           # Jab se lead, Cross se rear
    "Uppercut": ["Lead Uppercut", "Rear Uppercut"],
    "Hook": ["Lead Hook", "Rear Hook"]
}

REVERSE_MAPPING = {v: k for k, vals in CLASS_MAPPING.items() for v in vals}


# Em stance_utils_fixed.py, mude detect_stance():

def detect_stance(skeleton_frame, smoothing_window=None, use_ankles=True):
    # Escolher qual ponto usar
    if use_ankles:
        point_l, point_r = ANKLE_L, ANKLE_R
    else:
        point_l, point_r = HIP_L, HIP_R
    
    # ✅ CORRETO: Usar X (não Y)
    pos_l_x = skeleton_frame[point_l, 0]  # ← X, não Y
    pos_r_x = skeleton_frame[point_r, 0]  # ← X, não Y
    
    # Validar
    if pos_l_x <= 0 or pos_r_x <= 0:
        return None
    
    # ✅ OPERADOR CORRETO (foi invertido)
    stance = "L" if pos_r_x > pos_l_x else "R"  # ← "L" if, não "R" if
    
    # Resto do código (smoothing) continua igual...
    if smoothing_window is not None:
        smoothing_window.append(stance)
        stances_window = list(smoothing_window)
        stance = max(set(stances_window), key=stances_window.count)
    
    return stance


def expand_class(class_name_3, stance):
    """
    Mapeia classe simplificada (3) + stance → classe final (6).
    
    Args:
        class_name_3: "Straight", "Uppercut" ou "Hook"
        stance: "L" ou "R"
    
    Returns:
        class_name_6: "Jab", "Cross", "Lead Uppercut", "Rear Uppercut", "Lead Hook", "Rear Hook"
    
    Convenção:
        - Stance "R" (perna dir na frente) = lead side (mão dominante)
          → Straight=Jab, Hook=Lead Hook, Uppercut=Lead Uppercut
        - Stance "L" (perna esq na frente) = rear side (mão traseira)
          → Straight=Cross, Hook=Rear Hook, Uppercut=Rear Uppercut
    """
    if class_name_3 not in CLASS_MAPPING:
        return None
    
    options = CLASS_MAPPING[class_name_3]
    
    if stance == "R":
        return options[0]  # Lead/Jab
    elif stance == "L":
        return options[1]  # Rear/Cross
    else:
        return None


def collapse_class(class_name_6):
    """Inverte: classe 6 → classe 3 (Jab/Cross → Straight, etc)."""
    return REVERSE_MAPPING.get(class_name_6, None)


def apply_stance_smoothing(skeletons, window_size=7):
    """
    Pré-computa stance para todos os frames com suavização.
    
    Args:
        skeletons: (n_frames, 17, 2) keypoints
        window_size: tamanho da janela de suavização (maioria ganha)
    
    Returns:
        stances: list de "L", "R", ou None por frame
    """
    stances = []
    window = deque(maxlen=window_size)
    
    for frame in skeletons:
        stance = detect_stance(frame, smoothing_window=window)
        stances.append(stance)
    
    return stances