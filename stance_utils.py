"""
Expansão 3 classes -> 6 classes via inferência lead/rear por golpe.

O modelo LSTM é treinado em 3 classes de TIPO (Straight/Hook/Uppercut) porque o
tipo é aprendível da trajetória; a MÃO (lead vs rear) é ambígua numa janela
isolada e é decidida aqui, por GEOMETRIA pura, na hora da inferência.

A decisão lead/rear vive em stance.py (ver nota empírica no topo daquele módulo):
a stance postural ESTÁTICA (comparar x/y de tornozelos ou quadris) foi testada e
fica em nível de ruído neste dataset — o YOLOv8-pose alucina o lado ocluído e a
câmera lateral colapsa a profundidade frente/trás. O sinal que funciona (medido:
0.85 train / 0.77 cross-video / 0.74 test de acurácia lead/rear) é DINÂMICO e POR
GOLPE: o punho de maior deslocamento líquido define a mão que golpeou, e a
extensão desse punho define localmente a "frente" — o pé desse lado é o lead.

Este módulo é só a camada de mapeamento TIPO + lado -> classe final.
"""

from stance import lead_rear, hand_of_punch  # noqa: F401  (re-export p/ conveniência)

# Mapeamento: classe de TIPO (saída do modelo) + lado -> classe final (6)
CLASS_MAPPING = {
    "Straight": {"lead": "Jab",           "rear": "Cross"},
    "Hook":     {"lead": "Lead Hook",     "rear": "Rear Hook"},
    "Uppercut": {"lead": "Lead Uppercut", "rear": "Rear Uppercut"},
}

REVERSE_MAPPING = {
    final: tipo for tipo, sides in CLASS_MAPPING.items() for final in sides.values()
}


def expand_class(class_name_3, side):
    """Mapeia classe de tipo (3) + lado ('lead'/'rear') -> classe final (6).

    Args:
        class_name_3: "Straight", "Hook" ou "Uppercut"
        side: "lead" ou "rear" (saída de stance.lead_rear sobre a janela do golpe)

    Returns:
        "Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"
        ou None se a classe/lado forem desconhecidos.
    """
    return CLASS_MAPPING.get(class_name_3, {}).get(side)


def collapse_class(class_name_6):
    """Inverte: classe final (6) -> classe de tipo (3)."""
    return REVERSE_MAPPING.get(class_name_6)
