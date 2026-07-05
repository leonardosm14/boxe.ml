"""Stance (guarda) e mão lead/rear por GEOMETRIA pura sobre esqueletos COCO-17 2D.

Sem aprendizado de máquina: só geometria sobre os 17 keypoints normalizados [0,1].
A ideia (ver IDEIA_STANCE.md): o TIPO do golpe (reto/hook/uppercut) o modelo aprende
bem; a MÃO (lead vs rear) é ambígua de uma janela isolada, mas decorre da GUARDA.
Orthodox = pé/mão ESQUERDA na frente -> lead = esquerda (0).
Southpaw = pé/mão DIREITA na frente -> lead = direita (1).

COCO-17: 0 nariz · 1/2 olho E/D · 3/4 orelha E/D · 5/6 ombro E/D · 7/8 cotovelo E/D ·
9/10 punho E/D · 11/12 quadril E/D · 13/14 joelho E/D · 15/16 tornozelo E/D.
Junta (0,0) = NÃO detectada (sem canal de confiança separado). y cresce para baixo.

NOTA EMPÍRICA (medida neste dataset BoxingVI, ver bloco __main__):
o YOLOv8-pose aqui SEMPRE devolve as 17 juntas (lado oculto incluído, alucinado), e a
câmera é lateral fixa: a PROFUNDIDADE frente/trás está colapsada na projeção 2D. Por isso
a stance postural estática (qual pé está "atrás" em profundidade) fica em nível de ruído.
O sinal que sobrevive é a DIREÇÃO DINÂMICA do golpe: no instante de cada soco, a extensão
do punho que golpeou define localmente a "frente", e o pé desse lado é o lead. Logo a
decisão lead/rear é feita POR JANELA (por golpe), e não por uma stance global do clipe —
forçar um stance único por clipe mistura frames de orientações diferentes e cancela o
sinal (medido: cai de ~0.76 para ~0.61 no cross-video).
"""

import numpy as np

# Índices COCO-17 usados aqui
NOSE = 0
EYE_L, EYE_R = 1, 2
EAR_L, EAR_R = 3, 4
SHO_L, SHO_R = 5, 6
WRI_L, WRI_R = 9, 10
HIP_L, HIP_R = 11, 12
KNE_L, KNE_R = 13, 14
ANK_L, ANK_R = 15, 16

# Conjunto de golpes da MÃO da frente (lead). O resto é rear.
LEAD_CLASSES = {"Jab", "Lead Hook", "Lead Uppercut"}


def _detected(kp):
    """Junta detectada = não é exatamente (0,0)."""
    return bool((kp != 0).any())


def _real_frames(skeletons):
    """Filtra os frames reais (descarta o padding de zeros no fim).
    skeletons: (T,17,2). Retorna (n,17,2) só com os frames reais."""
    sk = np.asarray(skeletons, dtype=np.float64)
    mask = (sk != 0).any(axis=(1, 2))
    return sk[mask]


def _joint_mean(r, j):
    """Posição média da junta j ao longo dos frames reais r, ignorando onde ela
    não foi detectada. Retorna (2,) ou None se nunca detectada."""
    t = r[:, j, :]
    det = (t != 0).any(axis=1)
    return t[det].mean(axis=0) if det.any() else None


def _leg_point(r, ankle, knee):
    """Ponto representativo da perna = média de tornozelo e joelho (perna inteira),
    usando o que estiver detectado. Mais robusto a oclusão do pé do que só o tornozelo.
    Retorna (2,) ou None se ambos faltam."""
    pts = [p for p in (_joint_mean(r, ankle), _joint_mean(r, knee)) if p is not None]
    return np.mean(pts, axis=0) if pts else None


def body_facing_normal(frame):
    """Normal de orientação do corpo na imagem, a partir de UM frame (17,2).

    a = vetor através do corpo (lado esquerdo -> lado direito), média do eixo dos
    ombros (6-5) e dos quadris (12-11), usando só as juntas detectadas. A normal
    n = (-a_y, a_x) é perpendicular a `a` no plano da imagem (frente/trás do corpo).
    Retorna None se faltarem ombros E quadris para definir `a`.

    Obs.: neste dataset (pose alucinada, câmera lateral) o SINAL de n não distingue
    orthodox de southpaw de forma confiável; por isso a decisão lead/rear usa a
    direção dinâmica do golpe (#hand_of_punch + extensão), não esta normal estática.
    Mantida porque é a definição geométrica de orientação do torso pedida no contrato.
    """
    f = np.asarray(frame, dtype=np.float64)
    parts = []
    if _detected(f[SHO_L]) and _detected(f[SHO_R]):
        parts.append(f[SHO_R] - f[SHO_L])
    if _detected(f[HIP_L]) and _detected(f[HIP_R]):
        parts.append(f[HIP_R] - f[HIP_L])
    if not parts:
        return None
    a = np.mean(parts, axis=0)            # esquerda -> direita do corpo
    if np.linalg.norm(a) < 1e-9:
        return None
    n = np.array([-a[1], a[0]])           # perpendicular (frente/trás)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    return n / norm


def _net_disp(r, j):
    """Maior afastamento da junta j em relação à sua posição inicial (pico de
    `||p_t - p_0||`) nos frames reais em que ela foi detectada."""
    traj = r[:, j, :]
    det = (traj != 0).any(axis=1)
    pts = traj[det]
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts - pts[0], axis=1).max())


def _reach(r, wrist, shoulder):
    """Maior alcance punho-ombro no pico (`max ||punho - ombro||`), nos frames em que
    ambos foram detectados. Mede a extensão do braço; bom desempate quando os
    deslocamentos das duas mãos são parecidos."""
    t = r[:, wrist, :]
    s = r[:, shoulder, :]
    det = (t != 0).any(axis=1) & (s != 0).any(axis=1)
    return float(np.linalg.norm((t - s)[det], axis=1).max()) if det.any() else 0.0


def hand_of_punch(window):
    """Qual punho desferiu o golpe: o de maior DESLOCAMENTO líquido na janela.

    Para cada punho mede-se o maior afastamento da posição inicial (#_net_disp).
    Usa-se o deslocamento líquido (e não o comprimento total do caminho) porque o
    caminho total acumula o jitter do detector e pode eleger a mão errada; o pico de
    afastamento capta a extensão real do soco. Quando os dois punhos têm deslocamento
    parecido (diferença < 10%), desempata pelo maior ALCANCE punho-ombro (#_reach),
    que separa melhor o braço que de fato estendeu. Retorna 0 (esquerda) ou 1 (direita).
    """
    r = _real_frames(window)
    if len(r) < 2:
        return 0
    dl, dr = _net_disp(r, WRI_L), _net_disp(r, WRI_R)
    if abs(dl - dr) < 0.10 * max(dl, dr, 1e-9):
        rl, rr = _reach(r, WRI_L, SHO_L), _reach(r, WRI_R, SHO_R)
        return 0 if rl >= rr else 1
    return 0 if dl >= dr else 1


def _window_forward_sign(window):
    """Direção 'frente' da janela: sinal do deslocamento em x do punho que golpeou,
    da origem até o ponto de maior extensão. +1 = frente para a direita na imagem,
    -1 = para a esquerda. Retorna (sinal, extensão) ou None se inutilizável.

    É a 'frente' LOCAL do golpe (o que de fato funciona aqui), não uma frente global
    do clipe: cada soco se estende para onde o corpo aponta naquele instante.
    """
    r = _real_frames(window)
    if len(r) < 2:
        return None
    hand = hand_of_punch(window)
    wj = WRI_L if hand == 0 else WRI_R
    traj = r[:, wj, :]
    det = (traj != 0).any(axis=1)
    traj = traj[det]
    if len(traj) < 2:
        return None
    dist = np.linalg.norm(traj - traj[0], axis=1)
    pk = int(np.argmax(dist))
    s = np.sign(traj[pk, 0] - traj[0, 0])
    return (1 if s == 0 else int(s)), float(dist[pk])


def _window_lead_side(window):
    """Lado da mão lead SEGUNDO a geometria desta janela: o pé (perna inteira) que
    está mais na direção 'frente' do golpe é o pé da frente; a mão do mesmo lado é a
    lead. Retorna (lead_side 0/1, peso=extensão do golpe) ou None.
    """
    fs = _window_forward_sign(window)
    if fs is None:
        return None
    s, ext = fs
    r = _real_frames(window)
    fl = _leg_point(r, ANK_L, KNE_L)
    fr = _leg_point(r, ANK_R, KNE_R)
    if fl is None or fr is None:
        return None
    # pé na frente = maior x na direção s. lead_side=0 (esquerda) se o pé esq. está à frente.
    lead_side = 0 if fl[0] * s > fr[0] * s else 1
    return lead_side, ext


def clip_stance(skeletons):
    """Stance do clipe por VOTO MAJORITÁRIO ponderado sobre as janelas/golpes.

    `skeletons` pode ser uma sequência (T,17,2), uma janela (25,17,2), ou o vídeo
    inteiro empilhado em janelas (N,25,17,2). Cada janela vota em qual lado é o lead
    (#_window_lead_side), ponderada pela extensão do golpe (golpes maiores = sinal mais
    limpo). Retorna +1 se orthodox (lead = esquerda) ou -1 se southpaw (lead = direita).
    Sem votos -> +1 (orthodox, a guarda mais comum) como fallback neutro.

    A guarda é nominalmente constante no clipe; este agregado dá a handedness global do
    lutador. Mas, como a câmera/orientação variam, a decisão lead/rear POR GOLPE
    (#lead_rear, por janela) é mais precisa que aplicar esta stance global a todos os
    golpes — ver nota no topo do módulo.
    """
    arr = np.asarray(skeletons, dtype=np.float64)
    windows = _as_windows(arr)
    s0 = s1 = 0.0
    for w in windows:
        v = _window_lead_side(w)
        if v is None:
            continue
        lead_side, weight = v
        if lead_side == 0:
            s0 += weight
        else:
            s1 += weight
    if s0 == 0.0 and s1 == 0.0:
        return 1
    return 1 if s0 >= s1 else -1


def _as_windows(arr):
    """Normaliza a entrada para uma lista de janelas (cada uma (m,17,2)).
    Aceita (N,25,17,2) -> N janelas; (T,17,2) -> 1 janela."""
    if arr.ndim == 4:
        return [arr[i] for i in range(arr.shape[0])]
    return [arr]


def lead_rear(window, stance=None):
    """Classifica a mão do golpe desta janela como 'lead' ou 'rear'.

    Decisão POR JANELA (a que maximiza a acurácia, ver nota no topo): combina a mão
    que golpeou (#hand_of_punch) com o pé da frente segundo a direção de extensão do
    próprio golpe (#_window_lead_side). `stance` é aceito por compatibilidade de
    contrato; quando a geometria da janela é inconclusiva, cai para a stance global
    do clipe (+1 orthodox -> lead=esquerda(0); -1 southpaw -> lead=direita(1)).
    """
    hand = hand_of_punch(window)              # 0 esquerda, 1 direita
    wl = _window_lead_side(window)
    if wl is not None:
        lead_side = wl[0]
    else:
        # fallback: usa a stance global passada (ou orthodox se ausente)
        st = stance if stance is not None else 1
        lead_side = 0 if st > 0 else 1
    return "lead" if hand == lead_side else "rear"


def stance_confidence(skeletons):
    """Confiança da stance do clipe: fração (ponderada pela extensão) dos votos de
    janela que concordam com a decisão final de #clip_stance. ~1.0 = todos os golpes
    apontam a mesma guarda (sinal nítido); ~0.5 = empate (ângulos/oclusão ruins).
    Proxy de confiabilidade. Sem votos -> 0.0.
    """
    arr = np.asarray(skeletons, dtype=np.float64)
    windows = _as_windows(arr)
    final = clip_stance(arr)
    final_side = 0 if final > 0 else 1
    agree = total = 0.0
    for w in windows:
        v = _window_lead_side(w)
        if v is None:
            continue
        lead_side, weight = v
        total += weight
        if lead_side == final_side:
            agree += weight
    return float(agree / total) if total > 0 else 0.0


if __name__ == "__main__":
    from boxe_utils import load_video

    SPLITS = {
        "TRAIN (V1,V2,V7,V8)": ["V1", "V2", "V7", "V8"],
        "CROSS-VIDEO (V3,V4)": ["V3", "V4"],
        "TEST (V5,V9)": ["V5", "V9"],
    }

    for split_name, videos in SPLITS.items():
        tot = ok = 0
        confs = []
        for v in videos:
            sk, lb = load_video(v)
            W = sk.reshape(-1, 25, 17, 2)
            stance = clip_stance(W)          # handedness global (informativa)
            confs.append(stance_confidence(W))
            for window, label in zip(W, lb):
                r = _real_frames(window)
                if len(r) < 2:
                    continue
                pred = lead_rear(window, stance)   # decisão por golpe
                truth = "lead" if label in LEAD_CLASSES else "rear"
                tot += 1
                ok += int(pred == truth)
        acc = ok / max(tot, 1)
        conf = float(np.mean(confs)) if confs else 0.0
        print(f"{split_name}: n={tot} | lead/rear acc={acc:.3f} | stance_conf={conf:.3f}")
