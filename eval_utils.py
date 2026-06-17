"""Utilitários de AVALIAÇÃO (sem TensorFlow). Tudo em numpy puro, autossuficiente.

Avaliamos o classificador de golpes com validação cruzada leave-one-video-out
(LOVO) e reportamos:
  - acurácia com intervalo de confiança (bootstrap);
  - pisos de comparação (chute majoritário / aleatório);
  - métricas SEGMENTAIS de segmentação de ação (edit score + F1@k), porque a
    queixa real é o rótulo "piscando" (blinking) ao longo do vídeo, e a acurácia
    por frame é cega para isso — ela premia previsões fragmentadas que acertam
    frame a frame mas trocam de rótulo o tempo todo;
  - calibração (ECE), pra saber se a confiança do softmax bate com o acerto.

As métricas segmentais seguem a semântica padrão do MS-TCN/ASRF (Farha & Gall,
CVPR 2019; Lea et al.) — edit score sobre a sequência de rótulos dos segmentos e
F1@overlap por IoU temporal com casamento guloso 1-pra-1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Validação cruzada
# ---------------------------------------------------------------------------
def lovo_split(vids):
    """Leave-one-video-out: pra cada vídeo v, gera (todos os outros, v).
    `vids` é uma lista tipo ["V1",...,"V10"]. Gera tuplas (train_vids, test_vid)."""
    vids = list(vids)
    for v in vids:
        train = [u for u in vids if u != v]
        yield train, v


# ---------------------------------------------------------------------------
# Acurácia + intervalo de confiança
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, n=1000, seed=0):
    """Acurácia pontual + IC 95% por bootstrap. Reamostra os índices com reposição
    `n` vezes e pega os percentis 2.5 / 97.5 das acurácias. Retorna (acc, lo, hi)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_true == y_pred))
    N = len(y_true)
    if N == 0:
        return acc, acc, acc
    rng = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(np.float64)
    accs = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, N, size=N)        # reamostragem com reposição
        accs[i] = correct[idx].mean()
    lo = float(np.percentile(accs, 2.5))
    hi = float(np.percentile(accs, 97.5))
    return acc, lo, hi


def floors(y_true):
    """Pisos de comparação a partir da distribuição real dos rótulos:
    `majoritario` = fração da classe mais frequente (acurácia de chutar sempre ela);
    `aleatorio`   = 1/num_classes (chute uniforme). Retorna (majoritario, aleatorio)."""
    y_true = np.asarray(y_true)
    _, counts = np.unique(y_true, return_counts=True)
    majoritario = float(counts.max() / counts.sum())
    aleatorio = float(1.0 / len(counts))
    return majoritario, aleatorio


# ---------------------------------------------------------------------------
# Calibração: Expected Calibration Error
# ---------------------------------------------------------------------------
def ece(probs, y_true, bins=15):
    """Expected Calibration Error. `probs` é (N,C) softmax, `y_true` (N,) int.
    Padrão: agrupa pela confiança (prob máxima) em `bins` faixas de largura igual
    em [0,1]; ECE = soma_m (|B_m|/N) * |acc(B_m) - conf(B_m)|. Retorna float.
    Mede o quanto a confiança do modelo descola da taxa real de acerto."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    N = len(y_true)
    if N == 0:
        return 0.0
    conf = probs.max(axis=1)                 # confiança = prob da classe escolhida
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)  # faixas de largura igual em [0,1]
    total = 0.0
    for m in range(bins):
        lo, hi = edges[m], edges[m + 1]
        mask = (conf > lo) & (conf <= hi)    # faixa (lo, hi]
        if m == 0:
            mask |= (conf == lo)             # 1º bin também pega conf == 0.0
        if not mask.any():
            continue
        acc_m = correct[mask].mean()
        conf_m = conf[mask].mean()
        total += (mask.sum() / N) * abs(acc_m - conf_m)
    return float(total)


# ---------------------------------------------------------------------------
# Métricas SEGMENTAIS (MS-TCN / ASRF)
# ---------------------------------------------------------------------------
def get_segments(seq):
    """Colapsa uma sequência de rótulos por frame em runs contíguos.
    Retorna lista de (label, start, end), com `end` inclusivo (índice do
    último frame do run). Ex.: [0,0,1] -> [(0,0,1),(1,2,2)]."""
    seq = np.asarray(seq)
    segs = []
    if len(seq) == 0:
        return segs
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            segs.append((int(seq[start]), start, i - 1))
            start = i
    segs.append((int(seq[start]), start, len(seq) - 1))
    return segs


def _edit_distance(a, b):
    """Distância de Levenshtein entre duas sequências (DP simples, O(len_a*len_b)).
    Aqui `a` e `b` são listas de rótulos de segmento."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # dp[j] = distância entre a[:i] e b[:j]; mantemos só a linha anterior
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # remoção
                cur[j - 1] + 1,     # inserção
                prev[j - 1] + cost  # substituição/igual
            )
        prev = cur
    return prev[lb]


def segmental_edit(seq_true, seq_pred):
    """Edit score normalizado do MS-TCN. Compara só a sequência ORDENADA de rótulos
    dos segmentos (ignora durações), penalizando fragmentação/troca de rótulo.
    Retorna (1 - edit/max(n_true, n_pred)) * 100. 100 = sequência de golpes idêntica."""
    lab_true = [s[0] for s in get_segments(seq_true)]
    lab_pred = [s[0] for s in get_segments(seq_pred)]
    denom = max(len(lab_true), len(lab_pred))
    if denom == 0:
        return 100.0
    dist = _edit_distance(lab_true, lab_pred)
    return (1.0 - dist / denom) * 100.0


def _temporal_iou(seg_a, seg_b):
    """IoU temporal entre dois segmentos (label, start, end) com end inclusivo.
    Interseção/união contadas em frames."""
    _, s1, e1 = seg_a
    _, s2, e2 = seg_b
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)   # +1: end inclusivo
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def segmental_f1(seq_true, seq_pred, overlap):
    """F1@overlap do MS-TCN. Pra cada segmento PREVISTO, é verdadeiro positivo se
    existe um segmento real de MESMO rótulo, ainda não casado, com IoU temporal
    >= overlap (casamento guloso 1-pra-1: cada segmento real é usado no máximo uma
    vez). Senão é falso positivo. Segmentos reais não casados são falsos negativos.
    Retorna (precision, recall, f1), todos *100. `overlap` é fração tipo 0.25."""
    segs_true = get_segments(seq_true)
    segs_pred = get_segments(seq_pred)
    matched = [False] * len(segs_true)   # cada GT só pode casar uma vez
    tp = 0
    for sp in segs_pred:
        best_iou = 0.0
        best_j = -1
        for j, st in enumerate(segs_true):
            if matched[j] or st[0] != sp[0]:   # mesmo rótulo e ainda livre
                continue
            iou = _temporal_iou(sp, st)
            if iou >= overlap and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp += 1
            matched[best_j] = True
    fp = len(segs_pred) - tp
    fn = len(segs_true) - sum(matched)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision * 100.0, recall * 100.0, f1 * 100.0


# ---------------------------------------------------------------------------
# Smoke test: roda `python3 eval_utils.py` pra provar que tudo funciona
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Duas sequências por frame minúsculas: a previsão "pisca" cedo (frame 2 vira 1)
    true = [0, 0, 0, 1, 1, 2, 2, 2]
    pred = [0, 0, 1, 1, 1, 2, 2, 2]

    print("segmentos true:", get_segments(true))
    print("segmentos pred:", get_segments(pred))
    print("edit score    : %.2f" % segmental_edit(true, pred))
    for k in (0.10, 0.25, 0.50):
        p, r, f1 = segmental_f1(true, pred, k)
        print("F1@%.2f       : P=%.1f R=%.1f F1=%.1f" % (k, p, r, f1))

    # bootstrap + pisos num exemplo trivial
    yt = np.array([0, 0, 1, 1, 2, 2])
    yp = np.array([0, 0, 1, 2, 2, 2])
    acc, lo, hi = bootstrap_ci(yt, yp, n=500, seed=0)
    print("acc=%.3f  IC95=[%.3f, %.3f]" % (acc, lo, hi))
    print("pisos (maj, aleat): %.3f, %.3f" % floors(yt))

    # LOVO sanity
    print("lovo splits:", list(lovo_split(["V1", "V2", "V3"])))

    # ECE: probs perfeitamente confiantes e certas -> ECE alto só se errar
    probs = np.array([
        [0.9, 0.1],   # prevê 0, certo
        [0.8, 0.2],   # prevê 0, certo
        [0.6, 0.4],   # prevê 0, ERRADO (verdadeiro=1)
        [0.3, 0.7],   # prevê 1, certo
    ])
    y = np.array([0, 0, 1, 1])
    print("ECE exemplo   : %.4f" % ece(probs, y, bins=10))
