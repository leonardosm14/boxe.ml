"""Decodificação temporal da saída do classificador -> 1 rótulo contíguo por golpe.

O problema visível no vídeo: o rótulo "pisca". Frame a frame o classificador troca
de classe ou cai pra nada no meio de um único golpe (over-segmentation, no jargão de
action segmentation — MS-TCN / ASRF / Viterbi). A causa é tratar cada frame/janela como
uma decisão independente, sem prior temporal de que um golpe é um trecho contíguo.

A correção é pós-processar as posteriors por-frame com um decodificador "grudento"
(sticky), que penaliza trocar de estado a cada frame. Aqui ficam as primitivas puras
desse pós-processamento (só numpy; sem dependência de projeto), prontas pra serem
chamadas pelo boxe.py por cima das probabilidades já calculadas pelo modelo.

HONESTIDADE: o decodificador conserta FLICKER, não correção. Viterbi sobre um stream
errado dá um rótulo liso-porém-errado. Por isso decode_clip aceita um estado de fundo
(bg_class) — pra frames de repouso não serem empurrados pra dentro de um golpe — e um
piso de confiança (conf_floor) pra rejeitar spans ambíguos.
"""

import numpy as np
from scipy.signal import find_peaks


def detect_punches(speed, prominence=0.4, distance=8, floor=0.3):
    """Detecção POR PICO (não por histerese). Cada golpe = um pico local na velocidade do
    punho. Histerese/Schmitt funde combos rápidos e perde golpes leves; pico capta cada um.
    Pra cada pico, o span vai do onset (onde a velocidade subiu acima de `floor` antes do
    pico) até o offset (onde caiu abaixo de `floor` depois), limitado ao ponto médio entre
    picos vizinhos pra não sobrepor. Retorna [(onset, peak, offset), ...]."""
    speed = np.asarray(speed).ravel()
    peaks, _ = find_peaks(speed, prominence=prominence, distance=distance)
    spans = []
    for i, pk in enumerate(peaks):
        lo = (peaks[i - 1] + pk) // 2 if i > 0 else 0
        hi = (pk + peaks[i + 1]) // 2 if i < len(peaks) - 1 else len(speed) - 1
        on = pk
        while on > lo and speed[on - 1] >= floor:
            on -= 1
        off = pk
        while off < hi and speed[off + 1] >= floor:
            off += 1
        spans.append((on, pk, off))
    return spans


def mode_int(arr):
    """Inteiro mais frequente num array 1-D (moda via bincount). Vazio -> -1."""
    arr = np.asarray(arr).astype(np.int64).ravel()
    if arr.size == 0:
        return -1
    return int(np.argmax(np.bincount(arr - arr.min())) + arr.min())


def viterbi(posteriors, p_self, bg_prior=None):
    """Viterbi clássico (DP) em ESPAÇO-LOG sobre uma matriz de transição grudenta.

    posteriors: (T, C) probabilidades por-frame por-classe.
    Matriz de transição: diagonal = p_self, fora da diagonal = (1 - p_self) / (C - 1).
    p_self alto -> trocar de estado custa caro -> menos mudanças -> menos flicker.
    bg_prior (opcional, (C,)): viés no estado inicial (ex.: começar no fundo).
    Retorna o caminho de estados (T,) inteiro que maximiza a verossimilhança.
    """
    T, C = posteriors.shape
    eps = 1e-8
    log_emit = np.log(posteriors + eps)                  # log das emissões (probs por-frame)

    off = (1.0 - p_self) / max(C - 1, 1)
    trans = np.full((C, C), off)
    np.fill_diagonal(trans, p_self)
    log_trans = np.log(trans + eps)                       # custo de ir do estado i -> j

    # delta[c] = melhor log-prob de um caminho que termina no estado c no frame t
    delta = log_emit[0] + (np.log(bg_prior + eps) if bg_prior is not None else 0.0)
    back = np.zeros((T, C), dtype=np.int64)               # ponteiros pra reconstruir o caminho
    for t in range(1, T):
        scores = delta[:, None] + log_trans               # (C_anterior, C_atual)
        back[t] = np.argmax(scores, axis=0)               # de onde veio cada estado atual
        delta = scores[back[t], np.arange(C)] + log_emit[t]

    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(delta))                      # melhor estado no último frame
    for t in range(T - 1, 0, -1):                         # backtracking pelos ponteiros
        path[t - 1] = back[t, path[t]]
    return path


def gate_events(speed, t_hi, t_lo, min_len=5, gap=5):
    """Histerese (Schmitt trigger) sobre um sinal 1-D pra achar spans ativos.

    Entra em "ativo" quando speed >= t_hi e só sai quando speed < t_lo. O limiar duplo
    evita chatter liga/desliga que um único limiar produziria na borda. Funde spans
    separados por <= gap frames (dip curto dentro de um mesmo golpe) e descarta spans
    com < min_len frames (jitter). Retorna [(onset, offset), ...] (índices inclusivos).
    Espelha o detect_events do boxe.py, mas parametrizado e sobre um sinal dado.
    """
    speed = np.asarray(speed).ravel()

    active, segs, start = False, [], 0
    for i, v in enumerate(speed):
        if not active and v >= t_hi:
            active, start = True, i
        elif active and v < t_lo:
            active = False
            segs.append([start, i - 1])
    if active:
        segs.append([start, len(speed) - 1])

    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:        # mesmo golpe (dip curto)
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))

    return [(on, off) for on, off in merged if off - on + 1 >= min_len]


def relabel(seq, theta_t):
    """Relabeling de segmentos curtos no estilo ASRF (rede de fronteira).

    Percorre os runs contíguos de rótulo igual; qualquer run com menos de theta_t frames
    é sobrescrito pelo rótulo do run ANTERIOR (mais longo/estável). O primeiro run mantém
    o próprio rótulo (não há anterior). Remove flicker residual que sobrou. Retorna a
    sequência (T,) limpa.
    """
    seq = np.asarray(seq).astype(np.int64).copy()
    if seq.size == 0:
        return seq

    # fronteiras dos runs: onde o rótulo muda
    bounds = np.flatnonzero(np.diff(seq) != 0) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [seq.size]])           # fim exclusivo

    for k in range(1, len(starts)):                       # primeiro run fica como está
        if ends[k] - starts[k] < theta_t:                 # run curto -> herda do anterior
            seq[starts[k]:ends[k]] = seq[starts[k] - 1]
    return seq


def decode_clip(per_frame_probs, speed, n_classes, p_self=0.9, theta_t=6,
                t_hi=0.07, t_lo=0.04, min_len=5, gap=5, conf_floor=0.30,
                bg_class=None):
    """Pipeline completo: stream de posteriors por-frame -> 1 rótulo contíguo por golpe.

    Estágios:
      1. gate_events(speed): decide QUANDO há golpe (spans ativos por histerese).
      2. viterbi(per_frame_probs, p_self): caminho de estados liso (anti-flicker).
      3. por span ativo: o rótulo = MODA dos estados Viterbi dentro do span. Se a média
         da prob-máxima no span < conf_floor, o span é REJEITADO (vira fundo) — não
         inventa golpe em trecho ambíguo.
      4. monta o array de saída (T,): fundo em todo lugar, exceto dentro dos spans
         aceitos, onde o span inteiro recebe seu rótulo único.
      5. relabel(theta_t): rede de segurança contra flicker residual.

    Fundo = bg_class se dado, senão -1. Passe probs de 3 classes (em vez de 6) pra
    suavizar no nível de TIPO em vez de classe — a API não muda.
    Retorna o array (T,) de rótulos inteiros.
    """
    per_frame_probs = np.asarray(per_frame_probs, dtype=np.float64)
    T = per_frame_probs.shape[0]
    bg = bg_class if bg_class is not None else -1

    spans = gate_events(speed, t_hi, t_lo, min_len=min_len, gap=gap)
    path = viterbi(per_frame_probs, p_self)               # estados lisos pro clip todo
    max_prob = per_frame_probs.max(axis=1)                # confiança por-frame

    out = np.full(T, bg, dtype=np.int64)
    for on, off in spans:
        if max_prob[on:off + 1].mean() < conf_floor:      # span ambíguo -> rejeita
            continue
        label = mode_int(path[on:off + 1])                # 1 rótulo p/ o golpe = moda do Viterbi
        out[on:off + 1] = label

    return relabel(out, theta_t)                          # rede de segurança final


if __name__ == "__main__":
    # Teste sintético autocontido (sem imports do projeto): dois "golpes" plantados com
    # flicker e ruído, pra checar que a saída sai contígua e sem troca de classe no meio.
    rng = np.random.default_rng(0)
    T, C = 60, 6
    probs = np.full((T, C), 0.02)                          # base quase-uniforme/baixa
    probs[:, 0] += 0.01                                    # leve viés de fundo na classe 0

    # Golpe A: frames 10-20 majoritariamente classe 2, com flicker pra classe 5.
    for f in range(10, 21):
        probs[f] = 0.05
        probs[f, 2] = 0.70
    probs[13] = 0.05; probs[13, 5] = 0.65                  # flicker no meio do golpe A
    probs[17] = 0.05; probs[17, 5] = 0.60                  # outro flicker

    # Golpe B: frames 30-45 majoritariamente classe 0.
    for f in range(30, 46):
        probs[f] = 0.05
        probs[f, 0] = 0.72

    probs += rng.uniform(0, 0.01, size=probs.shape)        # ruidinho
    probs /= probs.sum(axis=1, keepdims=True)              # re-normaliza pra somar 1

    # Velocidade: alta durante os golpes, ~0 no resto.
    speed = np.full(T, 0.01)
    speed[10:21] = 0.12
    speed[30:46] = 0.10

    labels = decode_clip(probs, speed, n_classes=C, p_self=0.92, theta_t=6,
                         t_hi=0.07, t_lo=0.04, min_len=5, gap=5,
                         conf_floor=0.30, bg_class=-1)

    print("rótulos por-frame:")
    print(labels.tolist())

    span_a = labels[10:21]
    uniq_a = set(int(x) for x in span_a)
    rest = np.concatenate([labels[:10], labels[21:30], labels[46:]])

    no_switch = len(uniq_a) == 1 and -1 not in uniq_a      # 1 classe só, sem buraco
    bg_ok = set(int(x) for x in rest) == {-1}              # repouso = fundo

    print(f"span 10-20 -> classes presentes: {sorted(uniq_a)} (esperado: uma só, != -1)")
    print(f"frames de repouso -> só fundo (-1)? {bg_ok}")
    print("PASS" if no_switch else "FAIL", "- sem troca de classe no meio do golpe")

    assert no_switch, "span do golpe A tem flicker/troca de classe (esperado contíguo)"
    assert bg_ok, "frames de repouso foram empurrados pra dentro de um golpe"
    print("PASS - todas as asserções")
