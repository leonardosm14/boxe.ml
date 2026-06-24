import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# GPU configurável via ambiente (antes era fixo em "3", causava conflito no lab).
# Definir antes de importar tensorflow/ultralytics para a visibilidade valer.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import subprocess
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO

from boxe_utils import make_window, preprocess_windows

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU error: {e}")

# Ordem TEM que casar com o softmax do modelo: o build_label_mapping ordena os
# rótulos únicos (sorted) -> {'Hook':0, 'Straight':1, 'Uppercut':2}, então o índice
# do argmax aqui só significa a classe certa nesta ordem exata. São 3 saídas, não 6.
# Hardcode rápido: a versão robusta (ler os nomes do norm_stats.npz) fica para depois.
BOXING_CLASSES = ["Hook", "Straight", "Uppercut"]

# Fallback caso norm_stats.npz não exista (mean/std por eixo, no referencial do corpo).
X_MEAN_FALLBACK = np.array([0.0, 0.0], dtype=np.float32)
X_STD_FALLBACK  = np.array([1.0, 1.0], dtype=np.float32)

WRIST = [9, 10]      # COCO: punho esquerdo e direito

# Cores por SLOT (BGR). slot 0 = Boxer 1 (verde), slot 1 = Boxer 2 (ciano). A cor
# acompanha o slot, que é uma identidade PERSISTENTE (mesmo lutador o clipe inteiro,
# fixado pelo track ID) — não o lado da tela. Caixa e rótulo do topo dividem a mesma
# cor, então dá pra seguir o mesmo lutador mesmo depois de ele cruzar de lado: o NOME
# "Boxer N" e a COR são estáveis, e juntos mostram qual lado da tela ele ocupa agora.
# O esqueleto é desenhado pelo Annotator.kpts() da ultralytics (paleta de pose
# própria); só caixas e rótulos usam estas cores.
SLOT_COLORS = [(0, 255, 0), (255, 255, 0)]

# Fração mínima de frames em que um slot precisa aparecer para contar como um
# boxeador REAL. Mata detecções espúrias: num clipe de 1 boxeador, o YOLO às vezes
# vê "2 persons" por alguns frames (reflexo, sombra, treinador) -> isso criaria um
# segundo slot fantasma e o vídeo todo viraria "Left/Right" em vez de "Boxer". Só
# slots presentes em > MIN_PRESENCE_RATIO do clipe são desenhados/rotulados.
MIN_PRESENCE_RATIO = 0.10

# Inferência por SEGMENTO DE MOVIMENTO (não frame a frame, nem por pico). O dataset é
# segmentado por golpe (start/end por linha), então a inferência também é: um golpe = uma
# região contígua onde a velocidade do punho passa de SEG_THI e só termina quando cai
# abaixo de SEG_TLO (Schmitt trigger / histerese). A histerese mantém o golpe inteiro
# (windup -> impacto -> retração) como UM segmento -> 1 classificação -> 1 rótulo segurado
# do início ao fim do golpe. Mata por construção: troca de classe no meio do golpe,
# rótulo "sumindo", múltiplos rótulos por golpe, e o golpe-em-repouso.
SEG_THI    = 0.070   # entra em "movimento" (limiar alto)
SEG_TLO    = 0.040   # sai de "movimento" (limiar baixo) — a histerese segura o golpe contíguo
SEG_GAP    = 5       # funde segmentos separados por <= GAP frames (dip de detecção no mesmo golpe)
SEG_MINLEN = 5       # descarta segmentos < MINLEN frames (jitter, não é golpe)
CONF_FLOOR = 0.30    # confiança média mínima do evento p/ exibir
SPAN_POST  = 10      # frames após o impacto que o rótulo CONTINUA (cobre extensão + retração);
                     # a velocidade do punho cai a ~0 na extensão, mas o golpe ainda é visível


def load_norm_stats():
    """Carrega média/desvio salvos no treino (norm_stats.npz). Mantém treino e
    inferência sincronizados — nunca hardcode."""
    if os.path.exists("norm_stats.npz"):
        s = np.load("norm_stats.npz")
        mean, std = np.asarray(s["mean"], np.float32), np.asarray(s["std"], np.float32)
        print(f"--> Norm stats (corpo, por eixo): mean={mean} std={std} (norm_stats.npz)")
        return mean, std
    print("--> Aviso: norm_stats.npz não encontrado; usando fallback")
    return X_MEAN_FALLBACK, X_STD_FALLBACK


def ensure_25fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if abs(fps - 25.0) < 0.1:
        print(f"--> Video already at {fps:.2f} FPS")
        return video_path

    output_dir = "25fps"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(video_path))

    if os.path.exists(output_path):
        print(f"--> 25 FPS version already exists: {output_path}")
        return output_path

    print(f"--> Converting {fps:.2f} FPS video to 25 FPS...")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "fps=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", output_path,
    ]
    subprocess.run(cmd, check=True)
    print(f"--> Saved 25 FPS video: {output_path}")
    return output_path


def extract_skeletons(video_path):
    # EXTRAÇÃO MULTI-PESSOA POR FRAME COM TRACK ID OPCIONAL
    # -----------------------------------------------------
    # Usa `.track()` do YOLO (ByteTrack) e guarda TODAS as pessoas de CADA frame.
    # O track ID (`r.boxes.id`) é salvo quando disponível como sinal de identidade
    # para a atribuição de slots, mas NÃO é obrigatório: frames onde o tracker não
    # conseguiu atribuir IDs continuam aceitos (detecções sem ID recebem tid=None).
    # Isso garante a mesma cobertura de detecção do código original.
    #
    # Retorna:
    #   frames_dets  lista de tamanho T; frames_dets[f] = lista de detecções desse
    #                frame, cada uma um dict:
    #                  "coords" (17,2) x,y normalizados por largura/altura
    #                  "conf"   (17,)  confiança por junta
    #                  "box"    (4,)   caixa x1,y1,x2,y2 em PIXELS
    #                  "tid"    int|None  track ID do YOLO (None se indisponível)
    #   total_frames, video_width, video_height
    print(f"--> [1/3] Running YOLO-Pose: {video_path}")
    model_yolo = YOLO("yolov8m-pose.pt")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames_dets = [[] for _ in range(total_frames)]

    results = model_yolo.track(source=video_path, stream=True, device="cuda", conf=0.3)

    for frame_idx, r in enumerate(results):
        if frame_idx >= total_frames:
            break
        if r.boxes is None or r.keypoints is None:
            continue

        kp_all = r.keypoints.data.cpu().numpy()   # (N, 17, 3): x, y, conf
        xyxy   = r.boxes.xyxy.cpu().numpy()       # (N, 4) caixa em pixels

        # IDs podem não estar disponíveis em todos os frames (primeiro frame,
        # reset do tracker, etc). Quando ausentes, tid fica None.
        ids = None
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)

        for i_det, (kp, box) in enumerate(zip(kp_all, xyxy)):
            if kp.shape[0] != 17:
                continue
            if kp[:, 2].mean() < 0.5:
                continue

            coords_xy = kp[:, :2].copy()
            coords_xy[:, 0] /= video_width
            coords_xy[:, 1] /= video_height

            tid = int(ids[i_det]) if ids is not None else None

            frames_dets[frame_idx].append({
                "coords": coords_xy,
                "conf":   kp[:, 2].copy(),
                "box":    box.copy(),
                "tid":    tid,
            })

    n_dets = sum(len(d) for d in frames_dets)
    print(f"--> [2/3] Skeleton extraction complete. {n_dets} detecções em {total_frames} frames.")
    return frames_dets, total_frames, video_width, video_height


def preprocess_window(window, mean, std):
    """Normaliza UMA janela (25,17,2) pela pipeline compartilhada do boxe_utils
    (referencial do corpo -> padronização por eixo -> vel/acc). Treino == inferência."""
    return preprocess_windows(window[None], mean, std)


def wrist_speed_signal(skeletons):
    """Velocidade do punho dominante por frame (máx dos dois punhos), suavizada (5-tap)."""
    d = np.diff(skeletons[:, WRIST, :], axis=0)
    spd = np.linalg.norm(d, axis=2).max(axis=1)
    spd = np.concatenate([[0.0], spd])
    k = 5
    spd = np.convolve(np.pad(spd, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return spd


def detect_events(skeletons):
    """Segmenta por MOVIMENTO contínuo do punho (Schmitt trigger): entra em 'movimento'
    quando a velocidade passa de SEG_THI e só sai quando cai abaixo de SEG_TLO. A histerese
    captura o golpe inteiro (windup -> impacto -> retração) como UM segmento contíguo, então
    1 rótulo cobre o golpe todo sem cortes. Funde dips curtos (mesmo golpe) e descarta
    segmentos minúsculos (jitter). Retorna [(onset, peak, offset), ...] e o sinal."""
    spd = wrist_speed_signal(skeletons)

    active, segs, start = False, [], 0
    for i, v in enumerate(spd):
        if not active and v >= SEG_THI:
            active, start = True, i
        elif active and v < SEG_TLO:
            active = False
            segs.append([start, i - 1])
    if active:
        segs.append([start, len(spd) - 1])

    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= SEG_GAP:   # mesmo golpe (dip curto)
            merged[-1][1] = s[1]
        else:
            merged.append(list(s))

    events = []
    for on, off in merged:
        if off - on + 1 < SEG_MINLEN:                    # jitter, não é golpe
            continue
        peak = on + int(np.argmax(spd[on:off + 1]))      # frame de impacto (máx velocidade)
        events.append((on, peak, off))
    return events, spd


def assign_boxers(frames_dets, total_frames, video_width):
    # IDENTIDADE PERSISTENTE POR TRACK ID (com POSIÇÃO só de FALLBACK)
    # ---------------------------------------------------------------
    # Monta DOIS "slots" de boxeador. A identidade NÃO é por posição: cada slot
    # PERSISTE o mesmo lutador pelo track ID do ByteTrack, mesmo depois de eles
    # cruzarem de lado. A posição inicial só decide quem entra em qual slot na
    # primeira vez; daí em diante o ID manda. A estratégia é em DUAS FASES por frame:
    #
    # FASE 1 — MATCH POR ID: se uma detecção tem um track ID (`tid`) que já
    # foi associado a um slot em frames anteriores, ela vai direto para esse
    # slot — independentemente de qual lado da tela ela está agora. O ByteTrack
    # mantém IDs estáveis mesmo quando os lutadores trocam de lado, então o slot
    # segura o MESMO lutador o clipe inteiro (sem flickering de tags).
    #
    # FASE 2 — FALLBACK POR POSIÇÃO: detecções sem ID (tid=None) ou com ID
    # desconhecido (novo ID após oclusão) são atribuídas por posição, exatamente
    # como o código original fazia. Isso garante que NENHUMA detecção é perdida
    # por falta de ID — a cobertura é igual à versão anterior.
    #
    # Quando um novo ID é atribuído a um slot (via posição), o slot "lembra"
    # esse ID para os frames seguintes — absorve fragmentações do ByteTrack
    # naturalmente, sem precisar de merge explícito.
    slots = [
        {
            "coords":  np.zeros((total_frames, 17, 2)),
            "conf":    np.zeros((total_frames, 17)),
            "boxes":   np.zeros((total_frames, 4)),
            "present": np.zeros(total_frames, dtype=bool),
        }
        for _ in range(2)
    ]
    last_cx  = [None, None]   # último center_x conhecido de cada slot
    slot_tid = [None, None]   # track ID atualmente "dono" de cada slot

    def center_x(det):
        return (det["box"][0] + det["box"][2]) / 2.0

    def put(slot, det, f):
        slots[slot]["coords"][f]  = det["coords"]
        slots[slot]["conf"][f]    = det["conf"]
        slots[slot]["boxes"][f]   = det["box"]
        slots[slot]["present"][f] = True
        last_cx[slot] = center_x(det)
        # Atualiza o "dono" do slot: se a detecção tem ID, esse ID vira o novo
        # dono. Se um boxeador sai com ID 3 e volta com ID 7, o slot absorve
        # o ID 7 automaticamente via atribuição por posição + esta linha.
        tid = det.get("tid")
        if tid is not None:
            slot_tid[slot] = tid

    for f in range(total_frames):
        dets = frames_dets[f]
        if not dets:
            continue

        if len(dets) >= 2:
            # --- FASE 1: match por track ID ---
            assigned = [False, False]   # quais slots foram preenchidos
            used = set()                # quais detecções foram consumidas

            for di, d in enumerate(dets):
                tid = d.get("tid")
                if tid is None:
                    continue
                for si in range(2):
                    if slot_tid[si] == tid and not assigned[si]:
                        put(si, d, f)
                        assigned[si] = True
                        used.add(di)
                        break

            # --- FASE 2: fallback por posição para slots não preenchidos ---
            remaining = sorted(
                [d for di, d in enumerate(dets) if di not in used],
                key=center_x
            )

            if not assigned[0] and not assigned[1] and len(remaining) >= 2:
                # Nenhum slot matchou por ID — atribuição pura por posição
                put(0, remaining[0],  f)
                put(1, remaining[-1], f)
            elif not assigned[0] and remaining:
                # Slot 0 livre (não matchou por ID) — pega o mais à esquerda dos restantes
                put(0, remaining[0], f)
            elif not assigned[1] and remaining:
                # Slot 1 livre (não matchou por ID) — pega o mais à direita dos restantes
                put(1, remaining[-1], f)

        else:
            # Só 1 detecção: tentar match por ID, senão continuidade por posição
            d = dets[0]
            tid = d.get("tid")

            # Fase 1: ID match direto
            matched = False
            if tid is not None:
                for si in range(2):
                    if slot_tid[si] == tid:
                        put(si, d, f)
                        matched = True
                        break

            if not matched:
                # Fase 2: continuidade por posição (idêntica ao código original)
                x = center_x(d)
                d0 = abs(x - last_cx[0]) if last_cx[0] is not None else float("inf")
                d1 = abs(x - last_cx[1]) if last_cx[1] is not None else float("inf")
                if d0 == float("inf") and d1 == float("inf"):
                    slot = 0 if x < video_width / 2 else 1
                else:
                    slot = 0 if d0 <= d1 else 1
                put(slot, d, f)

    p0 = int(slots[0]["present"].sum())
    p1 = int(slots[1]["present"].sum())
    print(f"--> Identidade persistente (track ID + posição de fallback): "
          f"Boxer 1 presente {p0}/{total_frames} | Boxer 2 presente {p1}/{total_frames}")
    return slots


def build_dense_skeletons(track, total_frames):
    # MUDANÇA - MATRIZ DENSA POR BOXEADOR COM PREENCHIMENTO DE LACUNAS
    # ---------------------------------------------------------------
    # O classificador precisa de UM esqueleto por frame, sem buracos, porque
    # make_window() (boxe_utils) recorta uma tira de frames CONTÍGUOS. O track de
    # um boxeador é esparso: o id dele falta em alguns frames (oclusão, blur, borda
    # do frame, ou reprovado no filtro de confiança). Um buraco = linha de zeros =
    # o punho "teleporta" para (0,0) e volta = movimento explosivo falso = predição
    # lixo.
    #
    # Esta função transforma o track esparso na matriz densa (total_frames, 17, 2)
    # que o código de 1 boxeador antigo produzia. A regra de preenchimento é
    # EXATAMENTE a mesma que o extract_skeletons ANTIGO usava no nível de uma
    # pessoa (boxe.py:116-117 da versão antiga): "se este frame está vazio, copia o
    # frame anterior pra frente". Isso mantém a matriz contínua para que
    # velocidade/aceleração em preprocess_windows() não disparem numa linha de zero.
    #
    # Chamada em __main__, uma vez por boxeador selecionado: o resultado denso é
    # depois suavizado (mesma média móvel de 5 frames de antes, inline no __main__)
    # e passado a classify_events(). Separamos isto da extração porque, com 2
    # boxeadores, "copiar o frame anterior" tem que ser POR PESSOA - o buraco do
    # boxeador A copia a última pose do A, não a de outro.
    coords  = track["coords"]
    present = track["present"]

    dense = np.zeros((total_frames, 17, 2))
    for idx in range(total_frames):
        if present[idx]:
            dense[idx] = coords[idx]
        elif idx > 0:
            dense[idx] = dense[idx - 1]   # carrega a última pose conhecida pra frente
    return dense


def classify_events(skeletons, model, mean, std):
    # MUDANÇA - CLASSIFICAÇÃO EXTRAÍDA PARA RODAR POR BOXEADOR
    # -------------------------------------------------------
    # Isto é REORGANIZAÇÃO de código, não algoritmo novo. A lógica de detectar e
    # classificar golpes vivia GRUDADA dentro de save_video_with_predictions, que
    # assumia 1 boxeador. Aqui ela vira função própria que recebe o esqueleto denso
    # de UM boxeador e devolve `frame_label` (lista de tamanho T, cada item
    # (class_idx, conf) ou None). Assim dá pra CHAMAR uma vez por boxeador.
    #
    # A lógica é copiada igual da versão antiga:
    #   1. detect_events() acha rajadas de velocidade do punho (1 rajada = 1 golpe);
    #   2. cada golpe é classificado UMA vez = média do softmax de 5 janelas
    #      ancoradas no pico (instante de impacto, fase que o modelo viu no treino);
    #   3. CONF_FLOOR rejeita golpes ambíguos;
    #   4. o rótulo é segurado do onset até o fim do follow-through, sem invadir o
    #      próximo golpe.
    # Lacunas viram pose congelada (build_dense_skeletons) -> velocidade do punho ~0
    # -> nenhum evento falso ali.
    events, spd = detect_events(skeletons)

    X_batch, owner = [], []
    for ei, (on, pk, off) in enumerate(events):
        for p in range(pk - 2, pk + 3):
            window = make_window(skeletons, p)
            if window is None:
                continue
            X_batch.append(preprocess_window(window, mean, std))
            owner.append(ei)

    n = len(skeletons)
    frame_label = [None] * n
    n_shown = 0
    if X_batch:
        preds = model.predict(np.concatenate(X_batch), batch_size=512, verbose=0)
        owner = np.array(owner)
        for ei, (on, pk, off) in enumerate(events):
            sel = preds[owner == ei]
            if len(sel) == 0:
                continue
            avg = sel.mean(axis=0)                       # média das probs = 1 decisão por golpe
            ci, conf = int(np.argmax(avg)), float(avg.max())
            if conf < CONF_FLOOR:                        # rejeita evento ambíguo
                continue
            n_shown += 1
            # rótulo do snap (onset) até o fim do follow-through (impacto + retração),
            # sem invadir o próximo golpe -> o rótulo fica até o golpe acabar
            end = max(off, pk + SPAN_POST)
            if ei + 1 < len(events):
                end = min(end, events[ei + 1][0] - 1)
            end = min(end, n - 1)
            for f in range(on, end + 1):
                frame_label[f] = (ci, conf)               # 1 rótulo fixo no span do golpe
    print(f"--> {len(events)} segmentos | {n_shown} eventos exibidos após rejeição")
    return frame_label


def _convert_h264(temp_path, final_path):
    # Recodifica o mp4v temporário para H.264 (mesmo bloco ffmpeg de antes).
    # Fatorado porque agora geramos DOIS vídeos (caixa e caixa+esqueleto) e cada um
    # precisa da conversão.
    print(f"--> Converting codec to H.264: {final_path}")
    try:
        cmd = [
            'ffmpeg', '-y', '-i', temp_path,
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', final_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"--> Final video saved: {final_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")


def render_videos(video_path, out_box_path, out_pose_path, boxers, video_width, video_height):
    # MUDANÇA - DOIS VÍDEOS DE SAÍDA COM RÓTULO POR BOXEADOR
    # -----------------------------------------------------
    # Substitui a parte de desenho do save_video_with_predictions antigo (que fazia
    # 1 barra "STATUS" no topo para 1 boxeador). Agora escreve DOIS arquivos a
    # partir do MESMO frame de entrada:
    #   out_box  = só caixas delimitadoras (sem esqueleto)
    #   out_pose = caixas + esqueleto COCO desenhado por boxeador
    # Os dois mostram a mesma barra de topo com os dois rótulos lado a lado:
    #   "Boxer 1: Straight (98.0%)"
    #   "Boxer 2: Hook (66.2%)"
    #
    # `boxers` é uma lista de 1 ou 2 entradas vinda de assign_boxers: boxers[0] =
    # slot 0, boxers[1] = slot 1. O ÍNDICE do boxeador É o slot, e o slot é uma
    # IDENTIDADE PERSISTENTE (mesmo lutador o clipe inteiro, fixado pelo track ID do
    # ByteTrack) — NÃO é a posição na tela. Não há re-ordenação por x aqui. Cada entrada:
    #   frame_label (T) (class_idx,conf)|None  -> rótulo do golpe por frame
    #   coords      (T,17,2) normalizados      -> desenho do esqueleto (denormaliza p/ px)
    #   conf        (T,17)                      -> Annotator.kpts pula juntas fracas
    #   boxes       (T,4) pixels                -> caixa delimitadora
    #   present     (T,) bool                  -> só desenha caixa/esqueleto se visto
    #
    # O NOME na tela é estável: "Boxer 1"/"Boxer 2", colado na identidade do slot — não
    # muda de dono mesmo quando os lutadores se cruzam. Quem mostra o lado ATUAL da tela
    # é a COR da caixa (SLOT_COLORS): verde e ciano acompanham o slot, não o nome.
    from ultralytics.utils.plotting import Annotator

    # Nomes ESTÁVEIS de identidade: slot 0 = "Boxer 1", slot 1 = "Boxer 2". Persistem
    # com o lutador (track ID), por isso não usamos mais "Left"/"Right" no nome.
    slot_names = ("Boxer 1", "Boxer 2")

    # "Boxer 1"/"Boxer 2" só faz sentido com DOIS boxeadores. slot_active marca quais
    # slots são boxeadores REAIS = presentes em mais de MIN_PRESENCE_RATIO do clipe (não
    # só "em algum frame"). Esse limiar descarta o slot fantasma criado por
    # detecções espúrias de "2 persons" em poucos frames - sem ele, um clipe de 1
    # boxeador viraria "Boxer 1/Boxer 2" por causa de um reflexo de 5 frames.
    # two_boxers = ambos os slots são reais. Com um só, o rótulo vira "Boxer" e o
    # outro slot não é desenhado nem rotulado. active_idx = qual slot é o único real.
    total = len(boxers[0]["present"]) if boxers else 0
    slot_active = [
        bool(boxers[bi]["present"].sum() > MIN_PRESENCE_RATIO * total)
        for bi in range(len(boxers))
    ]
    two_boxers  = sum(slot_active) >= 2
    active_idx  = next((bi for bi in range(len(boxers)) if slot_active[bi]), 0)

    def disp_name(bi):
        # nome curto da caixa: identidade estável "Boxer 1"/"Boxer 2" com 2
        # boxeadores, senão só "Boxer" (sem número no caso de 1 lutador).
        return slot_names[bi] if two_boxers else "Boxer"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_box, tmp_pose = "temp_box.mp4", "temp_pose.mp4"
    out_box  = cv2.VideoWriter(tmp_box,  fourcc, fps, (video_width, video_height))
    out_pose = cv2.VideoWriter(tmp_pose, fourcc, fps, (video_width, video_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # quem está presente neste frame (índice do boxeador = slot de identidade fixo).
        # slot_active filtra o slot fantasma: um slot abaixo do limiar de presença
        # não é desenhado nem rotulado em frame nenhum, mesmo nos poucos frames
        # espúrios em que apareceu.
        present = [
            bi for bi in range(len(boxers))
            if slot_active[bi]
            and frame_idx < len(boxers[bi]["present"]) and boxers[bi]["present"][frame_idx]
        ]

        # texto de cada slot (ausente ou sem golpe -> "..."). slot_text[0]=Boxer 1,
        # slot_text[1]=Boxer 2 (identidade, não lado da tela).
        slot_text = ["...", "..."]
        for bi in present:
            fl = boxers[bi]["frame_label"]
            lab = fl[frame_idx] if frame_idx < len(fl) else None
            if lab is not None:
                ci, conf = lab
                slot_text[bi] = f"{BOXING_CLASSES[ci]} ({conf * 100:.1f}%)"

        # duas cópias do frame: caixa-só e caixa+esqueleto
        fb = frame.copy()
        fp = frame.copy()

        # esqueleto SÓ no vídeo pose, via Annotator.kpts (paleta de pose da ultralytics).
        # Monta keypoints em pixels (17,3) = x*W, y*H, conf.
        ann = Annotator(fp)
        for bi in present:
            k = np.zeros((17, 3), dtype=np.float32)
            k[:, 0] = boxers[bi]["coords"][frame_idx][:, 0] * video_width
            k[:, 1] = boxers[bi]["coords"][frame_idx][:, 1] * video_height
            k[:, 2] = boxers[bi]["conf"][frame_idx]
            ann.kpts(k, shape=(video_height, video_width), radius=4, kpt_line=True)
        fp = ann.result()

        # caixas + rótulos: idênticos nos dois vídeos
        for img in (fb, fp):
            for bi in present:
                color = SLOT_COLORS[bi]            # bi = slot/identidade (0=Boxer 1 verde, 1=Boxer 2 ciano)
                x1, y1, x2, y2 = boxers[bi]["boxes"][frame_idx].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, disp_name(bi), (x1, max(y1 - 8, 75)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # barra de topo. Com 2 boxeadores: "Boxer 1"/"Boxer 2" lado a lado (nome
            # de identidade estável; a COR é que indica o lado atual da tela). Com 1
            # só: um único "Boxer: ..." (sem número). Slot vazio não recebe rótulo.
            cv2.rectangle(img, (0, 0), (video_width, 60), (0, 0, 0), -1)
            if two_boxers:
                cv2.putText(img, f"Boxer 1: {slot_text[0]}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, SLOT_COLORS[0], 2, cv2.LINE_AA)
                cv2.putText(img, f"Boxer 2: {slot_text[1]}", (video_width // 2 + 20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, SLOT_COLORS[1], 2, cv2.LINE_AA)
            else:
                cv2.putText(img, f"Boxer: {slot_text[active_idx]}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, SLOT_COLORS[active_idx], 2, cv2.LINE_AA)

            frame_text = f"Frame: {frame_idx}"
            (tw, _), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.putText(img, frame_text, (video_width - tw - 20, video_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out_box.write(fb)
        out_pose.write(fp)
        frame_idx += 1

    cap.release()
    out_box.release()
    out_pose.release()

    _convert_h264(tmp_box, out_box_path)
    _convert_h264(tmp_pose, out_pose_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boxing Punch Classifier")
    parser.add_argument("-v", "--video",  required=True,               help="Input video path")
    parser.add_argument("-m", "--model",  default="modelo_boxe.keras", help="Model path")
    parser.add_argument("-o", "--output", default=None,                help="Custom output path")
    parser.add_argument("--clear-cache",  action="store_true",         help="Force skeleton cache reset")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: video '{args.video}' not found.")
        exit(1)

    video_path = ensure_25fps(args.video)

    if args.output is None:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(args.video))
    else:
        output_path = args.output

    print("--> Loading TensorFlow model...")
    loaded_model = load_model(args.model)

    # Sanidade: o nº de saídas do softmax tem que bater com BOXING_CLASSES. Se o
    # modelo for retreinado com outro conjunto de classes (ex.: voltar pras 6
    # antigas), os rótulos ficariam todos trocados em silêncio — aqui isso grita.
    n_out = loaded_model.output_shape[-1]
    if n_out != len(BOXING_CLASSES):
        print(f"--> AVISO: modelo tem {n_out} classes de saída, mas BOXING_CLASSES "
              f"tem {len(BOXING_CLASSES)} ({BOXING_CLASSES}). Os rótulos vão sair "
              f"errados — ajuste BOXING_CLASSES para casar com o modelo.")

    mean, std = load_norm_stats()

    # Cache: usa prefixo "tdets_" (tracked detections) para invalidar caches
    # antigos ("dets_" sem tid e "tracks_" com formato diferente).
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = f"tdets_{video_base_name}.npy"

    if args.clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print("--> Previous cache cleared.")

    if os.path.exists(cache_path):
        print(f"--> Cache found: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True).item()
        frames_dets  = cached["frames_dets"]
        total_frames = cached["total_frames"]
        video_width  = cached["video_width"]
        video_height = cached["video_height"]
    else:
        frames_dets, total_frames, video_width, video_height = extract_skeletons(video_path)
        np.save(cache_path, np.array(
            {"frames_dets": frames_dets, "total_frames": total_frames,
             "video_width": video_width, "video_height": video_height},
            dtype=object,
        ))

    # Identidade persistente por track ID (ByteTrack); posição só garante cobertura
    slots = assign_boxers(frames_dets, total_frames, video_width)

    # Este loop roda UMA VEZ POR SLOT (Boxer 1, depois Boxer 2). Cada slot é uma
    # identidade fixa (mesmo lutador o clipe todo), não um lado da tela. Tudo dentro
    # dele - inclusive a suavização 5-tap - executa por boxeador.
    boxers = []
    for i, slot in enumerate(slots):
        print(f"--> Boxer {i + 1}:")

        # 1) esqueleto DENSO (lacunas preenchidas) para a classificação
        dense = build_dense_skeletons(slot, total_frames)

        # 2) suavização temporal 5-tap (mesma conta do código original)
        window_size = 5
        for joint in range(17):
            for coord in range(2):
                signal = dense[:, joint, coord]
                padded = np.pad(signal, (window_size // 2, window_size // 2), mode="edge")
                smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode="valid")
                dense[:, joint, coord] = smoothed

        # 3) classifica os golpes deste boxeador (detect_events + média no pico)
        frame_label = classify_events(dense, loaded_model, mean, std)

        # 4) dados para o render. coords/conf/boxes/present vêm do slot CRU (não
        #    preenchido) porque o desenho só mostra frames realmente vistos.
        boxers.append({
            "frame_label": frame_label,
            "coords":  slot["coords"],
            "conf":    slot["conf"],
            "boxes":   slot["boxes"],
            "present": slot["present"],
        })

    # Dois arquivos de saída derivados de output_path: <base>_box e <base>_pose.
    base, ext = os.path.splitext(output_path)
    ext = ext or ".mp4"
    out_box_path  = f"{base}_box{ext}"
    out_pose_path = f"{base}_pose{ext}"

    print("--> Renderizando 2 vídeos (caixa e caixa+esqueleto)...")
    render_videos(video_path, out_box_path, out_pose_path, boxers, video_width, video_height)
