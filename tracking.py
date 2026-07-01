"""
Módulo de TRACKING: detecção de pose multi-pessoa + identidade persistente.

Concentra as três etapas que transformam um vídeo bruto em esqueletos limpos,
um por boxeador, prontos para classificação:

  1. extract_skeletons()      vídeo -> detecções de pose por frame (YOLO + ByteTrack)
  2. assign_boxers()          detecções -> 2 "slots" de identidade persistente
  3. build_dense_skeletons()  track esparso de um slot -> matriz densa sem buracos

Separado de boxe.py (que cuida de classificação e renderização) porque este é o
domínio de "seguir pessoas ao longo do tempo": IDs do ByteTrack, atribuição de
slots e preenchimento de lacunas por identidade. As funções não dependem de
nenhuma constante de boxe.py — recebem tudo por argumento e devolvem estruturas
puras (numpy / dicts).

Obs.: as variáveis de ambiente da GPU (CUDA_VISIBLE_DEVICES etc.) são definidas
por boxe.py ANTES de importar este módulo, então o torch/ultralytics já enxerga a
configuração certa quando extract_skeletons instancia o YOLO.
"""

import cv2
import numpy as np
from ultralytics import YOLO


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
