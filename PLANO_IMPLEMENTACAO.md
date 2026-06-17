# Plano de Implementação — Reconhecimento robusto de golpes (TIPO + MÃO, smoothness e generalização)

> Documento de plano. Escrito no mesmo nível do código do Leo (numpy/Keras functional, comentários
> em PT, simples e legível), baseado em pesquisa verificada (papers + docs oficiais) e nos números
> que **medimos** (não em achismo). Cada tarefa tem **arquivo → o que fazer → como medir → resultado
> esperado**. As decisões grandes vêm com a evidência que as sustenta.

**Goal:** Entregar um reconhecedor de golpes que (a) acerta os golpes parecidos — em especial a MÃO
(jab/cross, lead/rear hook, lead/rear uppercut) — e (b) mostra cada golpe de forma **contínua e
suave**, do início do movimento até o fim, sem piscar para o nada nem trocar de classe no meio.

**Architecture:** Mantém a pipeline correta do Leo (YOLOv8-Pose → janelas de 25 → modelo temporal →
classe), mas ataca os dois gargalos medidos com mudanças cirúrgicas: (1) **features view-invariantes**
(ângulos internos do corpo + direções, não só coordenadas) para o TIPO generalizar; (2) **decomposição
TIPO×MÃO** com a MÃO resolvida por **geometria de stance por-clipe** validada empiricamente antes de
confiar nela; (3) **decoder temporal** (gating por histerese + Viterbi + relabel) para a suavidade;
(4) **rigor de avaliação** (LOVO + CI + métricas segmentais + ECE) como fonte de verdade.

**Tech Stack:** Python 3.11, TensorFlow/Keras 2.21, numpy, scipy 1.13, scikit-learn, pandas, OpenCV,
Ultralytics YOLOv8m-pose. Execução nas H100 da UFSC via `jp run` (sem dependências novas — Viterbi é
~20 linhas de numpy; `librosa` **não** está no remoto e **não** será adicionado).

## Global Constraints

- **2D apenas, 17 keypoints COCO.** Sem profundidade, sem 3D-lifting, sem RGB no momento de classificar.
- **Pipeline treino == inferência.** Toda transformação de feature vive em `boxe_utils.py` e é usada
  igual no `train.py` e no `boxe.py`. Nunca hardcode de estatística (usar sidecar `norm_stats.npz`).
- **Métrica primária = CROSS-VIDEO / LOVO**, nunca same-source. Same-source (0.87) é a armadilha que
  premia o overfit de câmera. Todo hiperparâmetro (tau, p_self, theta_t, limiares) é escolhido no
  cross-video/LOVO.
- **Nada entra como "melhora" sem CI.** Média ± desvio + bootstrap, comparado com pisos
  (majoritário ~0.27 no global, aleatório 0.167). Decisão dentro do CI = ruído, não conta.
- **Eficiência de medição:** ablações/screening rodam no **split cross-video único (V3,V4), 3 seeds**
  (rápido). **LOVO completo (10 folds, 5 seeds)** só no baseline (T0), nas decisões de GATE e na seleção
  final (T8). Isso evita explosão `10×5×variantes` mantendo a decisão final no padrão-ouro.
- **Logits, não probabilidades, no treino.** Para o logit-adjustment funcionar, o `Dense` final **não
  tem ativação** (sai logits); a loss usa `from_logits=True` e soma `tau·log(pi)` aos logits; a
  **inferência aplica softmax à parte** (`ops.softmax(logits)`). Treino==inferência se a softmax de
  inferência for sempre a mesma (padrão `LogisticEndpoint` da doc oficial Keras).
- **Estilo do código:** Keras functional API, numpy puro, comentários curtos em PT, sem over-engineering.
  Arquivos de experimento começam com `_` (já estão no `.jpignore`, não sobem pro git nem pro vlab por
  `jp push`; rodam por `jp run`).
- **Sem modelos pesados.** Proibido CTR-GCN/HD-GCN/InfoGCN/Hyperformer/transformers grandes — a 5.5k
  clips eles overfitam e os ganhos de NTU (56k–114k clips) não transferem. 0.87 in-distribution já prova
  que **capacidade não é o gargalo**; o gargalo é representação/generalização/mão.
- **GPU:** ler a GPU mais vazia em runtime (`nvidia-smi`) e setar `CUDA_VISIBLE_DEVICES` antes de
  importar TF. Nunca fixar GPU.

---

## Estado atual (medido) e diagnóstico

Pipeline atual (branch `feat/body-frame-event-inference`): YOLOv8m-pose → 17 COCO → janela de 25
(≈10 frames reais no início + zero-pad) → norm relativa ao corpo (recentro quadril + escala ombro) →
padronização por eixo → +velocidade+aceleração (102 features) → BiLSTM(64)+MultiHeadAttention → softmax 6.

| Métrica | Valor medido |
|---|---|
| TEST (V5,V9 same-source) | **0.826–0.871** |
| CROSS-VIDEO (V3,V4 held-out) | **0.31** (3-classes TIPO: **0.49**) |
| Rear Hook recall | **~0.02–0.14** (133 amostras = 5.3% do treino) |
| Geometria da mão (lead/rear, por janela) | TRAIN 0.81 · CROSS 0.70 · TEST 0.71 |
| Híbrido (tipo+geometria) cross-video | 0.348 (vs direto 0.311) — geometria a 0.75 quase não ajuda |
| Oracle (tipo + mão verdadeira) cross-video | **0.493** (teto se a mão fosse perfeita) |

**Distribuição (medida, V1–V10):** Cross 25.2% · Jab 23.6% · Lead Hook 19% · Lead Upp 12.6% ·
Rear Upp 12.5% · **Rear Hook 7.2%**. LEAD 55% / REAR 45% (equilibrado). TIPO: reto 49% / hook 26% /
uppercut 25%. **Perna/pé visível em ~100% das janelas** (V1/V3/V5) → stance por geometria é viável; o
gargalo é a **regra**, não oclusão.

**Os dois problemas que o plano ataca (e a evidência):**
1. **Confusão de golpes parecidos = a MÃO.** Jab↔Cross, Lead↔Rear Hook/Uppercut só diferem pela mão
   (lead = mão do mesmo lado do pé da frente). Numa janela isolada e em 2D, depois da norm pelo corpo,
   uma stance ortodoxa e uma canhota ficam quase espelhadas → o softmax 6-classes não decide a mão e
   colapsa cross-video (0.31). O TIPO (3-classes) é a parte recuperável (0.49). **Metade do gap OOD é
   mão; a outra metade é o TIPO não generalizar.**
2. **Piscar / trocar de classe no meio do golpe.** É o problema clássico de *over-segmentation* de um
   classificador deslizante. A inferência por evento atual já ajuda, mas (a) a histerese mede velocidade
   em **pixels crus** (não invariante a ângulo), e (b) o rótulo vem da média de 5 janelas no pico (pode
   pegar a janela mais ambígua). Precisa de um decoder temporal de verdade.

---

## Estratégia consolidada (síntese da pesquisa — só o que foi verificado e cabe no nosso caso)

A pesquisa (7 dimensões, com verificação adversarial) convergiu nestes levers. Entre `[ ]` a confiança
e a fonte verificada. Itens marcados **GATE** só entram se passarem numa medição.

1. **Features view-invariantes (ângulos internos + direções), somadas às coordenadas.** Ângulos de
   cotovelo, linha de ombro, linha de quadril, torção (ombro−quadril), razão de direção da velocidade
   do punho (vertical/horizontal) — invariantes a translação/escala/rotação-no-plano. Atacam a metade
   "TIPO não generaliza" do gap (a que ninguém atacou ainda). `[alta — biomech: Frontiers 2024 PMC11466798
   cross/uppercut shoulder angle 105.5°/64.2°; ood: PMC11623218 angle features; 2s-AGCN bone stream]`.
   **Honesto:** ângulos removem só variação no plano; **não** removem mudança de ponto de vista
   fora-do-plano (frontal↔lateral). Esperar ganho parcial, não fechar o gap.
2. **De-roll no plano (nivelar a linha de quadril).** Rotação bem-posta em 2D que tira o roll/tilt de
   câmera. **Não** tentar frontalização total (mal-posta em 2D, destrói o arco do hook). `[alta — review 2025
   2506.00915 alerta que alinhamento de plano corporal perde movimento em 2D]`.
3. **Mirror augmentation com troca de rótulo — JÁ EXISTE no `train.py`** (`flip_body` + `SWAP_L`). Para
   o modelo de TIPO (3-classes) é ainda mais limpo (hook espelhado continua hook, sem troca). Ação:
   manter, confirmar que dispara e recalcula vel/acc depois do flip. `[verificado — a pesquisa
   re-propôs o que já temos]`.
4. **Logit-adjusted softmax (balanced softmax) no lugar do `class_weight='balanced'`.** Correção
   Bayes-ótima de prior para a Rear Hook faminta: `loss = CE(z + tau·log(pi), y)`, `pi` = freq de treino,
   `tau∈{0.5,1,1.5}` escolhido no cross-video; inferência com logits puros. **Não** empilhar com
   class_weight (corrige duas vezes). `[alta — Menon et al. ICLR 2021; Cortes/Mohri 2026 2512.23947]`.
5. **Pose-noise augmentation (robustez estilo PoseC3D, barata).** Jitter gaussiano por junta
   (σ≈2–3% da largura de ombro), dropout de junta (p≈0.1), rotação no plano pequena (±10–15°, tunar),
   escala (0.9–1.1), warp temporal — **antes** de recalcular vel/acc, **mild** (LSTM é sensível).
   `[alta — PoseC3D CVPR2022 robustez a ruído de keypoint; mild p/ LSTM]`.
6. **Máscara do padding (bug latente corrigido).** Hoje a padronização por eixo transforma o zero-pad
   em `−mean/std ≠ 0`, e o `np.diff` cria velocidade espúria na fronteira real/pad → o BiLSTM come lixo.
   Corrigir: manter o pad em 0 nas features e usar `Masking(0.0)` + `GlobalAveragePooling1D` que respeita
   máscara. `[verificado via context7 keras_io — Masking + GAP1D mask-aware existem no Keras 3]`.
7. **Decomposição TIPO(3)×MÃO(2). GATE na geometria.** O modelo aprende só o TIPO (view-invariante,
   0.49 cross-video). A MÃO sai de geometria de stance **por-clipe**. **Gate medido antes de confiar:**
   acurácia da mão no cross-video precisa subir dos atuais ~0.75 para **≥0.85** (idealmente ~0.90) com
   as melhorias (stance por-clipe, voto ponderado por visibilidade, "frente" pela orientação do torso e
   não pela extensão do punho, perna inteira). Se ficar ~0.75, a geometria entra **só adaptativa** (corrige
   a mão **apenas** quando o modelo 6-classes está incerto entre as duas mãos do mesmo TIPO; **nunca**
   sobrescreve in-distribution, onde a geometria piora 0.871→0.612). `[GATE — decompose verifier + nosso
   IDEIA_STANCE medido]`.
8. **Decoder temporal para suavidade (alto valor, baixo risco).** Duas etapas: (A) **gating** por
   histerese na velocidade do punho **normalizada pelo corpo** (não pixel cru); (B) **labeling**:
   expandir o softmax por-janela para por-frame, rodar **Viterbi com matriz pegajosa** (numpy puro,
   `p_self` tunado 0.85–0.97) sobre as posteriors, pegar a **moda da trilha** dentro de cada span ativo =
   1 rótulo contíguo. Depois **relabel** ASRF (apaga sub-segmentos < `theta_t`≈5–8 frames). **Honesto:**
   o decoder conserta o PISCAR, não o ACERTO — Viterbi sobre um stream 0.31 vira "errado mas suave".
   Por isso vem com *fallback de TIPO* (suaviza no nível do TIPO quando a mão é indecidível) e estado de
   background. `[alta — MS-TCN CVPR2019 (tau=4,λ=0.15 confirmado), ASRF WACV2021 (theta_t,kernel), Viterbi]`.
9. **Rejeição de "sem golpe".** Primário: gating por histerese + piso de confiança (já rejeita repouso).
   Secundário/opcional: 7ª classe **background** minerada dos frames inter-golpe do **V6** (única fonte
   raw — 46.497 frames per-frame, só 685 anotados; sem RGB cru das outras no remoto). `[alta — calib:
   C+1 background TAL; OS-SAR 2024 MSP basta; hard-negative mining ECCV2018]`.
10. **Arquitetura pequena (manter).** BiLSTM(64)+attention. **A/B opcional** contra um TCN dilatado
    estilo FenceNet (o análogo publicado mais próximo: 6 ações finas de esporte, 2D, ~650 clips, ganhou
    de depth+IMU). **Deep ensemble (3–5 seeds, média de softmax)** — melhor calibração sob shift, +2–4%
    acc, stream mais suave, sinal de desacordo p/ OOD de graça (trivial na H100). `[alta — FenceNet
    2204.09434; Deep Time Ensembles PMC8512601]`.
11. **Métricas (rigor — é a prioridade #1 do PLANO_ACAO).** LOVO (leave-one-video-out, V1–V10):
    média±desvio da acc 6-classes + acc TIPO 3-classes por fold (separa view de mão). CI bootstrap.
    **Segmentais**: edit distance + F1@{10,25,50} (frame-acc é cega ao piscar). **ECE** (15 bins) +
    AUROC/FPR95 punch-vs-background. `[alta — TAS survey; Guo 2017 ECE; Ovadia 2019: temp scaling quebra
    sob shift → escolher limiar no cross-video]`.

**Descartado conscientemente (com motivo):** GCN/transformer grande (overfit a 5.5k); frontalização 2D
total (mal-posta); temperature scaling como calibração principal (quebra sob shift); focal loss como
correção primária de prior (não move a fronteira por log(prior)); T-MSE smoothing loss e temporal label
smoothing (precisam de cabeça por-frame densa que não temos — adiados); 3D-lifting/PoseC3D (pesado, fere
"simples"); background das outras 9 fontes (precisaria baixar YouTube — IP-frágil, fora de escopo agora).

---

## Decisão de branch

- **Base:** `feat/body-frame-event-inference` (a mais avançada; tem norm pelo corpo + inferência por
  evento + os experimentos de stance). É a base correta — avanço incremental, não reescrita.
- **Nova branch:** `feat/type-stance-decoupling`.
- **Reaproveita (não mexe na essência):** norm pelo corpo (`body_frame_windows`), `make_window`,
  `load_video`/splits, `flip_body`+`SWAP_*` (mirror), `detect_events` (vira a Etapa A do decoder),
  arquitetura BiLSTM+attention, sidecar `norm_stats.npz`, `ensure_25fps`/extração YOLO do `boxe.py`.
- **Refatora:** `preprocess_windows` (adiciona features de ângulo/direção + de-roll + máscara);
  `train.py` (logit-adjust, máscara, ensemble, harness LOVO); `boxe.py` (decoder Viterbi no lugar da
  média-no-pico, gating na velocidade normalizada). Os experimentos `_stance*.py`/`_hybrid.py` viram
  o módulo `stance.py` validado.

---

## File structure

| Arquivo | Responsabilidade | Ação |
|---|---|---|
| `boxe_utils.py` | features compartilhadas treino==inferência | **modificar**: `engineered_features`, `deroll`, `feature_pipeline` (substitui `preprocess_windows`), helper de máscara |
| `stance.py` | geometria de stance/mão por-clipe (lead/rear) | **criar** (consolida `_stance*.py`/`_hybrid.py`, validado) |
| `eval_utils.py` | LOVO, CI bootstrap, métricas segmentais (edit/F1), ECE | **criar** |
| `model.py` | construção dos modelos (6-classes c/ máscara; TIPO 3-classes; ensemble) | **criar** (extrai `build_model` do `train.py`) |
| `train.py` | treino 6-classes + TIPO, logit-adjust, ensemble, relatório LOVO | **modificar** |
| `decode.py` | decoder temporal (gating normalizado + Viterbi numpy + relabel + background) | **criar** |
| `boxe.py` | inferência no vídeo usando `decode.py` + `stance.py` | **modificar** |
| `_verify_adam.py` | quebra adam.mp4 em frames e audita (suave? contíguo? classe certa?) | **criar** (experimento) |
| `MUDANCAS.md` / `PLANO_ACAO.md` | documentação viva | **atualizar no fim** |

---

## Tarefas

> Ordem importa: **rigor primeiro** (T0), depois features (T1–T2), treino (T3), stance+gate (T4),
> decomposição (T5), decoder (T6), background opcional (T7), seleção+vídeo (T8), docs (T9). Cada tarefa
> roda nas H100 via `jp run` e imprime a métrica que a aprova/reprova. Subagents implementam um arquivo
> por vez; eu reviso e meço.

### Task 0: Branch nova + harness de avaliação (LOVO + CI + segmentais + ECE)

**Files:**
- Create: `eval_utils.py`
- Create (experimento): `_eval_baseline.py`

**Interfaces:**
- Produces: `lovo_split(vids)` → iterador de `(train_vids, test_vid)`; `bootstrap_ci(y_true, y_pred,
  n=1000, seed=0)` → `(acc, lo, hi)`; `segmental_edit(seq_true, seq_pred)` → float; `segmental_f1(
  seq_true, seq_pred, overlap)` → float; `ece(probs, y_true, bins=15)` → float; `floors(y_true)` →
  `(majoritario, aleatorio)`.

- [ ] **Passo 1: criar a branch a partir da base correta**
```bash
cd /Users/pedro/Downloads/reconhecimento/boxe.ml
git checkout feat/body-frame-event-inference
git checkout -b feat/type-stance-decoupling
git config user.email pehqge@gmail.com   # atribuição correta do Pedro
```

- [ ] **Passo 2: escrever `eval_utils.py`** com LOVO, CI bootstrap, edit distance (Levenshtein sobre a
  sequência de SEGMENTOS, não frames), F1@{10,25,50} (IoU temporal contra spans de GT), ECE (15 bins),
  pisos. Funções puras numpy/sklearn. Edit/F1 seguem o protocolo MS-TCN (cada segmento predito conta como
  acerto se IoU temporal com um segmento GT de mesma classe ≥ k/100).

- [ ] **Passo 3: `_eval_baseline.py`** — carrega `modelo_boxe.keras` atual, roda LOVO nas 10 fontes,
  imprime por fold a acc 6-classes e a acc TIPO 3-classes + média±desvio + CI + pisos. Estabelece o
  número de partida honesto (substitui o "0.31 num split só").

- [ ] **Passo 4: medir (rodar no remoto, GPU mais vazia)**
```bash
jp run _eval_baseline.py
```
Esperado: imprime ~10 folds; média 6-classes na casa de 0.3–0.5 (varia por fold), TIPO maior. **Aprova
se** rodar limpo e os números forem coerentes com o 0.31/0.49 já medido (sanity check do harness).

- [ ] **Passo 5: commit**
```bash
git add eval_utils.py
git commit -m "feat(eval): harness LOVO + CI bootstrap + metricas segmentais (edit/F1) + ECE"
```

---

### Task 1: Features view-invariantes + de-roll + máscara do padding (`boxe_utils.py`)

**Files:**
- Modify: `boxe_utils.py` (adiciona funções; `feature_pipeline` substitui `preprocess_windows` mantendo
  o nome antigo como alias p/ não quebrar `boxe.py`)
- Create (experimento): `_ablate_features.py`

**Interfaces:**
- Consumes: `body_frame_windows(windows)` (existente).
- Produces: `deroll(windows)` → `(N,25,17,2)` com linha de quadril nivelada;
  `engineered_features(coords_bodyframe)` → `(N,25,F_eng)` com [ângulo cotovelo E/D, ângulo
  ombro-cotovelo-quadril E/D, ângulo linha-ombro, ângulo linha-quadril, torção, razão-vertical
  velocidade punho E/D, alcance-frontal punho E/D]; `feature_pipeline(windows, mean, std)` →
  `(N,25,C)` coords padronizadas ⊕ vel/acc ⊕ features de ângulo ⊕ máscara-aware; `window_mask(windows)`
  → `(N,25)` booleano (frame real?). Mantém `preprocess_windows = feature_pipeline` como alias.

- [ ] **Passo 1: `deroll`** — para cada frame real, `alpha = atan2(hipR.y−hipL.y, hipR.x−hipL.x)`
  (juntas 12,11), rotaciona todas as juntas por `−alpha` em torno do quadril médio. Pad fica 0.

- [ ] **Passo 2: `engineered_features`** — fórmulas exatas (índices COCO: ombro 5/6, cotovelo 7/8,
  punho 9/10, quadril 11/12):
  - ângulo de cotovelo (braço D = junta 8): `a=kp5_6−kp8 ; b=kp10−kp8 ; ang=atan2(|a×b|, a·b)`; idem
    braço E (5,7,9). **Sinais com `atan2(cross,dot)`** p/ preservar direção do hook.
  - linha de ombro `atan2(kp6−kp5)`, linha de quadril `atan2(kp12−kp11)`, torção = ombro−quadril.
  - razão vertical da velocidade do punho `|vy|/(|vx|+|vy|+eps)` (alto = uppercut); alcance-frontal
    assinado `vx` (alto = reto). Velocidades **no referencial do corpo já de-rolled**.
  - Sobre janelas com pad: features só nos frames reais; pad = 0.
  - **Caveat a checar empiricamente:** o de-roll alinha eixos no plano, mas a razão vertical/horizontal
    pode não bater entre frontal e lateral. Por isso T1 **adiciona** essas features às coords (não
    substitui) e a ablação (Passo 5) decide o que fica.

- [ ] **Passo 3: máscara do padding** — `feature_pipeline` mantém o pad **exatamente 0** (padroniza só
  os frames reais; vel/acc com `prepend` do primeiro frame real, não do zero). Expõe `window_mask` p/ o
  `model.py` plugar `Masking`.

- [ ] **Passo 4: `feature_pipeline`** monta `[coords_padr(34) ⊕ vel(34) ⊕ acc(34) ⊕ eng(F_eng)]`,
  shape final `(N,25,C)`. Garante treino==inferência (mesma função no `train.py` e `boxe.py`).

- [ ] **Passo 5: ablação (medir no remoto)** — `_ablate_features.py` treina 5 seeds em LOVO para 3
  variantes: (a) só coords (baseline), (b) coords+ângulos, (c) só ângulos+direções. Imprime
  média±desvio cross-video 6-classes e TIPO + CI.
```bash
jp run _ablate_features.py
```
**Aprova a variante** que melhora a acc TIPO cross-video além do CI sem derrubar same-source. Esperado:
coords+ângulos ≥ baseline (ângulos ajudam o TIPO a generalizar). Se ângulos não ajudarem, manter coords
e registrar o resultado negativo (empirismo honesto).

- [ ] **Passo 6: commit**
```bash
git add boxe_utils.py
git commit -m "feat(features): de-roll + features de angulo/direcao view-invariantes + mascara do padding"
```

---

### Task 2: Pose-noise augmentation (`boxe_utils.py` / pipeline de treino)

**Files:**
- Modify: `boxe_utils.py` (helper `augment_pose(windows, rng, cfg)`)
- Create (experimento): `_ablate_aug.py`

**Interfaces:**
- Produces: `augment_pose(coords, rng, jitter=0.02, drop_p=0.1, rot_deg=12, scale=(0.9,1.1),
  twarp=(0.8,1.2))` → coords aumentadas (aplicado **antes** de `feature_pipeline`, nos frames reais).

- [ ] **Passo 1:** implementar jitter gaussiano por junta (σ × largura de ombro), dropout de junta
  (zera junta → tratada como não-detectada), rotação no plano pequena em torno do quadril, escala,
  warp temporal (reamostra os frames reais). Tudo só nos frames reais; recalcular vel/acc **depois**.

- [ ] **Passo 2: ablação (medir)** — `_ablate_aug.py` compara treino com/sem cada componente em LOVO
  (5 seeds, CI). Mantém os componentes que ajudam cross-video; descarta os que pioram (LSTM odeia jitter
  forte). 
```bash
jp run _ablate_aug.py
```
**Aprova** os componentes com ganho cross-video > CI. Esperado: jitter+rotação leve ajudam; warp forte
pode atrapalhar.

- [ ] **Passo 3: commit**
```bash
git add boxe_utils.py
git commit -m "feat(aug): pose-noise augmentation (jitter/dropout/rot/scale/twarp) gated por ablacao LOVO"
```

---

### Task 3: Treino melhorado — logit-adjust + máscara + ensemble (`model.py`, `train.py`)

**Files:**
- Create: `model.py`
- Modify: `train.py`
- Create (experimento): `_train_eval.py`

**Interfaces:**
- Produces (`model.py`): `build_model(input_shape, n_classes, mask=True)` → BiLSTM(64,
  return_sequences)+attention com `Masking` na entrada e `GlobalAveragePooling1D` mask-aware;
  `logit_adjusted_loss(log_pi, tau=1.0, label_smoothing=0.05)` → loss Keras; `train_ensemble(Xtr,ytr,
  Xval,yval, n_classes, seeds, **cfg)` → lista de modelos; `predict_ensemble(models, X)` → softmax médio.
- Produces (`train.py`): salva `modelo_boxe.keras` (ou ensemble `modelo_boxe_e{k}.keras`) + `norm_stats.npz`.

- [ ] **Passo 1: `model.py`** — extrai `build_model` do `train.py`. **Máscara explícita (sem depender do
  comportamento de máscara do GAP1D):** `Masking(0.0)` na entrada → BiLSTM → attention → **masked-mean
  via `Lambda`** (`sum(x*mask)/sum(mask)`, computando a máscara de `Masking` com `compute_mask`/`!=0`),
  no lugar do `GlobalAveragePooling1D` cego. Garante média só dos frames reais. `Dense` final **sem
  ativação** (logits). `logit_adjusted_loss(log_pi, tau, label_smoothing)`: subclasse de `keras.losses.Loss`,
  `call(y, z)` → `CategoricalCrossentropy(from_logits=True, label_smoothing=ls)(y, z + tau*log_pi)`;
  `log_pi` constante das frequências de treino. **Inferência:** `softmax(model(x))` (a softmax é aplicada
  fora do modelo, sempre igual). Helper `predict_proba(model, X)` centraliza isso p/ treino==inferência.

- [ ] **Passo 2: `train.py`** — troca `class_weight='balanced'` por `logit_adjusted_loss` (não empilhar);
  usa `feature_pipeline` (T1) + `augment_pose` (T2); treina `n=5` seeds (ensemble); reporta via
  `eval_utils` LOVO + CI + TIPO.

- [ ] **Passo 3: varrer tau (medir)** — `_train_eval.py` varre `tau∈{0,0.5,1.0,1.5}` em LOVO (5 seeds),
  imprime acc 6-classes, **recall da Rear Hook**, macro-F1, CI.
```bash
jp run _train_eval.py
```
**Aprova o tau** que sobe o recall da Rear Hook sem derrubar a acc global (fora do CI). Esperado: tau~1.0
tira a Rear Hook de ~0.1 para ≥0.3.

- [ ] **Passo 4: medir ensemble vs único** — comparar `predict_ensemble` (5) vs 1 modelo em LOVO: acc,
  macro-F1, **ECE**. Esperado: ensemble melhora ECE ≥40% e acc +2–4%.

- [ ] **Passo 5: commit**
```bash
git add model.py train.py
git commit -m "feat(train): logit-adjusted softmax + mascara no BiLSTM + deep ensemble; relatorio LOVO/ECE"
```

---

> **Nota de paralelismo:** a Task 4 (stance) é **pura geometria sobre os esqueletos crus, independente do
> modelo** — pode ser implementada e **medida cedo, em paralelo com T1–T3**. O resultado do GATE decide o
> `mode` da T5, então convém rodar `_validate_stance.py` assim que `stance.py` existir.

### Task 4: Módulo de stance (mão lead/rear por geometria) + **GATE de validação** (`stance.py`)

**Files:**
- Create: `stance.py`
- Create (experimento): `_validate_stance.py`

**Interfaces:**
- Produces: `clip_stance(skeletons)` → `+1` (ortodoxo, pé/mão esquerda à frente) / `−1` (canhoto), via
  voto por-clipe ponderado por visibilidade; `hand_of_punch(window)` → lado do punho que golpeou (mão
  com maior caminho); `lead_rear(window, stance)` → `"lead"`/`"rear"`; `stance_confidence(skeletons)` →
  fração de concordância do voto (proxy de confiança).

- [ ] **Passo 1: `stance.py`** — consolida e **melhora** `_stance2.py` (medido ~0.75) com:
  - **frente pela orientação do torso**, não pela extensão do punho (hook vai pro lado): direção frontal
    = normal da linha de ombro/quadril `n=(-a_y,a_x)`, `a=0.5*((kp6−kp5)+(kp12−kp11))`; desambígua
    frente/trás por visibilidade de orelha/olho (3,4)/(1,2) — orelha sumida = perfil.
  - **stance por-clipe** (voto na maioria dos frames), não por-janela.
  - **perna inteira** (quadril→joelho→tornozelo) com fallback joelho quando tornozelo some.
  - **voto ponderado** pela visibilidade (junta `!=0`); descarta frames com <4 de {ombros,quadris}.
  - frontal (pés nivelados) → fallback na orientação do torso.

- [ ] **Passo 2: GATE — validar a acurácia da mão no cross-video** — `_validate_stance.py` mede, por
  janela e **por-clipe**, a acurácia de `lead_rear` contra o rótulo verdadeiro (LEAD vs REAR) em TRAIN,
  TEST e **CROSS-VIDEO (V3,V4) + LOVO**.
```bash
jp run _validate_stance.py
```
- [ ] **Passo 3: DECISÃO empírica (gate):**
  - **Se cross-video ≥ 0.85** → MÃO entra como decisão geométrica forte na Task 5 (decomposição plena).
  - **Se 0.75 ≤ cross-video < 0.85** → MÃO entra **só adaptativa** (Task 5): corrige a mão apenas quando
    o modelo 6-classes está incerto entre as duas mãos do mesmo TIPO.
  - **Se < 0.75** → geometria fica só como *tie-breaker* logado; o ganho vem das features/aug/ensemble.
  Registrar o número medido no `MUDANCAS.md` (honestidade: o IDEIA_STANCE previu ~0.75; queremos saber
  se as melhorias chegam a 0.85+).

- [ ] **Passo 4: commit**
```bash
git add stance.py
git commit -m "feat(stance): mao lead/rear por geometria de stance por-clipe (frente pelo torso, perna inteira, voto ponderado)"
```

---

### Task 5: Decomposição TIPO×MÃO + combinador adaptativo (`model.py`, `train.py`)

**Files:**
- Modify: `model.py` (cabeça de TIPO; combinador)
- Modify: `train.py` (treina TIPO 3-classes)
- Create (experimento): `_compare_heads.py`

**Interfaces:**
- Produces: modelo TIPO 3-classes (`modelo_tipo.keras`); `combine(type_probs, six_probs, stance, mode)`
  → classe 6 final, onde `mode ∈ {"direct6","decomp","adaptive"}` conforme o gate da Task 4;
  `MAP_TYPE_HAND` = mapeamento (TIPO, mão) → uma das 6 classes.

- [ ] **Passo 1:** treinar o modelo de TIPO (3-classes: reto/hook/uppercut) com a mesma pipeline
  melhorada (features T1, aug T2, máscara, ensemble). Mirror p/ TIPO é sem troca de rótulo (hook→hook).

- [ ] **Passo 2: combinador** conforme o gate:
  - `decomp`: `classe = MAP_TYPE_HAND[argmax(type_probs), lead_rear]`.
  - `adaptive`: usa `six_probs`; se as duas classes de topo são do mesmo TIPO e diferem só na mão **e** a
    margem < δ (tunar) → sobrescreve a mão por `lead_rear`; senão mantém `argmax(six_probs)`. Nunca
    sobrescreve quando o modelo está confiante (preserva in-distribution).
  - `direct6`: fallback (só features/aug/ensemble, sem geometria).

- [ ] **Passo 3: comparar (medir)** — `_compare_heads.py` mede em LOVO + CI: `direct6` vs `decomp` vs
  `adaptive`, acc 6-classes, macro-F1, recall por classe.
```bash
jp run _compare_heads.py
```
**Aprova o `mode`** com maior acc cross-video dentro do orçamento do gate. Esperado (se stance≥0.85):
`decomp`/`adaptive` aproxima cross-video do oracle 0.49. Se stance~0.75: `adaptive` ≥ `direct6`,
`decomp` não.

- [ ] **Passo 4: commit**
```bash
git add model.py train.py
git commit -m "feat(decomp): cabeca de TIPO 3-classes + combinador (direct6/decomp/adaptive) escolhido por LOVO"
```

---

### Task 6: Decoder temporal — gating normalizado + Viterbi (numpy) + relabel (`decode.py`, `boxe.py`)

**Files:**
- Create: `decode.py`
- Modify: `boxe.py` (usa `decode.py` + combinador da Task 5)
- Create (experimento): `_tune_decoder.py`

**Interfaces:**
- Produces (`decode.py`): `wrist_speed_bodyframe(skeletons, mean, std)` → velocidade do punho no
  referencial do corpo (não pixel); `gate_events(speed, t_hi, t_lo, min_len, gap)` → spans ativos
  (histerese); `viterbi(posteriors, p_self)` → trilha de estados (numpy, matriz pegajosa
  `transição[i,i]=p_self`, fora-diagonal `(1−p_self)/(C)`); `relabel(seq, theta_t)` → funde
  sub-segmentos curtos; `decode_clip(per_frame_probs, speed, cfg)` → rótulo por frame (1 contíguo por
  golpe, background nos gaps).

- [ ] **Passo 1: `viterbi` em numpy** — DP clássico em log-espaço sobre `(T, C)` posteriors + matriz de
  transição pegajosa. ~20 linhas, sem dependência nova (librosa ausente no remoto). Inclui estado
  **background** (prior baixo) p/ os gaps não serem forçados a um golpe.

- [ ] **Passo 2: gating normalizado** — `wrist_speed_bodyframe` mede a velocidade do punho **depois** da
  norm pelo corpo (comparável entre ângulos); histerese `t_hi`>`t_lo`, exige `≥3` frames abaixo de `t_lo`
  p/ fechar (substitui o gate em pixel cru do `boxe.py`).

- [ ] **Passo 3: `decode_clip`** — expande o softmax por-janela p/ por-frame; Viterbi sobre `(T, 6+1)`;
  dentro de cada span ativo, **moda da trilha** = rótulo do golpe inteiro; relabel apaga lampejos curtos;
  **fallback de TIPO**: se a massa 6-classes no span é quase uniforme entre as mãos do TIPO, exibe no
  nível do TIPO (ou aplica a mão da `stance`).

- [ ] **Passo 4: tunar (medir segmentais)** — `_tune_decoder.py` faz grid de `p_self∈{0.85..0.97}`,
  `theta_t∈{4..8}`, limiares do gate, **escolhendo no cross-video/LOVO** pela **edit distance segmental**
  (não frame-acc). Reporta edit + F1@{10,25,50} antes/depois.
```bash
jp run _tune_decoder.py
```
**Aprova** os parâmetros que maximizam o edit segmental cross-video. Esperado: edit sobe bastante
(menos over-segmentation), 1 rótulo contíguo por golpe.

- [ ] **Passo 5: `boxe.py`** — substitui a média-no-pico pela `decode_clip`; rótulo do span do golpe;
  repouso em branco/background.

- [ ] **Passo 6: commit**
```bash
git add decode.py boxe.py
git commit -m "feat(decode): gating normalizado + Viterbi pegajoso (numpy) + relabel; 1 rotulo continuo por golpe"
```

---

### Task 7 (opcional / se houver ganho): classe Background do V6

**Files:**
- Create (experimento): `_mine_background.py`
- Modify: `train.py`, `model.py` (7ª classe)

- [ ] **Passo 1:** minerar janelas de 25 frames dos frames do **V6** (formato `[frame_idx,x,y]`,
  46.497 frames) que estão **fora** de qualquer intervalo de golpe anotado (685 anotados), com banda de
  guarda ±5 frames ao redor de cada golpe. Extrair só `x,y` (cols 1,2), normalizar igual.

- [ ] **Passo 2:** treinar com 7ª classe (cap background ≈1–2× o total de golpes; manter peso/ajuste na
  Rear Hook p/ não soçobrar). Medir AUROC/FPR95 punch-vs-background + se a acc de golpe cai.

- [ ] **Passo 3: decisão** — só mantém se rejeitar repouso melhor que o gating sozinho **sem** derrubar
  a acc de golpe. Senão, fica documentado como tentado (o gating + piso de confiança já cobrem o repouso).

- [ ] **Passo 4: commit (se aprovado)**
```bash
git add _mine_background.py train.py model.py
git commit -m "feat(background): 7a classe minerada do V6 (gated por AUROC/FPR95)"
```

---

### Task 8: Seleção final por LOVO + **auditoria frame-a-frame do adam.mp4**

**Files:**
- Create (experimento): `_verify_adam.py`
- Modify: docs

**Interfaces:**
- Produces: `_verify_adam.py` gera `outputs/adam.mp4` com a config vencedora e dumpa o rótulo por frame;
  monta um contact-sheet e calcula métricas de suavidade no próprio vídeo (nº de transições, nº de
  trocas de classe dentro de um golpe, frames de repouso rotulados).

- [ ] **Passo 1:** escolher a config final (features+aug+loss+mode+decoder) pelo **melhor LOVO
  cross-video 6-classes com CI**, registrando o número.

- [ ] **Passo 2: gerar o vídeo**
```bash
jp run boxe.py -v videos/adam.mp4 -m <modelo_final>
jp pull   # traz outputs/adam.mp4
```

- [ ] **Passo 3: AUDITORIA frame-a-frame (local, ffmpeg)** — quebrar `outputs/adam.mp4` em todos os
  frames, ler o rótulo (barra STATUS) de cada um, e **verificar as regras do Pedro**:
  - cada golpe aparece **do início do movimento até o fim** (sem sumir no meio);
  - **sem piscar** para "..." dentro de um golpe;
  - **sem trocar** de classe dentro de um golpe;
  - repouso fica em branco (sem golpe fantasma);
  - a classe exibida está **correta** (conferir manualmente alguns golpes claros — jab reto mão da frente
    etc.).
  `_verify_adam.py` calcula: nº de spans, comprimento de cada, trocas-de-classe-intra-golpe (deve ser 0),
  transições liga/desliga (~2× nº de golpes), frames de repouso rotulados (~0).

- [ ] **Passo 4: gate visual** — se qualquer regra falhar, **voltar** à tarefa correspondente (decoder
  p/ piscar; stance/combinador p/ classe errada; gating p/ golpe fantasma), corrigir e repetir. Iterar
  até passar. **Eu (orquestrador) leio o contact-sheet e os frames críticos** — não confio só na métrica.

- [ ] **Passo 5: commit**
```bash
git add outputs/adam.mp4
git commit -m "chore: adam.mp4 final auditado frame-a-frame (suave, continuo, classe correta)"
```

---

### Task 9: Documentação

**Files:**
- Modify: `MUDANCAS.md`, `PLANO_ACAO.md`, `README.md`

- [ ] **Passo 1:** registrar no `MUDANCAS.md`, no mesmo estilo factual: cada lever, o número medido
  antes/depois (com CI), as decisões dos GATES (acurácia de stance medida, `mode` escolhido, parâmetros
  do decoder), e o que foi **tentado e falhou** (honestidade empírica).
- [ ] **Passo 2:** atualizar o `PLANO_ACAO.md` (o que dos itens #1–#5 foi feito) e o `README.md`
  (nova pipeline: features de ângulo, decomposição, decoder).
- [ ] **Passo 3: commit**
```bash
git add MUDANCAS.md PLANO_ACAO.md README.md
git commit -m "docs: registra levers, gates medidos e resultados (antes/depois com CI)"
```

---

## Critérios de sucesso (empíricos, pré-registrados)

1. **CROSS-VIDEO 6-classes (LOVO, com CI)** sobe de forma significativa acima de 0.31 em direção ao
   oracle 0.49 — sem derrubar o same-source (≥0.85). O quanto depende do gate de stance; meta honesta:
   fechar a maior parte da "metade TIPO" do gap via features/aug e a "metade mão" conforme o stance medido.
2. **Recall da Rear Hook (cross-video, CI)** sai de ~0.1 para **≥0.3** (logit-adjust + mirror + TIPO).
3. **Suavidade (edit segmental cross-video)** melhora claramente; no `adam.mp4`: **0 trocas de classe
   dentro de um golpe**, 1 rótulo contíguo do início ao fim, ~0 frame de repouso rotulado.
4. **Calibração (ECE)** melhora (sem "crava 100%"); medido same-source E cross-video.
5. **adam.mp4 passa a auditoria frame-a-frame** contra as 5 regras do Pedro (Task 8).
6. Todo número com **média±desvio + CI + pisos**; nada entra como melhora se estiver dentro do CI.

## Riscos e caveats honestos

- **O teto é o oracle 0.49 cross-video**, não 0.87 — em 2D, com câmeras tão diferentes e poucas fontes,
  OOD não "fecha"; entregamos um ganho real e medido + limitação documentada (já é um TCC excelente).
- **A geometria da mão é o gargalo conhecido (~0.75).** Todo o ganho da decomposição depende de subir
  isso. Por isso é um GATE medido **antes** de refatorar o modelo. Se não subir, o plano degrada
  graciosamente para `adaptive`/`tie-breaker` — não quebra.
- **O decoder conserta o piscar, não o acerto.** Viterbi sobre stream errado vira "errado mas suave",
  que pode ser pior numa demo. Mitigado com fallback de TIPO + background + escolha por edit segmental.
- **Razão vertical/horizontal pode não alinhar entre frontal e lateral** — por isso as features de
  direção entram por ablação (adicionadas, não substituem) e a decisão é empírica.
- **adam.mp4 é OOD** (pessoa/câmera fora do treino), mas é single-person shadow boxing, provavelmente
  mais perto do in-distribution que V3/V4 — a demo deve ficar boa com modelo forte in-distribution +
  decoder; o número científico forte continua sendo o cross-video/LOVO com CI.
