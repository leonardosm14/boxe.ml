# HANDOFF — estado do projeto boxe.ml (para a próxima sessão)

> Documento de passagem de bastão. Escrito ao fim de uma sessão longa para que uma sessão nova
> continue sem re-derivar nada. Honesto sobre o que funcionou, o que falhou e o que está bloqueado.
> Estilo do código do projeto: numpy/Keras functional, comentários PT, simples.

---

## TL;DR

- **Objetivo:** reconhecer golpes de boxe em vídeo (6 classes: Jab, Cross, Lead/Rear Hook,
  Lead/Rear Uppercut). Dois problemas: (1) confundir golpes parecidos (a MÃO: jab×cross, lead×rear);
  (2) o rótulo "piscar"/trocar de classe no meio do golpe.
- **O que foi entregue:** pipeline nova (branch `feat/type-stance-decoupling`), cross-video
  **0.31 → 0.41** (com IC), TEST in-distribution **0.85**, calibração honesta (ECE 0.073),
  decoder por pico (1 rótulo contíguo por golpe). Dois PRs abertos.
- **BLOQUEIO ATUAL / próximo passo:** no vídeo demo `adam.mp4` (fora do dataset) **a CLASSE ainda
  erra** — é OOD. Não há ground-truth do adam. Foi criada uma **ferramenta de anotação**
  (`annotate/`, publicada) pra o grupo gerar `adam_gt.csv`. **Quando o CSV chegar:**
  `jp run _measure_adam.py` mede a acurácia real e diagnostica se o erro é sistemático (mão
  espelhada → conserta) ou aleatório (OOD duro). Esse é o caminho.

---

## A ideia central (IDEIA_STANCE.md)

A confusão dos golpes parecidos é **sempre a MÃO**: jab×cross, lead×rear hook, lead×rear uppercut só
diferem por qual mão golpeou. E qual é "a da frente" (lead) depende da **guarda**: a mão do mesmo lado
do pé da frente (ortodoxo/destro → esquerda é a lead). Então a proposta era **decompor**:
- **TIPO** (3 classes: reto/hook/uppercut) → o modelo aprende (é invariante à mão, generaliza).
- **MÃO** (lead/rear) → resolver por **geometria** da stance, sem treinar.
- Output final = `MAP(tipo, mão)` → as 6 classes nomeadas (decisão confirmada com o Pedro: output 6).

**Veredito empírico (importante):** a geometria da mão ficou **fraca (~0.77 cross-video)** e **falha
em câmera espelhada** (não recupera). Então a decomposição `3cl+geometria` ficou ≈ igual ao modelo
direto de 6 classes. O ganho real veio de outras coisas (features/máscara/ensemble). A geometria
entrou só como **tie-breaker adaptativo** (corrige a mão só quando o modelo titubeia; nunca sobrescreve
confiante). Confirma o caveat que o próprio IDEIA_STANCE previa.

---

## O que foi feito (levers) e o que cada um deu (MEDIDO)

| Lever | Arquivo | Resultado medido |
|---|---|---|
| Features view-invariantes (ângulos cotovelo/ombro + torção + de-roll) | `boxe_utils.py` | ajuda: só-coords 0.362 → +eng 0.380; sem de-roll cai p/ 0.332 |
| **Máscara do padding (bug corrigido)** | `model.py` | padding virava `−mean/std` e o BiLSTM comia lixo; agora mascarado só no pooling |
| Logit-adjusted softmax (balanced softmax) | `model.py` | troca o `class_weight='balanced'`; pra Rear Hook (faminta) |
| Deep ensemble (5 seeds, média softmax) | `train.py` | cross-video single 0.38 → ensemble **0.406**; ECE 0.073 |
| **Decoder por PICO** (find_peaks na velocidade do punho) | `decode.py`/`boxe.py` | substitui histerese; pega ~20-25 golpes (era 9); 1 rótulo contíguo, 0 troca no meio |
| Stance geometria (mão lead/rear) | `stance.py` | gate ~0.77 cross-video; NÃO recupera fold espelhado (tie-breaker só) |
| Avaliação séria (LOVO + IC + edit/F1 + ECE) | `eval_utils.py` | número científico honesto |

**Resultados finais (5-seed ensemble, com IC bootstrap):**
- TEST (V5,V9 same-source): **0.849** [0.821, 0.872]
- **CROSS-VIDEO (V3,V4): 0.31 → 0.406** [0.380, 0.432] — significativo (IC não toca 0.31)
- TIPO 3-classes cross-video: 0.49 → **0.564**
- LOVO (parcial, 4 folds): V1 0.34 · V2 0.47 · V3 0.60 · **V4 0.011** (patológico, câmera espelhada)
- Rear Hook recall: ainda fraca (~0.09) — fome de dados (133 amostras).

**Tentado e que FALHOU (medido, não repetir):**
- pose-aug (rotação/jitter/scale/warp): **não move o cross-video** (4 variantes testadas). Em 2D,
  rotação no plano não simula mudança de ponto de vista fora-do-plano.
- tau=1.5 no logit-adjust: sem ganho sobre 1.0.
- stance pra recuperar a mão OOD: falha no V4 (0.013→0.013). É o limite 2D.

---

## Mapa dos arquivos (branch `feat/type-stance-decoupling`)

- `boxe_utils.py` — features compartilhadas treino==inferência: `body_frame_windows`, `deroll`,
  `engineered_features`, `feature_pipeline` (flag `use_eng`/`use_deroll`), `mirror_windows`,
  `augment_pose`, `window_mask`, `axis_stats`. **Fonte única de feature.**
- `model.py` — `build_model` (BiLSTM+MHA, **sem camada Masking**, máscara via `FrameMask`+`MaskedMean`
  só no pooling → cuDNN rápido + robusto a frame interno zerado), `LogitAdjustedLoss` (logits, não
  softmax), `predict_proba`/`predict_proba_ensemble`, `log_prior`.
- `train.py` — `train_ensemble` (por seed, **imprime por seed** — ver gotcha jp), `report` (LOVO/IC/ECE),
  `main` (treina 6cl + TIPO, salva `modelo_boxe_e0..e4.keras` + `norm_stats.npz`).
- `decode.py` — `detect_punches` (find_peaks, **detecção por pico** = a correta), `viterbi`,
  `gate_events`, `relabel`, `decode_clip` (legado de histerese; o boxe.py usa o por-pico).
- `boxe.py` — inferência no vídeo: YOLO → `per_frame_probs` → `wrist_speed_bodyframe` →
  `label_punches` (pico + classifica no pico + tie-breaker de mão adaptativo) → render. Dumpa
  `labels_<base>.npy` e imprime os spans.
- `stance.py` — geometria lead/rear por janela (`lead_rear`, `clip_stance`, `hand_of_punch`).
- `eval_utils.py` — `lovo_split`, `bootstrap_ci`, `segmental_edit`, `segmental_f1`, `ece`, `floors`.
- `gpupick.py` — escolhe a GPU mais vazia (chamar ANTES de importar TF).
- `annotate/` — ferramenta de anotação (`annotate.html` + fotos de referência). Versão pública
  publicada: `https://tinyurl.com/2cjue6ag` (ou `gistpreview.github.io/?47b7e95b79f6ce3e4a84f7e8f20d86b0`).
- `_*.py` — scripts de experimento (gitignored, rodados via `jp run`): `_headline`, `_ablate`,
  `_lovo`, `_adaptive_eval`, `_debug_adam`, `_debug2`, `_measure_adam` (pronto p/ o GT).
- Docs: `IDEIA_STANCE.md` (a ideia), `PLANO_IMPLEMENTACAO.md` (o plano), `MUDANCAS.md` (auditoria +
  resultados), `PLANO_ACAO.md` (plano antigo).

---

## Infra: jp + H100 (gotchas que custaram tempo — LER)

- Roda nas H100 da UFSC via **`jp run <script.py>`** (sobe o script e executa no remoto, streama a
  saída). Editar local → `jp push` (sincroniza) → `jp run`. Os `.npy`/dataset já estão no remoto.
- **GPU:** sempre escolher a mais vazia (`gpupick.pick_and_set_gpu()` no topo, ANTES de `import
  tensorflow`). Lab compartilhado — rodar em GPU cheia atrasa/falha.
- **GOTCHA #1 — idle-disconnect do `jp run`:** se o script fica **silencioso por muito tempo**
  (~150-250s, ex. treino com `verbose=0`), a sessão do `jp run` desconecta e o processo remoto morre
  (sai exit 0 mas a saída trunca no meio). **Solução:** emitir saída com frequência (`verbose=2` ou
  print por seed/fold). Por isso `train_ensemble` imprime por seed. **Não rodar `jp run` + `&`** (o
  `&` mata o processo remoto quando o wrapper fecha).
- **GOTCHA #2 — cuDNN + máscara:** `keras.layers.Masking` força o caminho lento E quebra o cuDNN com
  `_assert_valid_mask` quando há frame interno todo-zero (detecção perdida → máscara não right-padded).
  Solução adotada: **não usar camada Masking**; LSTM roda os 25 frames (cuDNN) e a máscara é aplicada
  só no pooling (`FrameMask`+`MaskedMean`). 5-10× mais rápido.
- **Keras 3.14 / TF 2.21:** `keras.saving.register_keras_serializable` (não `tf.keras.saving`).
- Pra pegar saída sem o tail bufferizar, redirecionar pra um log: `jp run x.py > /tmp/x.log 2>&1`.

---

## adam.mp4 — o entregável visível e por que ainda não está pronto

- adam é o **vídeo demo** (single-person shadow boxing, sala escura, lateral) — **NÃO está no
  dataset BoxingVI** (que só tem V1-V10). Logo **não há ground-truth**.
- Estado: o decoder por pico pega **~20 golpes** (Pedro contou ~18 reais → recall OK). **Mas a CLASSE
  erra** (Pedro confirmou: os primeiros 9 estavam todos errados). É OOD: o modelo treinou no BoxingVI
  e adam é outro domínio (pessoa/câmera/luz). Distribuição que o modelo cospe no adam: ~58% Cross,
  0% Rear — claramente enviesado.
- **Por que não dá pra consertar às cegas:** sem GT não sei se o erro é (a) **flip de mão**
  sistemático (conserta com geometria/espelho), (b) **TIPO errado** (modelo não generaliza), ou (c)
  ruído OOD. Chutar parâmetro foi o erro que me fez declarar "pronto" cedo demais. **Não repetir.**
- **Caminho certo (pronto pra rodar):**
  1. Grupo anota o adam na ferramenta (`annotate/`, link público) → baixa `adam_gt.csv`
     (formato `start_frame,end_frame,class`, 25fps, 498 frames).
  2. Salvar `adam_gt.csv` em `boxe.ml/` → `jp run _measure_adam.py`.
  3. O script imprime: acurácia 6-classes / TIPO / MÃO no adam + **padrão do erro** (matriz
     verdade→modelo) → diz se é sistemático ou aleatório → ataca direcionado.

---

## Estado do git / PRs

- **`pose-estimation-test` = branch do Leo (original), INTACTA** — não commitar nela.
- **`main`** = só "Initial commit" (vazio).
- **`feat/body-frame-event-inference`** = branch do Leo + 1 commit de **rework** (norm pelo corpo +
  inferência por evento + calibração + bugs). **PR #1** → `pose-estimation-test`
  (https://github.com/leonardosm14/boxe.ml/pull/1). Resultados: TEST 0.73→0.87, cross-video 0.18→0.31.
- **`feat/type-stance-decoupling`** = todo o trabalho desta sessão (decomposição + features + decoder).
  **PR #2 (DRAFT/WIP)** → `pose-estimation-test`
  (https://github.com/leonardosm14/boxe.ml/pull/2). Commitado como `pehqge@gmail.com`.
- Modelos commitados: `modelo_boxe.keras` + ensemble `e0..e4` + `norm_stats.npz`.

---

## Próximos passos sugeridos (em ordem)

1. **Pegar `adam_gt.csv`** (grupo anota) → `jp run _measure_adam.py` → diagnosticar a classe no adam.
2. Conforme o diagnóstico:
   - se **flip de mão** sistemático no adam → ajustar a stance/espelho pro caso espelhado.
   - se **TIPO errado** → é generalização: precisa de **dado do domínio** (re-extrair V6 com YOLO;
     gravar mais vídeos no setup do adam e anotar; talvez incluir adam-like no treino com held-out).
3. Terminar o **LOVO completo** (rodou parcial; `jp run _lovo.py`, agora com keep-alive, dá o número
   científico completo). Atenção: o fold V4 é patológico (espelhado) e derruba a média.
4. Rear Hook (recall ~0.09): re-extrair V6 (46k frames raw, só hook/uppercut) como mais dado.
5. Decidir honestamente o escopo do trabalho: in-distribution forte + cross-video melhorado com IC +
   limitação OOD documentada **já é um TCC excelente**. O adam perfeito em classe pode exigir dado do
   domínio dele, que hoje não existe.

## Caveats honestos (não esconder do grupo)
- O gap OOD **não fecha** em 2D com poucas fontes (research-hard). Teto = oracle ~0.49 cross-video.
- A mão lead/rear entre câmeras diferentes é ~irresolvível em alta acurácia (a stance não basta).
- O número forte e honesto é **cross-video com IC** + **in-distribution** + **suavidade do decoder**,
  não a perfeição de classe no adam.
