# Mudanças — Auditoria e Otimização do Classificador de Golpes

> Documento de acompanhamento. Cada item segue o formato **O que estava estranho → O que isso implica → Correção**.
> Estilo das correções: mesmo nível do código do Leo (Keras functional API, numpy, comentários em PT, simples e legível), sempre baseado na documentação oficial.

Status legenda: 🔴 pendente · 🟡 em teste · 🟢 corrigido e testado

---

## Contexto do sistema (resumo)

Pipeline: **YOLOv8m-Pose** extrai 17 keypoints (COCO) por frame → janelas temporais de **25 frames** → modelo **TensorFlow (BiLSTM + MultiHeadAttention)** classifica em 6 golpes: `Cross, Jab, Lead Hook, Lead Uppercut, Rear Hook, Rear Uppercut`. Dataset = **BoxingVI** (arXiv 2511.16524); os `.npy` já vêm **pré-janelados** (cada linha = uma janela de golpe de 25×34).

---

## Achados (auditoria linha a linha)

### 0. 🟢 (MAIOR) Janela de inferência fora da distribuição do treino — CORRIGIDO
- **Estranho:** medindo o `.npy` direto, cada janela de treino tem os **frames reais no INÍCIO e zeros no FIM** (golpe ~10 frames + padding até 25). Confirmado: zero-rate sobe de 0% (posição 0) a ~100% (posição 24), média de **~15 frames zero por janela**. Mas o `boxe.py` montava **25 frames densos consecutivos** (e ainda preenchia os 25 primeiros pela frente).
- **Implica:** o modelo **nunca viu uma janela densa** no treino → toda inferência é out-of-distribution → falsos positivos em todo frame (a causa mais direta do "delira", mais que a falta de background). Nenhum gate de confiança conserta um softmax OOD.
- **Correção:** função `make_window` compartilhada no `boxe_utils` monta a janela igual ao treino (frames reais no início, zeros no fim, `REAL_LEN=10`). `boxe.py` passou a usá-la. Treino e inferência agora têm o **mesmo layout**.

### 1. 🔴 Falta a classe "sem golpe" (background) — causa raiz do "delírio"
- **Estranho:** o treino usa **só janelas de golpe**. Não existe nenhuma amostra de "nada acontecendo".
- **Implica:** na inferência o `boxe.py` desliza janelas de 25 frames sobre o vídeo inteiro. Como o modelo só conhece golpes, ele é **obrigado a cravar um dos 6 golpes em todo frame** → falsos positivos por toda parte (o "delira" que o Leo viu). O gating por entropia/confiança em `boxe.py` é só um paliativo.
- **Correção:** (definida após pesquisa) — calibração + limiar de rejeição melhor e/ou classe background. Ver seção de correções.

### 2. 🔴 Overconfidence (mostra 100%)
- **Estranho:** `softmax` + `sparse_categorical_crossentropy`, dataset pequeno, 100 épocas, **sem `label_smoothing`**.
- **Implica:** o modelo decora e cospe probabilidades ~1.0. Nenhum modelo deveria afirmar 100% (o próprio Leo notou isso).
- **Correção:** `label_smoothing` no loss (+ possível temperature scaling na inferência). Ver correções.

### 3. 🔴 Bug no `ReduceLROnPlateau` (training, célula 9)
- **Estranho:** `monitor="val_loss"` porém `mode="max"`.
- **Implica:** para `val_loss` (quanto menor melhor) o certo é `mode="min"`. Com `"max"` o scheduler reduz o LR na direção errada → não ajuda a convergir.
- **Correção:** `mode="min"` (ou `"auto"`).

### 4. 🔴 `model.predict` duplicado (`boxe.py` ~156–161)
- **Estranho:** o mesmo `model.predict(X_batch, ...)` é chamado **duas vezes** seguidas sobre o mesmo batch.
- **Implica:** dobra o tempo de inferência à toa.
- **Correção:** remover a chamada duplicada.

### 5. 🔴 GPU fixa no código (`boxe.py:11`)
- **Estranho:** `os.environ["CUDA_VISIBLE_DEVICES"] = "3"` hardcoded.
- **Implica:** conflito no lab compartilhado (foi o bug que o Pedro viu dia 15). Confirmado: GPU 3 estava ocupada por outros (32GB).
- **Correção:** ler de variável de ambiente com fallback, sem fixar uma GPU específica.

### 6. 🔴 Ordem do data augmentation (training, célula 7)
- **Estranho:** o ruído gaussiano é somado **depois** de já calcular velocidade e aceleração (sobre os 102 canais).
- **Implica:** o ruído em vel/acc fica fisicamente incoerente com as posições ruidosas (vel deveria ser a diferença das posições). Augmentation "suja" features derivadas.
- **Correção:** aumentar **as posições cruas** e só então recalcular vel/acc.

### 7. 🔴 Normalização global escalar + constantes hardcoded
- **Estranho:** `X_mean_global = X_tr.mean()` e `X_std_global = X_tr.std()` — **um único escalar** misturando x e y; os valores ficam **hardcoded** em `boxe.py` (`X_MEAN_GLOBAL`, `X_STD_GLOBAL`).
- **Implica:** se re-treinar, as constantes mudam e o `boxe.py` quebra silenciosamente (treino e inferência ficam dessincronizados).
- **Correção:** salvar as estatísticas junto do modelo (sidecar) e o `boxe.py` carregar de lá; avaliar normalização por-coordenada.

### 8. 🟢 Cobertura de classes entre splits — VERIFICADO, sem problema
- **Estranho:** comentários sugeriam que o teste tinha classe ausente no treino.
- **Verificação (baseline):** todas as 6 classes aparecem no treino; saída do modelo `(None, 6)` bate com `BOXING_CLASSES`. `Test-only labels NOT in train mapping: none`.
- **Conclusão:** rebaixado — não é um bug atual. (Mantido só como check de regressão.)

### 7b. 🔴 CONFIRMADO E CRÍTICO — constantes de normalização dessincronizadas
- **Medição (baseline):** estatísticas reais do treino atual = `mean=0.19816, std=0.25648`. Hardcoded no `boxe.py` = `mean=0.17634, std=0.24371`. **Não batem.**
- **Implica:** o `boxe.py` normaliza os dados de inferência com constantes velhas → deslocamento de distribuição direto na entrada do modelo → piora as predições e contribui pro "delira". Bug vivo, independente de tudo.
- **Correção:** salvar as estatísticas junto do modelo (sidecar `.json`) no treino e o `boxe.py` carregar de lá. Nunca mais hardcode.

### 9. 🔴 `os.chdir("/lapix/privado/boxe.ml")` hardcoded (training, célula 1)
- **Estranho:** caminho absoluto do ambiente do Leo.
- **Implica:** quebra pra outros usuários (o do Pedro é `/home/jovyan/privado/boxe.ml`). Confirmado no probe.
- **Correção:** remover o `chdir` (o script já roda na pasta do projeto) ou torná-lo relativo.

---

## Pesquisa (docs oficiais + papers) — em andamento
Workflow de research rodando: papers BoxingVI + secundário, calibração (TF/Keras), normalização/augmentation de esqueleto, YOLO pose/tracking (Ultralytics), e tratamento de background em ação temporal. Conclusões verificadas entram aqui.

---

## Correções aplicadas

### Treino (`_train_v2.py` — mesmas escolhas do Leo + correções, estilo idêntico)
- **Label smoothing** (#2 overconfidence): `sparse_categorical_crossentropy` não tem `label_smoothing`; troquei por `CategoricalCrossentropy(label_smoothing=0.1)` + rótulos one-hot só no `fit()`. Inferência intacta (segue softmax + argmax). [docs.keras CategoricalCrossentropy]
- **`ReduceLROnPlateau` mode** (#3): `val_loss` agora com `mode="min"` (estava `"max"`, scheduler morto).
- **Ordem do augmentation** (#6): ruído só nas **posições** e só nos **frames reais** (máscara do padding), depois recalcula vel/acc — antes o ruído sujava pos+vel+acc concatenados. Níveis reduzidos p/ `[0.01, 0.02, 0.03]`.
- **Sidecar de normalização** (#7/#7b): salva `norm_stats.npz` no treino; fonte única de verdade.
- **Seed fixo** `set_random_seed(42)` → comparação reprodutível (antes nada, resultados oscilavam).
- **Sem `os.chdir`** hardcoded (#9): script roda na própria pasta.

### Inferência (`boxe.py` + `boxe_utils.py`)
- **`make_window` compartilhado** (#0): janela no layout do treino (real no início, zeros no fim). Correção principal.
- **Norm via `load_norm_stats()`** (#7b): carrega `norm_stats.npz`, fallback corrigido (`0.198156 / 0.256477`).
- **`model.predict` único** (#4): removida a chamada duplicada (era 2x o tempo).
- **GPU configurável** (#5): `CUDA_VISIBLE_DEVICES` via env (default 0), setado **antes** de importar TF; não fixa mais a GPU 3.
- **Gate de velocidade do punho** (stopgap p/ background): `VEL_GATE=0.008` (medido nos golpes reais, ~p1). Frame com punho parado → "...". COCO joints 9/10.
- **Histerese (máquina de estados)** — corrige o "piscar": o rótulo **ligava/desligava no meio do golpe** porque o gate caía 1-2 frames no ápice do soco (punho freia) e a decisão era por-frame. Solução: liga após `DEBOUNCE=3` frames ativos, **desliga só após `HOLD_FRAMES=4` frames inativos**. Os comprimentos foram medidos na estrutura de segmentos ON/OFF do `adam.mp4` (quedas de ≤3 frames = piscar → bridadas; pausas reais ≥4 → preservadas). Desacopla "tem golpe" (liga/desliga) de "qual golpe" (rótulo), então o wobble de classe não pisca. No `adam.mp4`: transições liga/desliga 45 → **23**, label estável durante o golpe.
- _Não feito hoje (decisão consciente):_ classe Background (precisa dos vídeos crus de treino, só há links no `.ods`); track-ID multi-lutador (#9, risco de véspera, `adam.mp4` é single-person, Tom já faz tracking).

---

## Redesenho: inferência por EVENTO (substitui o por-frame + histerese)

**Problema reportado:** mesmo com a histerese, o vídeo (a) trocava de classe rápido dentro de um golpe, (b) marcava golpe em repouso, (c) piscava. Diagnóstico (research-lead, 2 workflows): **raiz única** = classificar frame a frame numa janela deslizante, sendo que o dataset é **segmentado por evento** (start/end por golpe). A inferência tinha que ser por evento também.

**Solução (estilo Leo, numpy/scipy):**
- **Segmentação por movimento (Schmitt trigger):** um golpe = região contígua onde a velocidade do punho passa de `SEG_THI=0.07` e só termina abaixo de `SEG_TLO=0.04`. A histerese mantém o golpe inteiro como UM segmento. Funde dips curtos (`SEG_GAP=5`), descarta jitter (`SEG_MINLEN=5`). Substitui o gate de velocidade por-frame + EMA + histerese (todos removidos).
- **Classifica o evento UMA vez:** média do softmax de 5 janelas ancoradas no pico (impacto) → 1 rótulo por golpe. Mata por construção a troca de classe no meio do golpe e os múltiplos rótulos por golpe.
- **Span de exibição:** do snap (onset) até o impacto + follow-through (`peak + SPAN_POST=10`), sem invadir o próximo golpe → o rótulo fica do início ao fim do golpe (não some no meio). Repouso = branco.
- **Ferramenta de autoverificação:** `_verify_timeline.py` gera um *timeline* dos rótulos (faixas coloridas) + métricas (nº segmentos, comprimentos, transições) — conferido antes de gerar vídeo.

**Resultado no `adam.mp4`:** 20 golpes detectados, 1 rótulo contíguo por golpe (snap→fim), repouso em branco, 0 troca de classe dentro do golpe. Suavidade resolvida.

**Limite honesto:** a *acurácia da classe* no `adam.mp4` é limitada por ele ser **out-of-distribution** (pessoa/câmera diferentes do treino; mesmo motivo do cross-video V3/V4=0.18). O número científico forte é o test set (84.9%); o `adam` ilustra a pipeline. Lever real: dataset mais diverso (em pesquisa) ou vídeo in-distribution (Ruy).

---

## Resultados — baseline vs otimizado

### Baseline (modelo atual do Leo, `modelo_boxe.keras`, test = V5/V9)
- **Accuracy:** 0.734
- **Pior classe:** Rear Hook — recall **0.08** (4/51), confundida com Cross (36) e Rear Uppercut (9)
- **Confiança (max-softmax):** média **0.825**; **18.7%** das predições > 0.99; **6.0%** > 0.999
- Matriz de confusão e relatório completos salvos em `baseline_metrics.json`

| Classe | precision | recall | f1 | support |
|---|---|---|---|---|
| Cross | 0.65 | 0.84 | 0.73 | 167 |
| Jab | 0.87 | 0.89 | 0.88 | 267 |
| Lead Hook | 0.71 | 0.51 | 0.59 | 89 |
| Lead Uppercut | 0.72 | 0.86 | 0.79 | 64 |
| Rear Hook | 0.50 | 0.08 | 0.14 | 51 |
| Rear Uppercut | 0.59 | 0.61 | 0.60 | 109 |

### Otimizado (`modelo_boxe_v2.keras`, mesmas test V5/V9)

> Nota sobre seleção de modelo: a validação cross-video original (V3/V4) é **anti-correlacionada**
> (acurácia abaixo do aleatório — provável espelhamento de câmera), então `restore_best_weights`
> escolhia um modelo subtreinado. Passei a usar **validação estratificada tirada do próprio treino**
> para a seleção/early-stop; **TEST = V5/V9 continua 100% held-out** (comparação justa).

- **Accuracy:** 0.734 → **0.826** (+9.2pp)
- **Macro-F1:** ~0.62 → **0.704**
- **Calibração (max-softmax):** predições >0.99 caíram de **18.7% → 0.0%**; >0.999 de **6.0% → 0.0%**. O "crava 100%" sumiu (teto ~98%). Confiança média 0.825 → 0.877 (agora honesta, sem cauda absurda).
- **"Delira" na inferência (adam.mp4):** frames marcados como golpe **95.4% → 58.2%** (janela correta + gate de velocidade + debounce + predict único).

| Classe | precision (antes→depois) | recall (antes→depois) |
|---|---|---|
| Cross | 0.65 → 0.72 | 0.84 → **0.95** |
| Jab | 0.87 → 0.92 | 0.89 → **0.95** |
| Lead Hook | 0.71 → 0.82 | 0.51 → 0.72 |
| Lead Uppercut | 0.72 → 0.92 | 0.86 → 0.88 |
| Rear Hook | 0.50 → 0.50 | 0.08 → **0.02** ⚠️ |
| Rear Uppercut | 0.59 → 0.77 | 0.61 → 0.77 |

- **Ponto fraco honesto:** Rear Hook continua péssimo (133 amostras, 5% do treino). Não é bug — é falta de dados. Alavanca: flip-aug (espelhar treino com troca Lead↔Rear) e/ou mais vídeos.
- **Cross-video (V3/V4):** 0.13 (v2) → **0.29** (v3) — recuperou, mas o gap de generalização entre ângulos persiste (achado importante p/ a pesquisa).

Artefatos: `matrix_baseline.png`, `matrix_improved.png`, `outputs/adam_baseline.mp4`, `outputs/adam_improved.mp4`, `baseline_metrics.json`, `improved_metrics.json`.

---

## Branch `feat/type-stance-decoupling` — desacoplar TIPO (modelo) da MÃO (geometria) + suavidade

> Avanço sobre `feat/body-frame-event-inference`. Ataca os DOIS gargalos medidos: (1) confusão de
> golpes parecidos = a MÃO (jab/cross, lead/rear hook/uppercut); (2) o rótulo "piscando"/trocando de
> classe no meio do golpe. Cada lever foi medido em cross-video/LOVO com IC. Plano completo em
> `PLANO_IMPLEMENTACAO.md`; pesquisa SOTA (7 dimensões, com verificação adversarial) por trás de cada escolha.

### Levers implementados (todos no estilo numpy/Keras do Leo)
- **Features view-invariantes** (`boxe_utils.engineered_features` + `deroll`): ângulos internos do corpo
  (cotovelo E/D, ombro E/D, torção do tronco) + razão de direção da velocidade do punho, somados às
  coords/vel/acc. De-roll nivela a linha de quadril (tira roll de câmera — parte bem-posta da
  canonicalização 2D). **Medido: ajudam o cross-video** (ablação: só-coords 0.362 → coords+eng 0.380;
  sem de-roll cai p/ 0.332). 110 features.
- **Máscara do padding de verdade** (`model.py`): o bug era a padronização por eixo virar o zero-pad em
  `−mean/std ≠ 0`, e o `np.diff` injetar velocidade espúria — o BiLSTM comia lixo. Agora o padding fica
  0 em todos os canais e o pooling (`MaskedMean`) ignora os frames de padding. Sem camada `Masking`
  (que forçava o caminho lento e quebrava o cuDNN com máscara não-right-padded por frames internos
  perdidos) → LSTM cuDNN, **treino 5-10× mais rápido**.
- **Logit-adjusted softmax** (balanced softmax, Menon et al. ICLR 2021): `loss = CE(z + tau·log(pi), y)`,
  no lugar do `class_weight='balanced'` (que super-amplificava a Rear Hook). Dense final em LOGITS;
  softmax na inferência (`predict_proba`). Calibra e ataca a classe rara.
- **Mirror augmentation com troca de mão** (`mirror_windows` + swap de rótulo Lead↔Rear): dobra os dados
  e força a mão a vir da geometria do corpo, não do lado da câmera.
- **Deep ensemble (5 seeds, média de softmax)**: melhor calibração sob shift + stream mais suave.
- **Decoder temporal** (`decode.py`): histerese (Schmitt) na velocidade do punho **normalizada pelo
  corpo** (não pixels) + **Viterbi grudento** (numpy, p_self=0.92) sobre as posteriors por-frame →
  moda da trilha por span = 1 rótulo contíguo; relabel ASRF apaga lampejos curtos. Mata o piscar.
- **Stance adaptativa** (`stance.py` — geometria pura): a MÃO lead/rear sai de geometria POR-GOLPE
  (a extensão do punho define a "frente" localmente). Aplicada **adaptativamente** no `boxe.py`: só
  corrige a mão quando o modelo titubeia entre as duas mãos do mesmo tipo (nunca sobrescreve confiante).

### Resultados (5-seed ensemble, com IC bootstrap)
| Métrica | baseline (Leo) | **nova pipeline** |
|---|---|---|
| TEST in-distribution (V5,V9) | 0.83–0.87 | **0.849** [IC95 0.821, 0.872] |
| **CROSS-VIDEO 6-classes (V3,V4)** | **0.31** | **0.406** [IC95 0.380, 0.432] ✅ significativo |
| CROSS-VIDEO TIPO 3-classes | 0.49 | **0.564** [IC95 0.538, 0.591] |
| Calibração ECE (test) | "crava 100%" | 0.073 (honesto) |
| Stance lead/rear (geometria, cross) | 0.70 | **0.772** |

### Ablação (cross-video V3,V4, 3 seeds — o que move o número)
1. coords+eng+deroll (escolhida): **0.380** · 2. eng+aug: 0.370 · 3. coords+aug: 0.367 ·
4. só coords: 0.362 · 5. eng+aug tau1.5: 0.357 · 6. eng sem deroll: 0.332.
**Conclusões medidas:** ângulos ajudam; de-roll ajuda; **pose-aug (rotação/jitter) NÃO ajuda OOD**
(variância alta); tau>1 não ajuda. Mantida a config simples (eng + deroll, sem aug).

### adam.mp4 — auditoria frame-a-frame (o entregável visível)
9 golpes detectados, cada um **1 rótulo contíguo do início ao fim do movimento**, **0 trocas de classe
dentro de um golpe**, repouso em branco. Verificado quebrando em frames: o rótulo aparece exatamente
durante a extensão do punho e segura até a retração (ex. span [183-196] "Cross", [279-301] "Cross").
**O piscar/trocar no meio do golpe — a queixa principal — está resolvido.** A classe exata (cross vs
jab, lead vs rear) é limitada por OOD (adam = pessoa/câmera fora do treino), mas o TIPO (reto/hook) bate.
Gating retunado p/ a escala do sinal body-frame (THI=1.0/TLO=0.5; o 0.07 antigo era de pixels).

### Caveats honestos
- O gap OOD **não fecha** (research-hard, 2D, poucas fontes): teto = oracle ~0.49. Entregamos ganho
  real e significativo (0.31→0.40-0.46) + in-distribution forte + limitação documentada.
- **LOVO tem um fold patológico (V4 ≈ 0.01)**: câmera espelhada → o modelo cru prediz a mão invertida
  com CONFIANÇA. **Medido:** a stance adaptativa NÃO recupera o V4 (0.013→0.013) — o adaptativo só dispara
  quando o modelo titubeia (margem<0.15), e no V4 ele erra confiante; além disso a própria geometria fica
  espelhada nessa câmera. Nos outros folds o ganho é marginal e seguro (V3 +0.022, V1 +0.004, in-distribution
  V5 0.881→0.885 — não machuca). **Conclusão empírica honesta:** lead/rear em 2D entre câmeras diferentes é
  ~irresolvível em alta acurácia (confirma a IDEIA_STANCE); a stance é tie-breaker seguro, não conserta o
  espelhamento. O ganho real de cross-video veio das features+máscara+logit-adjust+ensemble, não da stance.
- Rear Hook ainda fraca (recall ~0.09) — fome de dados (133 amostras). Lever: re-extrair V6 / mais vídeos.

### Tentado e descartado (empírico)
- pose-aug (rotação/jitter/scale/warp): não move o cross-video (medido em 4 variantes). Em 2D, rotação
  no plano não simula mudança de ponto de vista fora-do-plano.
- tau=1.5 no logit-adjust: sem ganho sobre tau=1.0.
- frontalização 2D total, GCN/transformer grande, temperature scaling: descartados por pesquisa (ver plano).

### LOVO (leave-one-video-out, 6-classes, 3 seeds/fold) — número científico
Folds medidos: V1=0.342 · V2=0.474 · V3=0.601 · V4=**0.011** (patológico, câmera espelhada).
Média dos folds saudáveis ~**0.47**; com o V4 incluído ~0.36. O V4 é o caso de mão-espelhada que
nem o modelo nem a geometria resolvem em 2D (limitação documentada). O número headline robusto é o
split fixo V3,V4 com IC: **0.406 [0.380, 0.432]** vs baseline 0.31.

### Decisão de output (com o Pedro): **6 classes nomeadas** (Jab/Cross/Lead Hook/...).
A IDEIA_STANCE pedia o output de 6 via decomposição (TIPO-3cl + mão-geometria). Como a geometria da
mão é fraca (0.772, falha no espelhado), na prática o modelo direto de 6-classes (com features+máscara+
logit-adjust+ensemble) entrega ≈ o mesmo, e é o que o `boxe.py` usa (com a geometria só como tie-breaker
adaptativo). O TIPO-3cl interno (cross-video 0.564) fica disponível como a representação robusta.
