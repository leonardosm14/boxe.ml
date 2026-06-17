# Ideia: separar TIPO do golpe (modelo) da MÃO lead/rear (geometria)

> Documento autossuficiente e factual. Marca claramente **[MEDIDO]** (rodei o experimento) vs
> **[PROPOSTO]** (a fazer). Feito para ser lido em contexto limpo, sem depender de histórico.

---

## 1. Contexto do projeto (mínimo necessário)
- Tarefa: classificar **golpes de boxe** em vídeo. Pipeline: YOLOv8-Pose → 17 keypoints COCO por
  frame → janelas de 25 frames → modelo TensorFlow (BiLSTM + atenção) → classe do golpe.
- **6 classes:** `Cross, Jab, Lead Hook, Lead Uppercut, Rear Hook, Rear Uppercut`.
- Dataset: BoxingVI (10 vídeos). Splits usados: TRAIN=V1,V2,V7,V8 · TEST=V5,V9 (mesma fonte) ·
  CROSS-VIDEO=V3,V4 (held-out, proxy de generalização/OOD).
- Estado do modelo atual (com normalização relativa ao corpo): **[MEDIDO]** TEST 0.871,
  CROSS-VIDEO 0.311 (6 classes).

## 2. O problema que esta ideia ataca
"Golpes da mesma família só mudam a MÃO": **Jab vs Cross** (reto, mão da frente vs de trás),
**Lead vs Rear Hook**, **Lead vs Rear Uppercut**. A diferença é qual mão golpeou, que depende da
**guarda (stance)**: a mão "da frente" (lead) é a do mesmo lado do **pé que está na frente**.
- Orthodox (destro) = pé esquerdo na frente → mão esquerda é a lead.
- Southpaw (canhoto) = pé direito na frente → mão direita é a lead.

## 3. A ideia (decomposição)
Em vez de o modelo aprender 6 classes (e ter que adivinhar a mão, que de uma janela isolada é
ambíguo), **separar em dois sub-problemas**:
1. **TIPO** (3 classes: `reto / hook / uppercut`) → o modelo aprende isso (é o que ele faz bem e
   é invariante à mão).
2. **MÃO** (lead vs rear) → resolver por **geometria** do esqueleto (qual pé está na frente +
   qual mão golpeou), sem treinar.

**Mapeamento das 6 classes:**
| | mão da FRENTE (lead) | mão de TRÁS (rear) |
|---|---|---|
| **reto** | Jab | Cross |
| **hook** | Lead Hook | Rear Hook |
| **uppercut** | Lead Uppercut | Rear Uppercut |

3-classes (TIPO): `reto={Jab,Cross}`, `hook={Lead Hook,Rear Hook}`, `uppercut={Lead Uppercut,Rear Uppercut}`.
Conjunto LEAD = {Jab, Lead Hook, Lead Uppercut}. Conjunto REAR = {Cross, Rear Hook, Rear Uppercut}.

## 4. Evidência empírica [MEDIDO]
Tudo medido com o modelo atual (normalização relativa ao corpo) nas mesmas janelas anotadas.

**A confusão É a mão (confirma o diagnóstico):**
| | 6-classes direto | TIPO (acerto do tipo) | híbrido (tipo+geometria) | **oracle (tipo+mão verdadeira)** |
|---|---|---|---|---|
| TEST (V5,V9) | 0.871 | 0.871 | 0.612 | 0.871 |
| CROSS-VIDEO (V3,V4) | 0.311 | **0.493** | 0.348 | **0.493** |

Leitura:
- No CROSS-VIDEO o modelo acerta o **TIPO 0.493** mas o 6-classes cai pra **0.311** → **a mão é
  ~metade do erro OOD**. Com a mão perfeita (oracle), o OOD saltaria **0.311 → 0.493 (+18pp)**.
- No TEST o modelo já acerta a mão sozinho (direto = tipo = oracle = 0.871) → in-distribution a
  geometria só atrapalha (0.612).

**Acurácia da MÃO por geometria (regra crua, por janela):** [MEDIDO]
- TRAIN 0.807 · CROSS-VIDEO 0.697 · TEST 0.714.
- **Pé/perna visível em 100% das janelas** (viável). (Tentativa de stance por-clipe ficou em
  0.63–0.70 por bug meu; o número confiável é o por-janela ~0.70–0.81.)

**Conclusão dos números:** o potencial é real e grande (+18pp no OOD), mas a geometria a ~0.75 é o
**gargalo** — nesse nível o ruído quase anula o ganho (híbrido 0.348 vs direto 0.311). Para realizar
os +18pp, a geometria da mão precisa chegar a **~0.90**.

## 5. Regra geométrica usada (a versão medida, ~0.75) [MEDIDO]
Índices COCO-17: `0 nariz · 5/6 ombro E/D · 7/8 cotovelo E/D · 9/10 punho E/D · 11/12 quadril E/D ·
13/14 joelho E/D · 15/16 tornozelo E/D` (E=esquerdo, D=direito).
1. **Mão que golpeou** = punho (9 esq. / 10 dir.) com **maior caminho** (soma do deslocamento) na janela.
2. **Direção "frente"** = sinal do deslocamento em x do punho que golpeou (origem → ponto de maior extensão).
3. **Pé da frente** = tornozelo (15 esq. / 16 dir.; joelhos 13/14 como backup) cujo x está **mais
   na direção da frente**.
4. **Mão lead** = mesmo lado do pé da frente.
5. **Predição:** se (lado da mão que golpeou == lado da mão lead) → **Lead**; senão → **Rear**.

## 6. Por que a regra falha às vezes (e como subir 0.75 → ~0.90) [PROPOSTO]
- **Hooks vão pro LADO, não pra frente** → o passo 2 ("direção da frente" pela extensão do punho)
  é ruim em hooks. Determinar a "frente" pela **geometria do corpo** (ex.: lado que o nariz/ombros
  apontam) ou só pelos **golpes retos** do clipe, não pela extensão de cada golpe.
- **Stance é constante no clipe** → estimar a guarda **uma vez por clipe** (agregando todos os
  frames), não por janela. (Mais robusto; minha tentativa bugou, mas é o caminho.)
- **Usar a perna toda** (quadril→joelho→tornozelo), não só o pé (oclusão).
- **Aplicar adaptativo:** usar a geometria **só quando o modelo está incerto** entre Lead/Rear da
  mesma família (in-distribution o modelo acerta sozinho — não sobrescrever).

## 7. Bônus de DADOS (separado, e real) [PROPOSTO]
Treinar o TIPO em 3 classes **junta Lead+Rear** → **triplica as amostras por classe**. Isso ataca
direto a fome de dados do **Rear Hook** (só 133 amostras = 5.3% do treino, recall ~0.10–0.14):
ele se funde em "hook" (~400+ amostras). O mirror augmentation (espelho) é consistente com 3 classes
(um hook continua hook ao espelhar), então **dobra de novo**.

## 8. Plano de implementação [PROPOSTO]
1. Treinar o modelo de **TIPO (3 classes)** com a mesma pipeline (normalização relativa ao corpo +
   mirror aug, agora com label idêntico no espelho). Medir TIPO em TEST e CROSS-VIDEO com CI.
2. Implementar **stance robusta por clipe** (seção 6) e medir a acurácia da mão (alvo ~0.90).
3. Combinar: `6-classes = MAP(tipo_do_modelo, mão_da_geometria)`. Medir vs o 6-classes direto, em
   TEST e CROSS-VIDEO, **com intervalo de confiança** e os pisos (majoritário/aleatório).
4. Aplicar a geometria **adaptativamente** (só em incerteza/OOD).
5. Critério de sucesso (empírico): CROSS-VIDEO sobe de 0.311 em direção ao oracle 0.493, sem
   derrubar o TEST; Rear Hook recall sai de ~0.10.

## 9. Caveats honestos
- O **+18pp é o teto (oracle)**, com mão perfeita. O ganho real depende de quão perto a stance
  chega de 0.90. A 0.75 o ganho é marginal (+3.7pp medido).
- **Não sobrescrever a mão in-distribution** — lá o modelo já acerta; a geometria piora (0.871→0.612).
- Handedness (orthodox/southpaw) sai **de graça** lendo o pé real; troca de guarda no meio é tratada
  por golpe. Sem suposição de "todos destros".
- Isto **não resolve** o gap de generalização inteiro — só a parte que é mão (~metade). A outra
  metade é o TIPO não generalizar (representação/dados), atacada por outros itens (features de
  mecânica, mais dados diversos).
