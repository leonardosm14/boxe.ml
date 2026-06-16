# Plano de Ação — estado real + próximos passos (rumo à quase-perfeição, com empirismo)

> Documento vivo. Reflete o que **já foi feito e validado** e o que penso **agora**, depois de
> rodar os experimentos. Cada item: **problema → evidência medida → ação → impacto → como medir**.

---

## TL;DR estratégico (minha opinião sincera)
- **Reaproveitar a base do Leo, não recomeçar do 0.** A arquitetura (pose → janelas de esqueleto
  → modelo temporal → classifica) é a correta pro problema. O avanço é **incremental**, não reescrita.
- O gargalo real **não é a arquitetura** — é **generalização (OOD/cross-video)**, que depende de
  **dados diversos + features + rigor de avaliação**, tudo enxertável no código atual.
- **Expectativa honesta:** OOD **não vai ser "resolvido"** com poucas fontes (research-hard). Alvo
  realista: **demo forte in-distribution (vídeo do Ruy) + números cross-video com intervalo de
  confiança + limitação documentada.** Isso já é um trabalho de graduação excelente.

---

## Estado atual (medido)

| Métrica | Inicial (Leo) | **Atual (commitado)** |
|---|---|---|
| Test V5/V9 (same-source) | 73.4% | **87.2%** |
| Macro-F1 | 0.62 | **0.78** |
| **Cross-video V3/V4 (OOD)** | 0.18 | **0.31** (3-classes: 0.49) |
| Fake-100% (>99% conf) | 18.7% | **0.0%** |
| Inferência | per-frame (piscava) | **por evento, smooth** |
| Rear Hook recall | 0.08 | 0.14 (ainda fraco) |

**Já feito e validado (no commit):**
- ✅ **Normalização relativa ao corpo** (recentro no quadril + escala de ombro). Foi o lever #1: melhorou same-source E cross-video. Invariância a escala/translação verificada (diff ~1e-6).
- ✅ **Inferência por evento** (segmentação por movimento do punho, Schmitt trigger → 1 rótulo por golpe, do snap ao follow-through). Matou: piscar, troca de classe no meio, golpe em repouso.
- ✅ **Calibração** (label smoothing) — o "crava 100%" sumiu.
- ✅ **Mirror augmentation** no referencial do corpo (espelha + troca Lead/Rear).
- ✅ Bugs base: janela OOD, predict duplicado, norm dessincronizada, LR mode, GPU hardcoded — todos corrigidos.

**Tentado e que FALHOU (empírico):**
- ❌ **Incluir o V6** (46k frames crus) no treino: está em **pixels/qualidade ruim** e **contaminou** (derrubou o test pra 60%). A ideia "mais dados do V6" só vale **re-extraindo com YOLO** (próximos passos).

---

## As causas-raiz (atualizadas: quais foram atacadas, quais restam)

1. ✅ **Features absolutas** → **RESOLVIDO em grande parte** pela norm relativa ao corpo (cross-video 0.18→0.31).
2. ⚠️ **Classes minoritárias famintas** → V6/V10 tentado, V6 contaminou. **Resta** re-extrair V6 com YOLO + class-balance.
3. 🔴 **Faltam features de mecânica** (ângulo de cotovelo, vetores ósseos) → **não atacado ainda**. É o próximo lever de modelo.
4. 🔴 **Sem stance anchor** → o cross-video colapsado em 3-classes é **0.49** vs 6-classes **0.31**: ~metade do gap é **Lead/Rear espelhado** (resolvível com detecção de stance). **Não atacado.**
5. 🔴 **Avaliação não separa sinal de ruído** → **não atacado**. Promovi para a PRIORIDADE #1 (abaixo).

---

## Lista de ação RE-PRIORIZADA (meu pensamento agora)

### 🥇 #1 — Rigor de avaliação PRIMEIRO (LOVO + CI + métricas de detecção)
- **Por que subiu pra #1:** estamos otimizando no escuro. O ganho do flip (+2.3pp) ficou **dentro do ruído** (Wilson ±2.6pp em n=747). Até as minhas melhorias precisam de CI pra confirmar. Sem isso, não dá pra saber o que ajuda.
- **Ação:** loop de 5-10 seeds + CI bootstrap 1000x; harness **leave-one-video-out** (treina em 9, testa no 10º, roda); sempre imprimir piso majoritário (~0.32) e aleatório (0.167); McNemar pareado pra comparar modelos. Para a inferência por evento: métricas de **detecção** (P/R/F1 + IoU temporal contra os start/end dos CSVs).
- **Impacto:** credibilidade (não muda o número, mas faz todo número virar confiável). **Esforço:** horas (seeds+CI) a dias (LOVO+detecção).
- **Medir:** cada número passa a ter média±desvio + CI; decisões param de ser ruído.

### 🥈 #2 — Features de mecânica + stance (o lever de modelo pro OOD)
- **Evidência:** confusões são mecânicas (Rear Hook↔Rear Uppercut, mesmo braço); e 3-classes (0.49) >> 6-classes (0.31) → ~metade do gap cross-video é Lead/Rear.
- **Ação:** (a) anexar por frame **ângulo de cotovelo** (juntas 7,8), **vetores ósseos** (direção distal-proximal), **rotação do tronco** — re-treinar; (b) **detecção de stance** por-clipe (x de tornozelo/ombro) pra resolver Lead/Rear deterministicamente.
- **Impacto:** médio-grande em macro-F1 e na fronteira Hook/Uppercut; stance ataca direto ~metade do gap cross-video. **Esforço:** dias.
- **Medir:** células Rear Hook→Rear Uppercut e Lead Hook→Jab encolhem; cross-video 6-classes aproxima do 3-classes (0.49).

### 🥉 #3 — Dados diversos (o teto de verdade)
- **Evidência:** cross-video ~piso majoritário; fonte única = overfit. Research confirmou: **não existe dataset drop-in** com nossos 6 tipos (BoxMAC retirado, Olympic é binário).
- **Ação:** (a) **vídeo do Ruy** (lateral, setup parecido) como **2ª fonte real** + demo in-distribution *verificável*; (b) **re-extrair o V6 com YOLO-Pose** (normalizado [0,1], sem o lixo de pixel) → recupera Rear Hook +95 sem contaminar; (c) **classe Background** minerada dos gaps do V6 cru → mata falso-positivo de repouso de verdade; (d) opcional: Olympic re-rotulado (~centenas de golpes) como 2ª fonte.
- **Impacto:** o maior teto, mas o mais caro. Ruy = melhor relação esforço/demo. **Esforço:** dias (Ruy/V6) a semanas (Olympic/SSL).
- **Medir:** acurácia no vídeo do Ruy (in-distribution, verificável); Rear Hook recall com CI excluindo 0.14; falso-positivo de repouso medido (não assumido).

### #4 — Tracking multi-lutador (Tom já está nisso)
- Ler `r.boxes.id` (persist=True, bytetrack/botsort), travar no lutador por movimento de punho, ignorar juiz. Roda detect_events por track. **Impacto:** desbloqueia luta real. **Medir:** trocas de ID num clipe de 2 pessoas.

### #5 — Robustez de inferência (escala/fps) + housekeeping
- Velocidade do punho normalizada por torso; SEG_THI/TLO por percentil do clipe; thresholds em segundos×fps nativo; interpolação linear no lugar de hold-last. Checar **vazamento de sujeito** (mapa vídeo→lutador). Masking layer + excluir padding das stats. **Impacto:** médio (robustez em clipes fora de 25fps). **Esforço:** horas.

---

## Datasets externos — conclusão (não mudou)
Sem drop-in com os 6 tipos: **BoxMAC retirado**, **Olympic binário** (precisa re-rotular), NTU/MMA só pré-treino/pose. **O dado mais rico que temos é nosso** (V6, re-extraído) + o vídeo do Ruy. Detalhe na seção do research no histórico.

---

## Roadmap (reuso da base do Leo, incremental)
1. **Merge do PR atual** → branch do Leo vira a base nova.
2. **Sprint 1 — Honestidade:** LOVO + seeds + CI + métricas de detecção (#1). *Antes de qualquer novo claim.*
3. **Sprint 2 — Modelo:** features de mecânica + stance (#2). Um retreino, mede contra o #1.
4. **Sprint 3 — Dados:** vídeo do Ruy + re-extrair V6 + classe Background (#3).
5. Paralelo: tracking (#4, Tom), robustez (#5).
- Fluxo git: feature branch por item, sempre comparando com CI contra a base.

## Princípio de validação empírica
Toda ação amarrada a **métrica pré-registrada + piso sem-skill (majoritário ~0.32 / aleatório 0.167)**, medida **antes/depois nas mesmas seeds, com CI**. Recall por classe (com CI) pra classe rara; **LOVO + balanced acc** pra generalização; células da matriz pra features. **Nada entra como "melhora" se estiver dentro do CI.**
