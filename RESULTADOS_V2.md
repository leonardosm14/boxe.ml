# Resultados V2 — pipeline geométrica de golpes (honesto, medido)

> Sessão de refatoração+pesquisa. **Tudo aqui é medido contra os gabaritos** (adam_gt.csv,
> ruy.csv). Sem alucinar números: onde é teto físico, está dito e provado.

## TL;DR

- **Bug raiz da sessão anterior, achado e corrigido:** a extração de pose **zerava juntas
  de baixa confiança** → no adam (perfil lateral) o punho de TRÁS sumia em 48% dos frames →
  metade do erro. `pose.py` agora mantém a posição estimada (yolo11x): cobertura **52%→99%**.
- **Método novo:** decomposição **TIPO (geometria) + MÃO (geometria de stance)** → 6 classes.
  Sem modelo treinado na classificação (o BiLSTM do BoxingVI não generaliza p/ outro vídeo —
  medido: TIPO 0.44 no adam).
- **Vídeo do adam:** detecção **18/18**, **1 rótulo contínuo por golpe (sem piscar)**.

## Resultados (medidos)

| | adam (lateral) | ruy (frontal) |
|---|---|---|
| **TIPO** (reto/hook/upp) | **0.78** | 0.43 |
| **MÃO** (lead/rear) | **0.89** | 0.45 |
| **6-classes (spans do GT = qualidade do método)** | **0.72** | 0.20 |
| 6-classes ponta-a-ponta (detecção automática) | 0.61 | 0.07 |
| detecção temporal | 18/18 | 26/60 |

Comparação: modelo BiLSTM (BoxingVI) no adam = **0.44**. Sessão anterior = "errou todos".

## O teto do reto-vs-hook é FÍSICO (4 métodos, mesma parede)

A confusão **reto↔hook** é o gargalo (o Leo mede o mesmo: hook recall 17–48% no BoxingVI).
Não é falta de esforço — é a vista 2D:

- 2D mecânica, cross-rep no adam: **0.56** TIPO.
- 3D MediaPipe (BlazePose): **0.17** (punho de trás 3D tem visibilidade 0.30).
- BiLSTM BoxingVI no adam (OOD): **0.44**.
- RF mecânica, LOVO cross-video no BoxingVI (4766 golpes): **0.55**.

**Causa:** na vista LATERAL (adam) o arco horizontal do hook se projeta na PROFUNDIDADE
(invisível em 2D, mal-estimado em 3D mono). Na vista FRONTAL (ruy) é o oposto: a MÃO some (o
soco estende na profundidade → MÃO 0.45) e o hook fica visível. **Cada vista esconde um eixo
diferente.** Uma única câmera 2D nunca tem os dois.

## Codebase nova (limpa, estilo numpy/PT do Leo)

| módulo | o quê |
|---|---|
| `pose.py` | extração yolo11x, mantém juntas (a correção) |
| `mechanics.py` | features de mecânica do golpe (view-invariantes) |
| `detect.py` | detecção por pico de EXTENSÃO (1 golpe = 1 pico de alcance) |
| `stance.py` | MÃO lead/rear por geometria de guarda |
| `classify.py` | TIPO (gate uppercut + arco hook) + MÃO → 6 classes |
| `infer.py` | vídeo → detecção → classificação → render (1 rótulo contínuo) |
| `eval.py` | avaliação vs gabarito (método + ponta-a-ponta) |

Rodar: `python infer.py -v annotate/adam_clean.mp4` · medir: `python eval.py -s skel.npy -g gt.csv`

## Recomendações ao grupo (pra passar do teto)
1. **Vista frontal/3-4** (não lateral pura) p/ os vídeos → o hook fica visível. adam é lateral
   = o caso pior pro reto-vs-hook.
2. **Mais dados de hook** (rear hook é o mais faminto). Mirror aug ajuda, não resolve.
3. **Multi-câmera** (lateral resolve MÃO, frontal resolve TIPO) seria o ideal.
4. MÃO no adam (94%) é nossa contribuição forte — combinar com o modelo de TIPO do Leo.
