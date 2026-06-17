# Plano V2 — reconhecimento de golpes (adam + ruy), com base em MEDIÇÕES

> Escrito após diagnóstico empírico com os dois gabaritos (adam_gt.csv, ruy.csv).
> Marca **[MEDIDO]** vs **[A FAZER]**. Honesto sobre o que é teto físico.

## 1. Diagnóstico (tudo medido contra os gabaritos)

**Bug raiz da sessão passada [MEDIDO]:** a extração de pose **zerava juntas de baixa
confiança**. No adam (perfil lateral esquerdo) isso apagou o punho de TRÁS (direito) em
**48% dos frames** → geometria de mão virava lixo (metade do erro). Corrigido em `pose.py`:
manter a posição estimada do YOLO (yolo11x), guardar a confiança à parte. Cobertura do
punho de trás **52% → 99%**.

**Decomposição da tarefa:** 6 classes = TIPO (reto/hook/uppercut) × MÃO (lead/rear).

| Sub-problema | Método | adam (lateral) | ruy (frontal) |
|---|---|---|---|
| Detecção temporal | pico de velocidade do punho | 18/18 [MEDIDO] | a medir |
| MÃO (lead/rear) | geometria de stance | **94%** [MEDIDO] | **43%** [MEDIDO] — falha |
| TIPO uppercut | gate geométrico (vertical + cotovelo) | **~100%** [MEDIDO] | a medir |
| TIPO reto-vs-hook | — | **~56% (teto)** [MEDIDO] | melhor (arco visível) |

**O teto do reto-vs-hook no adam é FÍSICO, não falta de esforço [MEDIDO em 4 métodos]:**
- 2D mecânica, cross-rep (treina 2 reps, testa 1): **0.56** TIPO (uppercut limpo, reto/hook moeda).
- 3D MediaPipe (BlazePose world landmarks): **0.17** — pior; punho de trás 3D tem visibilidade 0.30.
- Modelo BiLSTM BoxingVI (OOD) no adam: TIPO **0.44**.
- Leo (BiLSTM, BoxingVI in-dist): hook recall **17–48%**.

**Causa:** numa vista LATERAL, o arco horizontal do hook se projeta no eixo de
PROFUNDIDADE (toca a câmera), invisível em 2D e mal-estimado em 3D monocular. Numa vista
FRONTAL (ruy) ocorre o inverso: a MÃO some (o soco estende na profundidade) e o hook
aparece. **Cada vista esconde um eixo diferente.**

## 2. Targets honestos

- **adam (lateral):** MÃO ~95% + uppercut ~100% + reto-hook ~60-70% → **6-classes ~75-82%**.
  É ~2× o modelo anterior (0.44). near-100% **não é alcançável** desta footage lateral mono.
- **ruy (frontal):** TIPO melhor; MÃO é o gargalo. Alvo após corrigir MÃO frontal: a medir.
- Vídeo: **rótulo contíguo por golpe, sem piscar** (decoder por pico, já funciona).

## 3. Arquitetura (decomposta, view-adaptativa)

```
vídeo → pose.py (yolo11x, mantém juntas) → detect (pico) → por golpe:
   TIPO = uppercut-gate(geom) || classificador reto/hook
   MÃO  = geometria stance (lateral) || modelo/heurística (frontal)
   6cls = MAP(TIPO, MÃO)
→ decode suave (1 rótulo contíguo) → render
```

Módulos: `pose.py`✓ `mechanics.py`✓ `stance.py`(mão) `detect.py`(pico) `classify.py`(tipo+mão→6cls)
`decode.py`(suave) `infer.py`(orquestra+render) `train.py`(H100) `eval.py`(vs GT).

## 4. Passos

1. [✓] pose.py (correção), mechanics.py (features), diagnóstico medido.
2. [A FAZER] detect.py + classify.py + infer.py + eval.py limpos (estilo Leo, ponytail).
3. [A FAZER] medir baseline decomposto em adam + ruy (eval.py).
4. [A FAZER] maximizar: MÃO frontal (ruy), ensemble reto/hook, tune do uppercut-gate.
5. [A FAZER] (H100) retreinar modelo com pose corrigida + aug forte de hook; comparar.
6. [A FAZER] render adam + ruy, **verificar frame-a-frame** (sem piscar).
7. [A FAZER] docs honestos (teto + recomendações), commit, PR, avisar Leo.

## 5. Recomendações ao grupo (pra subir além do teto)
- Gravar adam/treino em vista **frontal ou 3/4** (não lateral pura) → hook visível.
- Mais dados de **hook** (rear hook é o mais faminto) — mirror aug ajuda mas não resolve.
- Multi-câmera (lateral p/ mão + frontal p/ tipo) seria o ideal.
