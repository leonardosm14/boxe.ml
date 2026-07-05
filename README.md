# Detecção e Classificação de Golpes em Boxe

> **Aviso de uso de IA:** este projeto utilizou assistentes de IA generativa como
> ferramenta de apoio (pesquisa, boilerplate, depuração e revisão). O compilado de
> prompts e o detalhamento do uso estão em [`PROMPTS.md`](PROMPTS.md).

Pipeline para reconhecimento automático de golpes de boxe utilizando **YOLOv8 Pose** para extração de esqueletos humanos e **TensorFlow** para classificação automática dos golpes.

Os golpes atualmente suportados são:

* Jab
* Cross
* Lead Hook
* Lead Uppercut
* Rear Hook
* Rear Uppercut

O sistema recebe um vídeo como entrada, extrai os keypoints corporais utilizando o modelo de pose do YOLOv8 (com tracking ByteTrack para manter a identidade de cada lutador), realiza o pré-processamento dos esqueletos e aplica um modelo LSTM treinado em TensorFlow para identificar o **tipo** do golpe (Straight/Hook/Uppercut). A **mão** do golpe (lead vs rear) é decidida em seguida por geometria pura sobre os keypoints (`stance.py`), expandindo as 3 classes do modelo para as 6 classes finais (`stance_utils.py`).

> **Como funciona a divisão 3+lead/rear:** o tipo do golpe é aprendível da
> trajetória, mas a mão é ambígua numa janela isolada. Por golpe detectado, o
> punho de maior deslocamento líquido define a mão que golpeou e a direção de
> extensão define a "frente" local — o pé desse lado é o lead. Medido no
> BoxingVI: 0.85 (treino) / 0.77 (cross-video) / 0.74 (teste) de acurácia
> lead/rear; no nosso vídeo próprio anotado (adam): 0.89 por segmento.
> Validação: `python3 stance.py` (gate no dataset) e `python3 eval_leadrear.py`
> (end-to-end contra `adam_gt.csv`). O treino do modelo de tipo está em
> `training.ipynb`.

---

# Instalação

## 1. Clonar o projeto

```bash
git clone <repositorio>
cd boxe.ml
```

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## Modelo treinado

Por padrão o script procura por `modelo_boxe.keras` na raiz do projeto. Também é possível especificar outro modelo utilizando o parâmetro `--model` ou (`-m`).

---

# Executando a Inferência

## Exemplo básico

```bash
python3 boxe.py --video videos/exemplo.mp4
```

O vídeo processado será salvo automaticamente em:

```text
outputs/exemplo.mp4
```

---

## Especificando um modelo customizado

```bash
python3 boxe.py --video videos/exemplo.mp4 --model modelos/meu_modelo.keras
```

---

## Especificando o vídeo de saída

```bash
python3 boxe.py --video videos/exemplo.mp4 --output outputs/
```

---

## Limpando o cache de esqueletos

Durante a primeira execução, os esqueletos extraídos pelo YOLO são armazenados em cache para acelerar execuções futuras.

Para forçar uma nova extração:

```bash
python3 boxe.py --video videos/exemplo.mp4 --clear-cache
```

---

# Pipeline de Processamento

O processamento ocorre em três etapas principais.

## 1. Extração de Esqueletos

O YOLOv8-Pose detecta o atleta e extrai:

* 17 keypoints corporais (formato `COCO`)
* Coordenadas normalizadas pela resolução do vídeo
* Rastreamento temporal dos movimentos

Os esqueletos são armazenados em cache no formato:

```text
tdets_<nome_do_video>.npy
```

---

## 2. Classificação dos Golpes

Para cada frame:

* Uma janela temporal de 25 frames é construída
* Os keypoints são normalizados
* São calculadas posições, velocidades e acelerações

Esses dados são enviados para o modelo TensorFlow treinado.

---

## 3. Geração do Vídeo

O sistema:

* Aplica suavização temporal das predições
* Filtra predições com baixa confiança
* Sobrepõe os rótulos dos golpes no vídeo
* Exporta o resultado em H.264

---

# Configuração TensorFlow + GPU (H100)

## Problema

Em alguns ambientes, o TensorFlow pode carregar uma versão incompatível do CuDNN:

```text
Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0.
Dnn is not supported
```

## Instalar CuDNN 9.3

```bash
pip install nvidia-cudnn-cu12==9.3.0.75
```

## Criar links simbólicos

```bash
CUDNN_SRC=/root/anaconda3/envs/lapixdl/lib/python3.11/site-packages/nvidia/cudnn/lib
CUDNN_DST=/lib/x86_64-linux-gnu

ln -sf $CUDNN_SRC/libcudnn.so.9                          $CUDNN_DST/libcudnn.so.9
ln -sf $CUDNN_SRC/libcudnn_adv.so.9                      $CUDNN_DST/libcudnn_adv.so.9
ln -sf $CUDNN_SRC/libcudnn_cnn.so.9                      $CUDNN_DST/libcudnn_cnn.so.9
ln -sf $CUDNN_SRC/libcudnn_graph.so.9                    $CUDNN_DST/libcudnn_graph.so.9
ln -sf $CUDNN_SRC/libcudnn_heuristic.so.9                $CUDNN_DST/libcudnn_heuristic.so.9
ln -sf $CUDNN_SRC/libcudnn_ops.so.9                      $CUDNN_DST/libcudnn_ops.so.9
ln -sf $CUDNN_SRC/libcudnn_engines_precompiled.so.9      $CUDNN_DST/libcudnn_engines_precompiled.so.9
ln -sf $CUDNN_SRC/libcudnn_engines_runtime_compiled.so.9 $CUDNN_DST/libcudnn_engines_runtime_compiled.so.9
```

## Verificação

```bash
ldconfig -p | grep cudnn
```

Os caminhos exibidos devem apontar para:

```text
/root/anaconda3/envs/lapixdl/...
```

## Reiniciar o ambiente

Após criar os links simbólicos, reinicie o terminal ou kernel do Jupyter para garantir que o TensorFlow carregue as bibliotecas corretas.

---

# Ambiente de Referência

| Componente     | Versão                     |
| -------------- | -------------------------- |
| TensorFlow     | 2.21.0                     |
| Ultralytics    | 8.x                        |
| CuDNN          | 9.3.0.75                   |
| CUDA Toolkit   | 12.4                       |
| CUDA Driver    | 12.8                       |
| Driver NVIDIA  | 570.195.03                 |
| Python         | 3.11                       |
| Ambiente Conda | `lapixdl`                  |
| GPU            | NVIDIA H100 80GB HBM3 (8x) |

---

# Observações

Na primeira execução, o YOLOv8 fará automaticamente o download do modelo:

```text
yolov8m-pose.pt
```

Esse download ocorre apenas uma vez e o arquivo será armazenado no cache do Ultralytics para reutilização em execuções futuras.
