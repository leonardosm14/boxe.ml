# Detecção e Classificação de Golpes em Boxe

Pipeline para reconhecimento automático de golpes de boxe utilizando **YOLOv8 Pose** para extração de esqueletos humanos e **TensorFlow** para classificação automática dos golpes.

Os golpes atualmente suportados são:

* Jab
* Cross
* Lead Hook
* Lead Uppercut
* Rear Hook
* Rear Uppercut

O sistema recebe um vídeo como entrada, extrai os keypoints corporais utilizando o modelo de pose do YOLOv8, realiza o pré-processamento dos esqueletos e aplica um modelo treinado em TensorFlow para identificar os golpes ao longo do vídeo.

> **Atualizações** (detalhes em [`MUDANCAS.md`](MUDANCAS.md), próximos passos em [`PLANO_ACAO.md`](PLANO_ACAO.md)):
> a normalização passou a ser **relativa ao corpo** (recentro no quadril + escala de ombro,
> invariante a posição/escala/câmera) e a inferência passou a ser **por evento** (segmentação
> por movimento do punho → 1 rótulo estável por golpe). O modelo é treinado por `train.py`.

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
python3 boxe.py \
    -v videos/exemplo.mp4
```

O vídeo processado será salvo automaticamente em:

```text
outputs/exemplo.mp4
```

---

## Especificando um modelo customizado

```bash
python3 boxe.py \
    -v videos/exemplo.mp4 \
    -m modelos/meu_modelo.keras
```

---

## Especificando o vídeo de saída

```bash
python3 boxe.py \
    -v videos/exemplo.mp4 \
    -o resultado.mp4
```

---

## Limpando o cache de esqueletos

Durante a primeira execução, os esqueletos extraídos pelo YOLO são armazenados em cache para acelerar execuções futuras.

Para forçar uma nova extração:

```bash
python3 boxe.py \
    -v videos/exemplo.mp4 \
    --clear-cache
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
skeletons_<nome_do_video>.npy
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
