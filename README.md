# Configuração do Ambiente — TensorFlow + GPU (H100)

Instruções para resolver o conflito de versão do CuDNN que impede o TensorFlow de usar a GPU.

## Problema

O sistema possui CuDNN 9.1 instalado em `/lib/x86_64-linux-gnu/`, mas o TensorFlow 2.21 foi compilado com CuDNN 9.3. O TF carrega a versão do sistema (9.1) antes da versão correta instalada pelo pip, resultando no erro:

```
Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0.
Dnn is not supported
```

## Solução

Substitua as libs do sistema por symlinks apontando para a versão 9.3 instalada no ambiente conda.

### 1. Instalar o CuDNN 9.3 via pip (se ainda não instalado)

```bash
pip install nvidia-cudnn-cu12==9.3.0.75
```

### 2. Criar os symlinks

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

### 3. Verificar

```bash
ldconfig -p | grep cudnn
```

Todos os caminhos devem apontar para `/root/anaconda3/envs/lapixdl/...`.

### 4. Reiniciar o kernel do Jupyter

Após os symlinks, reinicie o kernel.

## Ambiente

| Componente | Versão |
|---|---|
| TensorFlow | 2.21.0 |
| CuDNN (runtime) | 9.3.0.75 |
| CUDA toolkit | 12.4 |
| Driver NVIDIA | 570.195.03 |
| CUDA (driver) | 12.8 |
| Python | 3.11 |
| Conda env | `lapixdl` |
| GPU | NVIDIA H100 80GB HBM3 (×8) |