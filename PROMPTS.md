# Uso de Inteligência Artificial no Projeto

> **Aviso de uso de IA.** Este projeto utilizou assistentes de IA generativa
> (Claude e ChatGPT) como ferramenta de apoio durante o desenvolvimento, nos
> moldes permitidos pela disciplina. As IAs foram usadas para: pesquisa e
> levantamento bibliográfico, geração de código-base (boilerplate), depuração
> de erros de ambiente (CUDA/CuDNN), refatoração e revisão de código, e apoio
> à escrita/formatação do relatório. **Todas as decisões de projeto, a
> modelagem, a anotação do dataset, os experimentos e a validação dos
> resultados foram feitos e conferidos pelos integrantes do grupo.**

Abaixo, um compilado representativo dos prompts utilizados ao longo do
trabalho, organizados por etapa.

## 1. Pesquisa e definição do tema

```text
Quais trabalhos e datasets públicos existem para detecção e classificação de
golpes de boxe (jab, cross, hook, uppercut) usando visão computacional e pose
estimation? Liste papers com código disponível.
```

```text
Explique como funciona o YOLOv8-Pose e o formato dos 17 keypoints COCO.
Quais as limitações para vídeos com duas pessoas se movimentando (oclusão)?
```

## 2. Dataset e pré-processamento

```text
Tenho um dataset (BoxingVI) com esqueletos por golpe em janelas de 25 frames
(17 keypoints x,y normalizados). Como normalizar os keypoints para ficarem
invariantes à posição e à escala do lutador na imagem? Sugira um referencial
centrado no corpo.
```

```text
Escreva uma função em numpy que adiciona velocidade e aceleração como features
a partir das posições dos keypoints ao longo da janela temporal.
```

## 3. Modelo e treinamento

```text
Para classificar golpes de boxe a partir de sequências de esqueletos (janelas
de 25 frames), qual arquitetura é mais indicada: LSTM, GRU ou 1D-CNN? O golpe
depende da trajetória, não de um frame isolado.
```

```text
Meu modelo LSTM está com overfitting no dataset de golpes (acurácia de treino
alta, validação baixa). Sugira técnicas de regularização e aumentação de dados
adequadas para sequências de keypoints.
```

## 4. Tracking de dois lutadores

```text
Preciso rastrear dois lutadores de boxe em um vídeo mantendo a identidade de
cada um mesmo quando eles trocam de lado ou se ocluem. O YOLO tem tracking
integrado (ByteTrack)? Como usar os track IDs do model.track() da ultralytics?
```

```text
Refatore este código para separar a lógica de tracking (extração de esqueletos,
atribuição de identidade e preenchimento de lacunas) em um módulo dedicado,
mantendo o comportamento idêntico.
```

## 5. Inferência lead/rear (3 → 6 classes)

```text
O modelo classifica o TIPO do golpe (reto, hook, uppercut), mas preciso saber
se foi com a mão da frente (lead) ou de trás (rear) para diferenciar jab de
cross. Dá para decidir isso por geometria pura sobre os keypoints, sem treinar
outro modelo? A câmera é lateral e o YOLO alucina o lado ocluído.
```

```text
A decisão lead/rear por stance estática (comparar x dos tornozelos) está em
nível de ruído. Teste uma abordagem por golpe: o punho de maior deslocamento
líquido define a mão que golpeou e a direção de extensão define a "frente"
local. Meça a acurácia nos splits de treino/validação/teste.
```

## 6. Depuração de ambiente (GPU/CUDA)

```text
TensorFlow no servidor está dando "Loaded runtime CuDNN library: 9.1.0 but
source was compiled with: 9.3.0. Dnn is not supported" nas H100. Como corrigir
sem quebrar o PyTorch instalado no mesmo ambiente conda?
```

```text
ImportError: libc10_cuda.so: undefined symbol cudaGetDriverEntryPointByVersion
ao importar torch depois do tensorflow no mesmo processo. Qual a causa e como
contornar?
```

## 7. Relatório e apresentação

```text
Revise este trecho do relatório (em typst) sobre a metodologia de segmentação
de eventos por velocidade do punho (Schmitt trigger). Deixe o texto mais claro
e acadêmico, sem mudar o conteúdo técnico.
```

```text
Gere a matriz de confusão do modelo em matplotlib com rótulos legíveis para
incluir no pôster A1.
```
