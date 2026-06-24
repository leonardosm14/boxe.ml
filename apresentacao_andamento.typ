#import "@preview/diatypst:0.9.1": *
#show: slides.with(
  title: "Detecção e Classificação de Golpes em Boxe",
  subtitle: "Apresentação de Andamento",
  date: "Visão Computacional — 2026",
  authors: ("João Pedro Faraoni", "Leonardo Marques", "Pedro Gimenez", "Tom Hunt"),
  layout: "medium",
  ratio: 16/9,
  title-color: blue.darken(80%),
  toc: false,
)
#set text(lang: "pt", size: 11pt)
#set par(justify: true)

== Recap da Proposta
- *Título do projeto:* Detecção e Classificação de Golpes em Boxe
- Pipeline que recebe um vídeo de boxe, extrai esqueletos corporais e classifica automaticamente 6 tipos de golpe (Jab, Cross, Lead Hook, Rear Hook, Lead Uppercut, Rear Uppercut)
#v(8pt)
#align(center)[
  #image("assets/pipeline.png", width: 92%)
  #v(-2pt)
  #set text(size: 9pt, weight: "bold", fill: blue.darken(70%))
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [Vídeo],
    [Esqueleto],
    [Classificação],
    [Output],
  )
]
== Revisão da Literatura
#set text(size: 10.5pt)
Paper principal de referência @boxingVI.
#align(center)[
    #image("assets/paper_boxing1.png", width: 70%)
    #v(8pt)
    #set text(size: 10.5pt)
]

- Paper sobre reconhecimento de ações de boxe.
- Definição do conjunto de *classes de golpes* e a abordagem de *janela temporal deslizante de 25 frames* para capturar a dinâmica do movimento.

#v(1fr)
== Revisão da Literatura
#v(1fr)
  #align(center)[
    #image("assets/paper_pose.png", width: 70%)
  ]

  #set text(size: 10.5pt)
  - Paper de *pose estimation* multi-pessoa em tempo real.
  - *Normalização de coordenadas* dos 17 keypoints COCO.
  - *Filtragem por confiança média* para garantir qualidade dos esqueletos extraídos.
  - Base para a extração de keypoints com YOLOv8m-Pose e a estratégia de filtragem por confiança (≥ 0.5) usada no pipeline.

#v(1fr)
== Dataset

Utilizamos o dataset desenvolvido no projeto BoxingVI @boxingVI como principal fonte de dados.

#align(center)[
  #image("assets/dataset.png", width: 60%)
]

== Dataset
#v(0.8em)
O dataset disponibilizado no GitHub@boxingvi_github conta com marcações sob 10 vídeos individuais de treino de boxe. Para cada  vídeo temos o arquivo de anotação em CSV e o arquivo de esqueleto (_skeleton_) do lutador em `.npy`.

- Para uma janela de no máximo 25 frames, tem-se a marcação em arquivo CSV do frame inicial, frame final e o golpe realizado, para cada vídeo:

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,

    table.header(
      [ID],
      [start_frame],
      [end_frame],
      [class]
    ),

    [1], [6675], [6688], [Jab],
    [2], [6689], [6697], [Cross],
    [3], [6698], [6710], [Lead Hook],
    [4], [6722], [6734], [Jab],
    [5], [6735], [6745], [Cross],
    [⋮], [⋮], [⋮], [⋮]
  ),
  caption: [CSV relativo à V2.]
)

== Dataset
#v(0.8em)

Contabilizando todos os CSVs, temos o total de golpes marcados:

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,

    table.header(
      [Classe],
      [Quantidade]
    ),

    [Cross], [1373],
    [Jab], [1289],
    [Lead Hook], [1037],
    [Lead Uppercut], [689],
    [Rear Hook], [394],
    [Rear Uppercut], [681],
  ),
  caption: [Distribuição de golpes.]
) <tbl-boxingvi-distribution>

Percebe-se uma discrepância em relação à quantidade de golpes marcados para Jab, Cross e Lead Hook (+1000) quando comparados aos demais.

== Dataset

Os esqueletos são armazenados em arquivos NumPy (.npy) contendo tensores de dimensão $(N,T,J,C)$, onde $N$ representa o número de golpes anotados, $T$ o número máximo de frames por golpe, $J$ o número de articulações do modelo corporal COCO e $C$ as coordenadas espaciais de cada articulação.

Por exemplo, o arquivo V1.npy possui dimensão $(1866,25,17,2)$, indicando 1866 golpes segmentados, cada um representado por uma sequência temporal de até 25 frames contendo 17 articulações descritas pelas coordenadas normalizadas $(x,y)$.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,

    table.header(
      [Keypoint],
      [x],
      [y]
    ),

    [Nose], [0.346], [0.295],
    [Left Eye], [0.343], [0.300],
    [Right Eye], [0.343], [0.278],
    [⋮], [⋮], [⋮]
  ),
  caption: [Exemplo de coordenadas normalizadas.]
)

== Modelagem de Dados

Devido a discrepância entre anotações de golpes como Rear Hook e Rear Uppercut, nossos primeiros resultados não foram satisfatórios. Portanto, optamos por agrupar os golpes nas seguintes classes:

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,

    table.header(
      [Classe],
      [Conjunto de Dados],
      [Quantidade]
    ),

    [Straight], [Jab $union$ Cross], [2662],
    [Uppercut], [Lead Uppercut $union$ Rear Uppercut],[1370],
    [Hook], [Lead Hook $union$ Rear Hook], [1431]
  ),
  caption: [União de dados para criação do Modelo]
)

Observando a guarda do lutador, temos a seguinte heurística: se perna direita está posicionada à frente da perna esquerda, então a mão direita é a "mão da frente", enquanto a mão esquerda é a "mão de trás". O contrário também é válido;
- Golpes realizados com a "mão da frente" são: *Jab, Lead Hook, Lead Uppercut*;
- Golpes realizados com a "mão de trás" são: *Cross, Rear Hook, Rear Uppercut*.

Logo, na pipeline final basta verificar a posição do lutador e classificar o golpe em lead/rear.


== Modelo de Machine Learning

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,

    table.header(
      [Camada],
      [Parâmetros],
      [Função]
    ),

    [Entrada], [(25, 102)], [Sequência temporal de poses e derivadas -- 102 features no total para posição, aceleração e velocidade],

    [BiLSTM], [64 unidades], [Modelagem das dependências temporais em ambas as direções],

    [Dropout], [0.3], [Redução de sobreajuste],

    [Multi-Head Attention], [2 cabeças, key_dim=32], [Ênfase nos frames mais relevantes do golpe],

    [Residual + LayerNorm], [-], [Estabilização do treinamento],

    [Global Average Pooling], [-], [Agregação temporal da sequência],

    [Dense], [64 neurônios, ReLU], [Extração de características discriminantes],

    [Dropout], [0.3], [Regularização adicional],

    [Saída], [Softmax (3 classes)], [Classificação do golpe]
  ),
  caption: [Arquitetura da rede proposta para classificação dos golpes.]
) <tbl-network>

== Modelo de Machine Learning

Código Python, desenvolvido com a biblioteca TensorFlow 9.3:
#figure(
  align(left)[
    #block(
      fill: rgb("f8f9fa"),
      inset: 12pt,
      radius: 4pt,
      stroke: 0.5pt + rgb("e2e8f0"),
      width: 100%,
      [
        #set text(font: "Fira Code", size: 9pt)

        ```python
        def build_model(input_shape, nc):
            inp = Input(shape=input_shape)

            x = Bidirectional(
                LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
            )(inp)
            x = Dropout(0.3)(x)
            att = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
            x = Add()([x, att])
            x = LayerNormalization()(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation="relu",  kernel_regularizer=l2(5e-4))(x)
            x = Dropout(0.3)(x)

            return Model(inp, Dense(nc, activation="softmax")(x))
        ```
      ]
    )
  ],
  caption: [Arquitetura proposta para o classificador de golpes baseada em Bi-LSTM],
  supplement: [Código],
) <fig-modelo-boxe>

== Treinamento do Modelo
#v(3em)
Para realizar o treinamento, consideramos para treino as anotações e esqueletos dos vídeos *V1, V2, V7 e V8*:

- Straight: 1114, Hook: 596, Uppercut: 408.

Para validação os vídeos *V3 e V4*;

- Straight: 197, Hook: 105, Uppercut: 72.

Por fim, o conjunto de teste foi composto pelos vídeos *V5, V9* e *V10*:

- Straight: 519, Hook: 172, Uppercut: 208.

O vídeo *V6* não foi utilizado devido a problemas de formatação nos dados disponibilizados.

== Técnica de Aumento de Dados

Com o objetivo de aumentar a variabilidade do conjunto de treinamento e reduzir o risco de sobreajuste, foi utilizada uma estratégia de espelhamento horizontal dos esqueletos. Inicialmente, todas as coordenadas horizontais $(x)$ são invertidas em relação ao referencial corporal. Em seguida, as articulações correspondentes aos lados esquerdo e direito do corpo são trocadas, preservando a consistência anatômica do esqueleto. Dessa forma, ombros, cotovelos, punhos, quadris, joelhos e tornozelos esquerdos passam a ocupar as posições anteriormente associadas ao lado direito, e vice-versa. Após o espelhamento, as mesmas etapas de normalização e extração de características são aplicadas, gerando uma nova amostra sintética para cada golpe presente no conjunto de treinamento.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,

    table.header(
      [Classe],
      [Original],
      [Espelhado],
      [Total]
    ),

    [Straight], [1114], [1114], [2228],
    [Hook], [596], [596], [1192],
    [Uppercut], [408], [408], [816],

    [Total], [2118], [2118], [4236],
  ),
  caption: [Impacto da estratégia de aumento de dados por espelhamento horizontal.]
) <tbl-augmentation>

== Resultados: Treinamento do Modelo

O modelo foi treinado utilizando pesos de classe para mitigar o desbalanceamento, aplicando o otimizador com decaimento de taxa de aprendizado e critério de parada antecipada (*Early Stopping*).

- *Configuração dos Callbacks:*
  - `EarlyStopping`: Paciência de 15 épocas monitorando `val_accuracy`.
  - `ReduceLROnPlateau`: Fator de 0.5 com paciência de 5 épocas sob `val_loss`.
  - `ModelCheckpoint`: Salvamento exclusivo do melhor estado (`best_model.keras`).

- *Conclusão do Treinamento:*
  O treinamento foi interrompido na *época 60* pelo critério de parada, restaurando os pesos da época de melhor desempenho geral na validação.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    table.header(
      [Conjunto], [Acurácia], [Loss], [Val. Acurácia], [Val. Loss]
    ),
    [Época 60 (Final)], [90.18%], [0.3564], [93.05%], [0.2357],
    [*Melhor Estado*], [91.08%], [0.3490], [*93.05%*], [*0.2232*]
  ),
  caption: [Métricas obtidas no final do treinamento (Época de convergência vs Última Época).]
)

== Resultados: Avaliação no Conjunto de Teste

A avaliação final foi realizada de forma estrita no conjunto de teste independente (Vídeos V5, V9 e V10), totalizando 899 golpes anotados.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    table.header(
      [Classe], [Precisão], [Recall], [F1-Score], [Suporte]
    ),
    [Hook], [0.72], [0.66], [0.68], [172],
    [Straight], [0.89], [0.80], [0.84], [519],
    [Uppercut], [0.63], [0.83], [0.71], [208],
    table.hline(),
    [*Acurácia Geral*], [], [], [*0.78*], [*899*],
    [Macro Avg], [0.74], [0.76], [0.75], [899],
    [Weighted Avg], [0.80], [0.78], [0.78], [899]
  ),
  caption: [Relatório de classificação obtido no conjunto de teste.]
)

- *Destaque:* A classe *Straight* (Jab/Cross) obteve o maior F1-Score (0.84), impulsionada pelo maior volume de dados históricos. A classe *Uppercut* apresentou alto *recall* (0.83), indicando baixa taxa de falsos negativos.

== Resultados: Matriz de Confusão

#v(0.5fr)
#grid(
  columns: (52%, 48%),
  gutter: 16pt,
  align: horizon,
  [
    #set text(size: 10.5pt)
    Análise das predições versus os rótulos verdadeiros obtidos no conjunto de teste:

    - *Observações de Desempenho:*
      - O modelo demonstra solidez na identificação de trajetórias retilíneas (*Straight*).
      - Há uma intersecção moderada de confusão entre *Hooks* e *Uppercuts*.
      - Isso decorre de similaridades dinâmicas nas assinaturas de aceleração das janelas de esqueleto quando os golpes não são executados com velocidade máxima.
  ],
  [
    #figure(
      image("assets/matrix.png", width: 100%),
      caption: [Matriz de confusão resultante.],
    )
  ]
)
#v(1fr)

== Pipeline Principal
#v(2em)
Como parte de aplicar o modelo, criamos o módulo `boxe.py`, que funciona via CLI para aplicação da predição de golpes em um vídeo de entrada. Ex:

```sh
python3 boxe.py --video v1.mp4 --model modelo_boxe.keras --output outputs/
```

O script atualmente realiza:

- Carregamento do vídeo com OpenCV,
- Cria esqueleto do lutador utilizando YOLOv8m-Pose em formato `.npy`, COCO 17 e coordenadas (x, y),
- Carrega o Modelo com TensorFlow,
- Aplica o modelo sobre o esqueleto, escrevendo sobre os frames do vídeo os golpes preditos,
- Salva o vídeo em `outputs/`, com predição de golpes.

== Demonstração Prática: Entrada vs. Output

#v(1fr)
#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: -20pt,
    align: horizon,
    [
      #figure(
        image("assets/adam_input.png", width: 48%),
        caption: [Frame original de entrada do fluxo.],
      )
    ],
    [
      #figure(
        image("assets/adam_output.png", width: 48%),
        caption: [Output inferência de classe.],
      )
    ]
  )
]
#v(1fr)

== Tracking de Dois Lutadores: Motivação

A pipeline original assumia *um único lutador* por vídeo: a extração mantinha apenas a primeira pessoa detectada (`keypoints.data[0]`), descartando as demais. Para aplicar o sistema a *sparrings reais*, é necessário rastrear os *dois* lutadores simultaneamente e classificar o golpe de cada um.

#v(0.6em)
*Desafio central:* distinguir os lutadores sem confundir quem é quem. O identificador (_id_) do rastreador do YOLO não é estável — ele troca (2 → 3 → 4) sempre que há *oclusão ou clinch*, o que inviabiliza seguir cada lutador pelo _id_.

#v(0.6em)
*Abordagem adotada:* identidade por *posição horizontal* (esquerda/direita), independente do _id_ do rastreador. Como, nesta etapa, assume-se apenas os dois lutadores em quadro, "quem está à esquerda" e "quem está à direita" é uma identidade estável o suficiente e que sobrevive à troca de _id_.

== Tracking: Pipeline Multi-Pessoa

+ *Extração multi-pessoa:* YOLOv8m-Pose com rastreamento; guardam-se *todas* as detecções de cada frame (17 keypoints COCO normalizados, confiança por junta e caixa delimitadora em pixels). Mantém-se o filtro de *confiança média ≥ 0.5*.

+ *Atribuição por posição:* em cada frame, as detecções são ordenadas pelo *centro horizontal* da caixa; a mais à esquerda torna-se _Left boxer_ e a mais à direita _Right boxer_, formando uma sequência de esqueleto por lado.

+ *Preenchimento de lacunas (gap-fill):* frames em que um lado não é detectado repetem a última pose conhecida, mantendo a sequência contínua exigida pela janela de 25 frames.

+ *Classificação por lutador:* o *mesmo* classificador por evento (gatilho de Schmitt sobre a velocidade do punho + média do _softmax_ no pico) é executado para *cada* sequência. Reuso integral do modelo treinado, *sem retreino*.

== Identidade por Posição

A identidade de cada lutador é recomputada a cada frame, ordenando as caixas pelo centro horizontal — não há dependência do _id_ do rastreador:

#figure(
  align(left)[
    #block(
      fill: rgb("f8f9fa"),
      inset: 12pt,
      radius: 4pt,
      stroke: 0.5pt + rgb("e2e8f0"),
      width: 100%,
      [
        #set text(font: "Fira Code", size: 9pt)

        ```python
        def center_x(det):
            return (det["box"][0] + det["box"][2]) / 2.0  # (x1 + x2) / 2

        dets = sorted(dets, key=center_x)   # esquerda -> direita
        if len(dets) >= 2:
            put(slot=0, dets[0])    # mais a esquerda  -> Left boxer
            put(slot=1, dets[-1])   # mais a direita   -> Right boxer
        ```
      ]
    )
  ],
  caption: [Núcleo da atribuição de identidade por posição.],
)

#v(0.4em)
Usa-se o *centro* da caixa (e não a borda) por ser mais robusto quando as caixas têm larguras diferentes — por exemplo, um lutador parcialmente ocluído.

== Rótulos, Saídas e Robustez

- *Rótulos sobre o vídeo:* `Left boxer: Jab (98%)` à esquerda e `Right boxer: Cross (66%)` à direita, simultaneamente. Com um único lutador em quadro, o rótulo é apenas `Boxer`.

- *Limiar de presença mínima (10% dos frames):* descarta o "lutador fantasma" gerado por detecções espúrias (reflexo, sombra, treinador) que o YOLO ocasionalmente reporta como _2 persons_ em poucos frames.

- *Dois vídeos de saída:* (1) apenas as *caixas delimitadoras*; (2) caixas *+ esqueleto* COCO desenhado por lutador.

- *Limitação atual (etapa sem persistência):* se os lutadores *cruzam de lado*, os rótulos esquerda/direita trocam de dono. Por isso o rótulo descreve a *posição* atual, e não um número fixo. A persistência de identidade (re-_ID_ por aparência) fica como trabalho futuro.

#v(0.4em)
Execução via CLI, gerando os dois vídeos automaticamente:
```sh
python3 boxe.py -v videos/spar.mp4 --clear-cache
# Gera: outputs/spar_box.mp4  e  outputs/spar_pose.mp4
```

== Demonstração: Spar com Dois Lutadores

#v(0.5fr)
#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 16pt,
    align: horizon,
    [
      #figure(
        rect(width: 100%, height: 5cm, fill: luma(235), stroke: 0.5pt)[
          #align(center + horizon)[#text(size: 9pt)[PLACEHOLDER \ Saída: apenas caixas \ (`spar_box.mp4`)]]
        ],
        caption: [Dois lutadores rastreados — rótulo de golpe por lado.],
      )
    ],
    [
      #figure(
        rect(width: 100%, height: 5cm, fill: luma(235), stroke: 0.5pt)[
          #align(center + horizon)[#text(size: 9pt)[PLACEHOLDER \ Saída: caixas + esqueleto \ (`spar_pose.mp4`)]]
        ],
        caption: [Mesma cena com o esqueleto COCO sobreposto.],
      )
    ]
  )
]
#v(1fr)

== Próximos Passos
#v(1fr)
- Tracking como YOLO Pose, para permitir lutas reais com duas pessoas.
- Melhoramento do modelo, visto que ainda há confusão entre classes de golpes.
- Desenvolvimento do relatório final da disciplina.

#v(1fr)
// == Tracking

// Em andamento, essencial para aplicação em lutas com dois lutadores.

== Referências
#bibliography("refs.bib")
