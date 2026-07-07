//// CONFIGURAÇÕES E PALETA DE CORES
#set page(
  width: 594mm, 
  height: 841mm, 
  margin: 1.2cm, 
  fill: rgb("#F0F4F8")
)
#set text(lang: "pt")
#set text(font: "Roboto Mono", size: 22pt, fill: rgb("#2D3748"))

#set par(justify: true, leading: 1.0em, first-line-indent: (amount: 1.0em, all: true))

#let cor-primaria = rgb("#0D47A1")
#let cor-secundaria = rgb("#1976D2")
#let cor-texto-claro = rgb("#BEE3F8")
#let cor-borda = rgb("#CBD5E0")
#let gradiente-titulo = gradient.linear(cor-primaria, cor-secundaria, angle: 45deg)

// CARDS PERSONALIZADOS
#let card(titulo, corpo, espaco-extra: 0pt) = {
  block(
    width: 100%,
    fill: white,
    radius: 12pt,
    stroke: 2pt + cor-borda,
    inset: 1.5cm,
    breakable: false,
    outset: 0pt
  )[
    #text(size: 30pt, weight: "bold", fill: cor-primaria)[#titulo]
    #v(-0.3cm)
    #line(length: 100%, stroke: 3pt + cor-secundaria)
    #set text(size: 18pt)
    #corpo
    #v(espaco-extra)
  ]
}


// CABEÇALHO

#block(
  width: 100%,
  fill: gradiente-titulo,
  radius: 16pt,
  inset: 1.8cm,
  outset: 0pt
)[
  #v(-0.5em) 
  #place(left + horizon)[
    #grid(
      columns: 2,
      gutter: 15pt,
      image("img/ufsc_logo.svg", width: 120pt),
      image("img/ine_logo.svg", width: 130pt),
    )
  ]
  #place(right + horizon)[
    #grid(
      columns: 2,
      gutter: 15pt,
image("img/qrcode-youtube.png", width: 170pt),
      image("img/qrcode-github.png", width: 170pt),
    )
  ]
  #align(horizon + center)[
    #text(size: 16pt, fill: cor-texto-claro, weight: "bold")[
      Reconhecimento de Padrões - Visão Computacional (UFSC/INE)
    ]
    #v(-0.5em)
    #text(size: 30pt, weight: "bold", fill: white)[
      Detecção e Classificação de Golpes \ em Boxe com Visão Computacional
    ]
    #v(-0.5em)
    #text(size: 16pt, fill: cor-texto-claro, weight: "bold")[
      João Pedro Tamburo Faraoni, Leonardo de Sousa Marques,\
      Pedro Henrique Gimenez, Tom Pereira Hunt
    ]
   #v(-0.5em) 
  ]
]


// COLUNAS DE CONTEÚDO

#columns(2, gutter: 0.8cm)[

  // COLUNA 1
  #card("1. Introdução")[
    O boxe se destaca como um esporte de combate caracterizado pela combinação rápida de técnicas ofensivas e defensivas em ambiente altamente dinâmico. Uma dificuldade dos telespectadores é identificar os golpes realizados durante a partida devido à similaridade e rapidez. Para solucionar isso, apresentamos o *boxe.ml* v1.0.0, uma ferramenta _Open Source_ em Python que recebe um vídeo via CLI, rastreia os lutadores e classifica os golpes detectados.
  ]

  #card("2. Dataset e Modelagem de Dados")[
    A fonte principal de dados proveio do _benchmark_ *BoxingVI* @boxingVI, representado na @videos-dataset, com repositório público @boxingvi_github e 9 vídeos de boxe anotados. Foram utilizados arquivos CSV para marcar os frames de ínicio, fim e golpe realizado, e arquivos NPY para criação dos esqueletos dos lutadores, no formato COCO 17. Na @tabela-dados-iniciais abaixo, observa-se as quantidades de dados iniciais para cada golpe.
    #v(-0.0cm)
    #grid(
      columns: (1.5fr, 1fr),
      gutter: 1em,
      align: horizon,
      [
        #align(center)[
          #figure(
            image("img/dataset.png", width: 100%),
            caption: [Vídeos do Dataset @boxingVI]
          ) <videos-dataset>
        ]
      ],
      [
        #v(1em)
        #figure(
          table(
            columns: (1.5fr, 1fr),
            stroke: 1.0pt,
            inset: (y: 0.65em),
            fill: (col, row) => if row == 0 { cor-texto-claro } else { none },
            table.header(
              [*Classe*],
              [*Dados*]
            ),
            [Cross], [1373],
            [Jab], [1289],
            [Lead Hook], [1037],
            [Lead Uppercut], [689],
            [Rear Hook], [394],
            [Rear Uppercut], [681],
          ),
          caption: "Dados Iniciais",
          kind: table,
        ) <tabela-dados-iniciais>
        #v(0.3em)
      ]
    )
    #v(0.5cm)
     As classes originais foram agrupadas nas três categorias da @dados-agrupados para mitigar o desbalanceamento por baixo volume de dados. Os vídeos $V_1$, $V_2$, $V_3$, $V_4$, $V_7$ e $V_8$ foram usados para treino/validação (_split_ 65%/35%), e $V_5$, $V_9$ e $V_10$ para teste, garantindo diversidade de lutadores e cenários. A divisão final é mostrada na tabela @dados-split.

     #grid(
        columns: (1fr, 1fr),
        gutter: 1em,
        align: horizon,
        [
          #figure(
            table(
              columns: (auto, 0.8fr, 0.3fr),
              stroke: 1.0pt,
              align: (center),
              inset: (y: 0.5em),
              fill: (col, row) => if row == 0 { cor-texto-claro } else { none },
              table.header(
                [*Classe*], [*Conjunto*], [*Dados*]
              ),
              [Straight], [Jab, Cross],                       [2662],
              [Uppercut], [Lead Uppercut,\ Rear Uppercut],    [1370],
              [Hook],     [Lead Hook,\ Rear Hook],             [1431],
            ),
            caption: [União de dados.]
          ) <dados-agrupados>
        ],
        [
          #v(-0.0em)
          #figure(
            table(
              columns: (auto, auto, auto, auto),
              stroke: 1.0pt,
              align: (center, center, center, center),
              inset: (y: 0.92em),
              fill: (col, row) => if row == 0 { cor-texto-claro } else { none },
              table.header(
                [*Conjunto*], [*Straight*], [*Hook*], [*Uppercut*]
              ),
              [Treino],    [1390], [673], [450],
              [Valid.], [749],  [362], [243],
              [Teste],     [519],  [172], [208],
            ),
            caption: [Distribuição de dados.]
          ) <dados-split>
        ]
      )
    #v(0.3cm)
    Por fim, aplicou-se _data augmentation_ por espelhamento horizontal, dobrando as amostras de treino para *5.026 golpes*.
  ]

  #card("3. Modelo de Machine Learning (ML)")[
    O modelo de ML para a classificação temporal com rede *Bi-LSTM* foi desenvolvido com TensorFlow, processando 25 frames sequenciais com dados de 17 juntas corporais extraídas via *AlphaPose*.

    #v(1.4em)
    #align(center+horizon)[
      #scale(x: 140%, y: 140%)[
        #include "modelo.typ"
      ]
    ]
   #v(1.6em)
  ]

  #colbreak()

  #card("4. Resultado do Treinamento")[
    O modelo convergiu após 48 épocas, com acurácia de *81,38%* (treino) e *84,42%* (validação). Em testes com vídeos inéditos ($V_5$, $V_9$, $V_10$), o modelo manteve *83% de acurácia global*.

    #v(-3em)
    #grid(
      columns: (2),
      gutter: 1em,
      align: horizon,
      [
        #v(3em)
        #figure(
          table(
            columns: (auto, auto, auto, auto, auto),
            stroke: 0.5pt,
            align: (center, center, center, center, center),
            inset: (y: 0.68em),
            fill: (col, row) => if row == 0 { rgb("#BEE3F8") } else if row == 4 or row == 5 { rgb("#EDF2F7") } else { none },
            table.header(
              [*Classe*],
              rotate(-90deg, reflow: true)[*Precisão*],
              rotate(-90deg, reflow: true)[*Revocação*],
              rotate(-90deg, reflow: true)[*F1-score*],
              rotate(-90deg, reflow: true)[*Suporte*],
            ),
            [Hook],        [0,68], [0,69], [0,68], [172],
            [Straight],    [0,91], [0,91], [0,91], [519],
            [Uppercut],    [0,76], [0,76], [0,76], [208],
            [*Macro*],     [0,78], [0,78], [0,78], [899],
            [*Ponderada*], [0,83], [0,83], [0,83], [899],
          ),
          caption: [Métricas de Teste],
        ) <tab-resultados-poster>
      ],
      [
        #v(2.9em)
        #align(center+horizon)[
          #figure(
            image("img/matrix.pdf", width: 100%),
            caption: [Matriz de Confusão]
          )
        ]
      ]
    )
  ]

  #card("5. Pipeline")[
    A pipeline do *boxe.ml* é demonstrada pelo diagrama abaixo.

    #v(2.8em)
    #align(center+horizon)[
      #scale(x: 170%, y: 170%)[
        #include "pipeline.typ"
      ]
    ]

    #v(2.8em)

    #figure(
      image("img/spar_leadrear.jpg", width: 81.65%),
    )

    #v(-1em)

  ]

  #card("6. Considerações Finais")[
    #v(0cm)
    O trabalho realizado cumpre com o objetivo de inferir golpes em vídeos reais de partidas de boxe, a partir do uso de tecnologias de Machine Learning modernas e já consolidadas, atingindo bons resultados. Como trabalhos futuros, busca-se ampliar a diversidade dos vídeos de treinamento, a fim de aumentar a acurácia do modelo.
    #v(-1em)
  ]

   #card("7. Referências")[
    #v(-2.6em)
    
    #bibliography(title: "", "bibliography.bib")
    #v(-1em)
  ]

]


// // RODAPÉ

// #v(1fr)
// #block(
//   width: 100%,
//   fill: rgb("#CBD5E0"),
//   radius: 10pt,
//   inset: 1.2cm
// )[
//   #align(center)[
//     #text(size: 20pt, fill: rgb("#4A5568"))[
//       Repositório: *github.com/leonardosm14/boxe.ml*
//     ]
//   ]
// ]