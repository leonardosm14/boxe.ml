// Formtação
#set page(paper: "a4",margin: 2cm,numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 1em, first-line-indent: (amount: 1.5em, all: true))
#set heading(numbering: "1.")
#show heading: set block(above: 1.5em, below: 1.0em)
#set text(lang: "pt")
#show table.cell: set align(center + horizon)
#show link: set text(fill: blue.darken(50%))
#show ref: set text(fill: blue.darken(50%))

// Texto + infos
#show: doc => [
  #align(center)[

    #image("img/ufsc_logo.svg", width: 15%)
    #text(size: 18pt, weight: "bold")[
      Detecção e Classificação de Golpes em \ Boxe com Visão Computacional
    ]
    #v(0.8em)
    #text(size: 12pt)[
      João Pedro Tamburo Faraoni,
      Leonardo de Sousa Marques,\
      Pedro Henrique Gimenez,
      Tom Pereira Hunt
    ]
    #v(0.4em)
    #text(size: 11pt)[
      Departamento de Informática e Estatística \
      Universidade Federal de Santa Catarina
    ]
    #v(0.4em)
    #text(size: 10pt)[
    {#link("mailto:joao.faraoni@grad.ufsc.br")[joao.faraoni],
      #link("mailto:leonardo.sm@grad.ufsc.br")[leonardo.sm],
      #link("mailto:pedro.gimenez@grad.ufsc.br")[pedro.gimenez],
      #link("mailto:tom.hunt@grad.ufsc.br")[tom.hunt]}\@grad.ufsc.br
  ]
  ]
  #v(2em)
  #columns(2, doc)
]

// Seções

// Introdução
#include "sections/intro.typ"

// Metodologia
#include "sections/background.typ"

// Datasets
#include "sections/datasets.typ"

// Desenvolvimento
#include "sections/development.typ"

// Resultados
#include "sections/results.typ"

// Pipeline
#include "sections/pipeline.typ"

// Conclusões
#include "sections/conclusions.typ"

// Refs
#bibliography(title: "Referências Bibliográficas", "bibliography.bib")