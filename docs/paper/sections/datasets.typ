= _Datasets_

A fonte de dados principal deste trabalho foi os dados disponibilizados no repositório público do BoxingVI @boxingvi_github. Conforme a @dataset, pode-se observar a sequência dos 15 vídeos originais utilizados para a criação do conjunto de treino do _benchmark_. Todavia, na base de dados disponibilizada, foi dado acesso a apenas 9 vídeos.

#figure(
  image("../img/dataset.png"),
  caption: [_Dataset_ do BoxingVI @boxingVI.],
) <dataset>

No total, tivemos acesso a 5.463 dos clipes, um valor razoável quando comparado ao total do _benchmark_. A distribuição do total de golpes é vista na @tbl-boxingvi-distribution.

#figure(
  table(
    columns: (1fr, 1fr),
    stroke: 0.5pt,

    table.header(
      [*Classe*],
      [*Quantidade*]
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

O _dataset_ possui três pastas principais: `annotation_files`, `skeleton_data` e `rgb_videos`. As duas primeiras são as mais importantes para a realização do trabalho. A pasta de anotação contém os nove arquivos no formato CSV, com as seguinte colunas: `start_frame` (frame inicial), `end_frame` (frame final) e `class` (classe), que descreve o golpe realizado naquela janela temporal. Vale ressaltar, também, que os golpes foram anotados sob os vídeos numa janela de 25 fps, o que impacta diretamente na medição dos dados. Na tabela @dados-csv, pode-se observar a formatação comentada, em relação ao vídeo 2 ($V_2$).

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,

    table.header(
      [*id*],
      [*start_frame*],
      [*end_frame*],
      [*class*]
    ),

    [1], [6675], [6688], [Jab],
    [2], [6689], [6697], [Cross],
    [3], [6698], [6710], [Lead Hook],
    [4], [6722], [6734], [Jab],
    [5], [6735], [6745], [Cross],
    [⋮], [⋮], [⋮], [⋮]
  ),
  caption: [CSV relativo à $V_2$.]
) <dados-csv>

A pasta de dados dos esqueletos armazena os nove arquivos em formato NPY (binário padrão da biblioteca `NumPy`), de modo que cada arquivo contém as informações dos 17 principais pontos do corpo humano, conforme padronizado pelo Microsoft _Common Objects in Context_ (COCO) @coco, em determinado instante de tempo.

Os dados com os vídeos completos não foi utilizada na prática. Apenas foi realizado um _cross-check_ para verificar a consistência das anotações e garantir a diversidade dos vídeos e lutadores.