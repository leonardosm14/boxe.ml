#import "@preview/cetz:0.2.2": canvas, draw

= _Pipeline_

Com o modelo pronto e testado, iniciou-se a parte final deste trabalho, que foi desenvolver uma _pipeline_ que realiza os passos descritos abaixo, a partir da execução do arquivo `boxe.py`:

+ Receber vídeo MP4 de entrada;
+ Converter vídeo para 25 fps, com FFmpeg #footnote[Repositório do FFmpeg: #link("https://github.com/ffmpeg/ffmpeg")] ;
+ Carregar vídeo com OpenCV #footnote[Repositório do OpenCV: #link("https://github.com/opencv/opencv")] ;
+ Extrair esqueletos com YOLOv8m-Pose (COCO 17, coordenadas $x, y$) + _tracking_;
+ Carregar modelo Bi-LSTM;
+ Aplicar modelo sobre janelas do esqueleto;
+ Inferir _lead_ ou _rear_ pela geometria do golpe (punho que golpeou + perna da frente);
+ Escrever predições sobre os frames;
+ Gerar o vídeo anotado.

== _Tracking_ de Dois Lutadores
A presença de dois lutadores no vídeo introduz um desafio que vai além da simples detecção de poses: é necessário manter a identidade de cada boxeador ao longo do tempo, mesmo quando eles se cruzam ou saem momentaneamente do enquadramento. Para isso, utilizou-se o YOLOv8m-Pose em conjunto com o algoritmo de rastreamento ByteTrack, integrado nativamente ao framework Ultralytics @yolov8. A cada frame, o modelo detecta todas as pessoas presentes e, para cada detecção, retorna 17 _keypoints_ no padrão COCO (coordenadas x, y) normalizadas pela resolução do vídeo e uma confiança por junta, além de uma caixa delimitadora em pixels e um `track_id` persistente atribuído pelo ByteTrack @bytetrack. Detecções com confiança média inferior a 50% são descartadas.
A atribuição de identidade a cada boxeador, denominados *Boxer 1* e *Boxer 2*, é realizada em dois estágios a partir das detecções brutas. No primeiro, cada detecção com um `track_id` previamente associado a um dos dois _slots_ é diretamente atribuída a esse _slot_, independentemente de sua posição na tela, garantindo que a identidade de cada lutador se mantenha estável mesmo após cruzamentos de lado. No segundo estágio, detecções sem `track_id` ou com ID desconhecido (situações comuns no frame inicial ou após oclusões prolongadas) são atribuídas pela posição horizontal: a detecção mais à esquerda é associada ao Boxer 1 e a mais à direita ao Boxer 2. Quando um novo ID é atribuído por posição, o _slot_ o registra para os frames seguintes, absorvendo naturalmente fragmentações do rastreador.
Como o traqueamento é esparso por natureza, a matriz de esqueletos de cada boxeador contém lacunas, frames em que o lutador não foi detectado, representadas por linhas de zeros. Uma lacuna implica que o punho "teleporta" para a origem e retorna ao frame seguinte, gerando um falso pico de velocidade e um evento de golpe sem contexto. Para evitar isso, frames sem detecção são preenchidos com a última pose conhecida daquele boxeador, mantendo a continuidade do sinal. Adicionalmente, o parâmetro `MIN_PRESENCE_RATIO` estabelece um limiar mínimo de presença: _slots_ detectados em menos de 10% dos frames do clipe, gerados por reflexos, sombras ou outras interferências visuais momentâneas são descartados da renderização e da rotulação, evitando a criação de um segundo boxeador fantasma em clipes com apenas um atleta. Na @fig-spar-leadrear, observamos a saída final da _pipeline_ com o traqueamento completo dos lutadores e inferência do golpe, discutido na @inferencia-label. Como referência, utilizou-se o vídeo `videos/fight.mp4`, presente no repositório.

#figure(
  image("../img/spar_leadrear.jpg", width: 100%),
  caption: [Saída da _pipeline_ com dois lutadores: identidade persistente (Boxer 1/Boxer 2), esqueletos COCO-17 e classe final do golpe com a mão inferida (_lead_/_rear_) por boxeador.],
) <fig-spar-leadrear>

== Inferência do Golpe

<inferencia-label>

O modelo de classificação descrito anteriormente prevê apenas o *tipo* do golpe (Straight, Hook ou Uppercut). A distinção entre a mão da frente (_lead_) e a mão de trás (_rear_) -- que separa, por exemplo, o Jab do Cross -- é realizada na própria _pipeline_, por geometria pura sobre os _keypoints_, sem um segundo modelo treinado. Essa separação é consequência direta das escolhas de projeto: a aumentação por espelhamento horizontal só dobra os dados de treinamento porque a responsabilidade de distinguir a mão do golpe foi removida do modelo.

A primeira abordagem testada foi inferir a guarda (_stance_) de forma estática, comparando a posição dos tornozelos e a orientação do tronco para decidir qual pé está à frente. Essa estratégia se mostrou equivalente a ruído neste cenário, por duas razões verificadas empiricamente: o YOLOv8m-Pose sempre retorna as 17 juntas, "alucinando" a posição do lado ocluído do corpo, e a câmera lateral colapsa a profundidade frente/trás na projeção 2D. Agregar a guarda em um único valor por clipe também degrada o resultado (de 0,76 para 0,61 na validação cruzada entre vídeos), pois a orientação do lutador em relação à câmera varia ao longo do clipe.

A decisão adotada é feita *por golpe*, usando o sinal dinâmico do próprio soco: (i) a mão que golpeou é o punho com maior deslocamento líquido na janela do evento (pico de $norm(p_t - p_0)$), com desempate pelo maior alcance punho-ombro quando os deslocamentos diferem menos de 10%; (ii) o sinal do deslocamento horizontal desse punho define a direção "frente" local do golpe; (iii) a perna (média de joelho e tornozelo) mais avançada nessa direção indica o pé da frente. Se a mão que golpeou está do mesmo lado do pé da frente, o golpe é _lead_; caso contrário, _rear_. A classe final é obtida pelo mapeamento direto: Straight vira Jab (_lead_) ou Cross (_rear_), e Hook e Uppercut recebem o prefixo Lead ou Rear.

Um detalhe de implementação relevante: a janela usada nessa decisão emprega o esqueleto *sem* a suavização temporal aplicada no restante da _pipeline_. Um soco rápido é um movimento de ida e volta em poucos frames, e a média móvel de 5 frames achata justamente o deslocamento do punho que golpeou -- com janelas suavizadas, a acurácia _lead_/_rear_ no vídeo de validação próprio cai de 0,89 para 0,28.

Avaliada isoladamente sobre as janelas anotadas do BoxingVI, a geometria atinge 0,85 de acurácia _lead_/_rear_ no conjunto de treino, 0,77 em validação cruzada entre vídeos e 0,74 no conjunto de teste. Em um vídeo gravado pelo próprio grupo e anotado manualmente (18 golpes, câmera traseira em ângulo de três quartos), a decisão por segmento anotado acerta 16 de 18 golpes (0,89), indicando que a abordagem geométrica generaliza para pontos de vista diferentes dos do dataset. A avaliação completa pode ser reproduzida pelo script `eval_leadrear.py` do #link("https://github.com/leonardosm14/boxe.ml/blob/main/evaluation/eval_leadrear.py", "repositório").


== _Setup_ e Utilização via CLI

O primeiro passo é clonar o #link("https://github.com/leonardosm14/boxe.ml", "repositório") do `boxe.ml`. Depois, é necessário instalar as dependências (TensorFlow, Ultralytics, ...) para conseguir executar a _pipeline_. Todas as dependências requisitadas foram listadas no arquivo `requirements.txt`. Para instalar, siga a @requirements. Sugere-se que o usuário crie um ambiente virtual do Python para realizar essa instalação, porém os dois primeiros comandos são opcionais.

#figure(
  block(
    fill: rgb("eff6ff"),
    inset: 12pt,
    radius: 4pt,
    width: 100%,
    align(left, [
      ```sh
      python3 -m venv venv \ 
      source venv/bin/activate \ 
      pip3 install -r requirements.txt
      ``` 
    ])
  ),
  caption: [Instalação dos requisitos (Linux).]
) <requirements>

Com isso, já é possível utilizar o código para detecção e classificação de golpes em vídeos selecionados. Basta executar o comando da @main. A _flag_ `--video` (ou `-v`) indica o caminho para o vídeo em MP4 desejado, enquanto `--output` (ou `-o`) indica a pasta em que o vídeo final analisado será armazenado. Existem _flags_ adicionais, como `--model` (ou `-m`), para que o usuário possa carregar seu próprio modelo (no formato `keras`), caso desejado -- por padrão, o módulo carrega o `modelo_boxe.keras` treinado para este projeto, também presente no repositório. E a _flag_ `--clear-cache`, para que os esqueletos de um vídeo já processado sejam removidos da pasta e a _pipeline_ execute do zero.

#figure(
  block(
    fill: rgb("eff6ff"),
    inset: 12pt,
    radius: 4pt,
    width: 100%,
    align(left, [
      ```sh
      python3 boxe.py \ 
      --video path/to/your/video.mp4 \ 
      --output output/
      ``` 
    ])
  ),
  caption: [Execução da _pipeline_.]
) <main>

A documentação completa de como executar os _scripts_ também está descrita no arquivo `README`, no #link("https://github.com/leonardosm14/boxe.ml/blob/main/README.md", "repositório").