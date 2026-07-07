#import "@preview/lovelace:0.3.0": *

= Desenvolvimento

O `boxe.ml` foi projetado na linguagem de programação Python, versão 3.12. A estrutura do projeto segue a @estrutura-projeto abaixo. 
#v(-1.84em)
#figure(
  ```bash

  ├── dataset/
  │   ├── clean_annotation_data/
  │   ├── skeleton_data/
  │   └── rgb_videos/
  ├── utils/
  │   ├── stance_utils.py
  │   └── boxe_utils.py
  ├── boxe.py
  ├── stance.py
  ├── tracking.py
  └── training.ipynb
  ```
,
caption: [Estrutura do projeto.]
) <estrutura-projeto>

== Manipulação de Dados

Em primeiro plano, foi necessário realizar uma limpeza dos dados de anotação em CSV retirados de @boxingvi_github, visto que uma classe, em determinadas anotações, poderia estar escrita de formas diferenças (_e.g.,_ "Jab", "jab"). Os dados foram então adicionados à pasta `clean_annotation_data`.

Como escolha de projeto, a fim de não atribuir à rede neural a função de diferenciar entre os seis modelos, optou-se por separá-los em apenas três classes: *Straight*, composta por Jab e Cross, *Uppercut*, composta por Lead Uppercut e Rear Uppercut, e *Hook*, composta por Rear Hook e Lead Hook. Tomou-se essa decisão após uma massiva testagem com as seis classes originais, e percebeu-se que, por conta do baixo volume de dados para algumas das classes, o modelo não conseguia atingir uma boa acurácia. Essa manipualção de dados é realizada diretamente no módulo de treinamento. Conforme a @total-tres-classes, observa-se um melhor balanceamento dos dados quando comparados à @tbl-boxingvi-distribution.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: center,

    table.header(
      [*Classe*],
      [*Conjunto*],
      [*Quantidade*]
    ),

    [Straight], [Jab $union$ Cross], [2662],
    [Uppercut], [Lead Uppercut \ $union$ \ Rear Uppercut],[1370],
    [Hook], [Lead Hook \ $union$ \ Rear Hook], [1431]
  ),
  caption: [União de dados.]
) <total-tres-classes>

Optou-se por utilizar os vídeos $V_1$, $V_2$, $V_3$, $V_4$, $V_7$ e $V_8$ para treinamento e validação, com um _split_ aleatório de 65% e 35% dos dados para cada etapa, respectivamente -- utilizou-se o método `train_test_split` da biblioteca `Scikit-learn`, com semente = 42, a fim de garantir reprodutibilidade dos resultados. Já para a etapa de teste do modelo, utilizou-se os vídeos $V_5$, $V_9$ e $V_10$. Tanto os vídeos de treinamento/validação e teste foram separados de acordo com sua diversidade. Isso garante que nosso modelo de ML não "decore" os golpes baseando-se em um lutador, pose ou cenário em específico. Conforme a @tbl-split-dataset, observa-se a divisão gerada de golpes para cada uma das etapas mencionadas. No total, teve-se 2513 golpes de treinamento, 1354 golpes de validação e 899 golpes de teste.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,

    table.header(
      [*Conjunto*],
      [*Straight*],
      [*Hook*],
      [*Uppercut*]
    ),

    [Treino], [1390], [673], [450],
    [Validação], [749], [362], [243],
    [Teste], [519], [172], [208],
  ),
  caption: [Distribuição das amostras utilizadas nos conjuntos de treino, validação e teste.]
) <tbl-split-dataset>

Por fim, antes da criação e treinamento do modelo de ML, realizou-se a técnica de aumentação de dados por espelhamento @mirror. Como decidiu-se verificar se um golpe é _lead_ ou _rear_ somente na _pipeline_, percebeu-se a oportunidade de aumentar a quantidade de dados através do espelhamento. Portanto, como entrada do modelo, vamos ter o dobro de dados de treinamento (5026).

== Modelo de _Machine Learning_

Depois de todo o tratamento de dados, foi possível criar o modelo de _Machine Learning_ utilizado. Para este projeto, foi feito uso do Tensorflow v2.21.0, por conta da facilidade
que a biblioteca oferece para criação de um modelo de ML em camadas no Python.
Visto que se deseja prever movimentos em vídeos, optou-se por utilizar uma rede
neural _Bidirectional Long Short-Term Memory_ (Bi-LSTM), combinada com um
mecanismo de atenção multi-cabeça. O modelo é composto pelas seguintes camadas:

*1. Camada de Entrada (_Input Layer_).* Recebe sequências de esqueleto
pré-processadas com shape $("batch", 25, 102)$, onde 25 corresponde ao número
de frames da janela temporal e 102 às features extraídas de 17 juntas corporais
--- coordenadas $(x, y)$, velocidade $(v_x, v_y)$ e aceleração $(a_x, a_y)$
para cada junta.

*2. Bi-LSTM.* É a parte mais importante do
modelo, com 85.504 parâmetros. Um LSTM convencional lê a sequência de frames
do início ao fim; a versão bidirecional executa dois LSTMs em paralelo --- um
no sentido direto (frame $i arrow j$) e outro no sentido inverso
(frame $j arrow i$) --- e concatena as saídas, resultando em 128 unidades.
Isso é vantajoso porque um golpe possui contexto tanto no movimento de preparação
quanto no de retração. Regularizações de _dropout_ ($p = 0.30$) e
_recurrent dropout_ ($p = 0.20$) são aplicadas internamente.

*3. _Dropout_ ($p = 0.30$).* Após o Bi-LSTM, 30% dos neurônios são desligados
aleatoriamente a cada _batch_ durante o treino. Trata-se de uma técnica de
regularização que força o modelo a não depender de nenhum neurônio específico,
reduzindo o overfitting.

*4. _Multi-Head Attention_.* Com 2 cabeças e dimensão de
chave 32, esse mecanismo, inspirado na arquitetura Transformer, computa
relações de atenção entre todos os frames simultaneamente, identificando quais
instantes temporais são mais relevantes para a classificação do golpe. A
operação é definida por

$ "Attention"(Q, K, V) = "softmax"((Q K^top) / sqrt(d_k)) dot V $

onde $Q$, $K$ e $V$ são projeções lineares da mesma entrada (_self-attention_).

*5. Conexão Residual (_Add_ $xor$).* A saída do Dropout é somada à saída da camada
de atenção. Essa _skip connection_ garante que o gradiente flua diretamente pelo
caminho residual durante o _back propagation_, mitigando o problema de gradiente
que desaparece em redes mais profundas.

*6. Normalização de Camada.* Normaliza cada amostra
individualmente ao longo da dimensão de features, estabilizando o treinamento após a
operação de adição residual. Possui 256 parâmetros
treináveis.

*7. Pooling Global por Média.* Colapsa o eixo
temporal calculando a média ao longo dos 25 frames, transformando o tensor de
shape $(b a t c h, 25, 128)$ em $(b a t c h, 128)$. Essa agregação produz uma
representação compacta de toda a sequência antes da classificação final.

*8. Camada Densa com ReLU.* Camada totalmente conectada com 64 unidades e ativação
ReLU, responsável por aprender combinações não-lineares das _features_ agregadas.
Regularização $L_2$ ($lambda = 5 times 10^(-4)$) é aplicada aos pesos para
penalizar valores muito grandes.

*9. _Dropout_ ($p = 0.30$).* Segunda camada de _dropout_, aplicada após a densa,
mantendo a regularização próxima à saída do modelo.

*10. Camada de Saída (_Softmax_).* Camada densa com 3 unidades e ativação `softmax`,
que converte os logits em probabilidades para as três classes de golpe: Straight, Uppercut e Hook. A classe
predita é aquela com maior probabilidade.

No @alg-modelo-boxe abaixo, pode-se observar o pseudo-código que descreve a criação do modelo, seguindo as camadas supracitadas, em que _shape_ corresponde à dimensão de cada amostra de entrada, e $N$ é o número de classes (3).

#figure(
  pseudocode-list(booktabs: true, stroke: none)[
    + *function* BuildModel(*shape*, *N*):
      + i $<-$ Input(*shape*)
      + x $<-$ BiLSTM(64, 0.3, 0.2)(inp)
      + x $<-$ Dropout(0.3)(x)
      + att $<-$ MultiHeadAttention(2, 32)(x, x)
      + x $<-$ LayerNorm(x + att)
      + x $<-$ GlobalAveragePooling(x)
      + x $<-$ Dense(64, ReLU, $5 times 10^(-4)$)(x)
      + x $<-$ Dropout(0.3)(x)
      + *return* Model(i, Dense(*N*, softmax)(x))
  ],
  caption: [Modelo com rede neural Bi-LSTM],
  supplement: [Algoritmo],
) <alg-modelo-boxe>

O modelo proposto com camadas Bi-LSTM e atenção multi-cabeça 
resulta em uma arquitetura compacta de apenas 127.299 parâmetros
treináveis, conforme descrito na @tab-params-modelo.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    table.header(
      [*Parâmetros*],
      [*Quantidade*],
      [*Tamanho*],
    ),
    [Treináveis],     [127.299], [497,26 KB],
    [Não-treináveis], [0],       [0,00 B],
    [*Total*],        [*127.299*], [*497,26 KB*],
  ),
  caption: [Parâmetros do modelo proposto.],
) <tab-params-modelo>


== Balanceamento de Classes
Para lidar com o desbalanceamento entre as classes, foram atribuídos pesos
inversamente proporcionais à frequência de cada classe durante o treinamento,
calculados via 
$ w_c = frac(N, C dot n_c), $
onde $N$ é o total de amostras, $C$ o número de classes e $n_c$ o número
de amostras da classe $c$. Ainda assim, mesmo após a duplicação das
amostras por espelhamento horizontal, a disparidade entre as classes permaneceu
acentuada — especialmente para Hook e Uppercut, que apresentaram frequência
significativamente menor que Straight. Por isso, os pesos dessas duas classes
foram amplificados por um fator adicional de 2.5, escolhido empiricamente, resultando nos valores
apresentados na @tab-class-weights.

#figure(
  table(
    columns: (1fr, 1fr),
    stroke: 0.5pt,
    table.header(
      [*Classe*],
      [*Peso*],
    ),
    [Hook],     [3,11],
    [Straight], [0,60],
    [Uppercut], [4,65],
  ),
  caption: [Pesos por classe para o treinamento.],
) <tab-class-weights>

== _Callbacks_

O treinamento foi controlado por três _callbacks_ complementares. Da biblioteca do Tensorflow, foi utilizado o método
`EarlyStopping`, que interrompe o treinamento caso a acurácia de validação
não melhore por 15 épocas consecutivas, restaurando automaticamente os
pesos da melhor época observada. Também utilizou-se o `ReduceLROnPlateau`, com o objetivo de reduzir a taxa de
aprendizado pela metade sempre que a perda de validação estagna por 5
épocas, com valor mínimo de $10^(-6)$, permitindo refinamentos mais
finos na convergência. Por fim, o `ModelCheckpoint` salva em disco
apenas o modelo, em formato `keras`, com melhor acurácia de validação ao longo do
treinamento, garantindo que o modelo final avaliado seja o ótimo
encontrado e não necessariamente o da última época.