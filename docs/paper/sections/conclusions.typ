= Considerações Finais

O trabalho apresentando cumpriu com o objetivo de desenvolver uma _pipeline_ totalmente automatizada para resolver o problema da detecção e classificação de golpes em partidas de boxe. 

O uso de ferramentas tecnológicas já consolidadas, como TensorFlow, YOLO e outras, possibilitou o alcance de ótimos resultados para um problema muito complexo: realizar predição com modelos de _machine learning_ em sequências temporais (vídeos), considerando características como velocidade, aceleração e posição dos lutadores.

Além disso, a solução de traqueamento possiblita que essa solução seja utilizada em ecossistemas reais de lutas de boxe, visto a persistência dos lutadores durante a partida. Como trabalho futuro, busca-se a implementação de um sistema de pontuação para cada boxeador, considerando os golpes inferidos.

Por fim, a fim de melhorar o modelo, sugere-se uma nova anotação de dados, com maior diversidade de lutadores, iluminação e ambientes, a fim de aumentar a acurácia final ainda mais.

Como última observação, apontamos que este trabalho fez uso de assistentes de Inteligência Artificial Generativa (IAGen) como ferramenta de apoio para pesquisa, _boilerplate_, depuração e revisão. Todavia,  todas as decisões de projeto, a modelagem, os experimentos e a validação dos resultados foram feitos e conferidos pelos integrantes do grupo. Mais informações podem ser encontradas no arquivo `PROMPT.md` no #link("https://github.com/leonardosm14/boxe.ml/blob/main/PROMPTS.md", "repositório").