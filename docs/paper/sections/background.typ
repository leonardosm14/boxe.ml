= Revisão Bibliográfica

Primeiramente, para o desenvolvimento do projeto, realizou-se uma pesquisa acerca das tecnologias existentes. Percebeu-se a existência de soluções muito complexas, como o `Jabbr.ai`#footnote[Jabbr.ai: https://jabbr.ai/], porém que não estão disponibilizadas em código aberto.

Além disso, foram encontradas soluções com _Machine Learning_ (ML), porém que não satisfazem os requisitos de identificação dos seis golpes primordiais do boxe, que só funcionam para um único lutador e em tempo real, sem possibilidade de análise de lutas em vídeo.

Tendo esse contexto, buscaram-se _datasets_ com dados já anotados, a fim de diminuir o esforço e dificuldade do tema. Com isso, foi encontrado o artigo#footnote[Aponta-se, entretanto, que o trabalho é um _pre-print_. Ou seja, não foi oficialmente publicado em _journals_, revistas ou conferências.] *"BoxingVI: A Multi-Modal Benchmark for Boxing Action Recognition and Localization"* @boxingVI. No trabalho citado, foi criado um _banchmark_ a partir da anotação de dados de 20 vídeos, com um total de 6.915 clipes anotados.

Por fim, realizou-se um estudo acerca do trabalho *"AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time"* @alphapose. Tal ferramenta foi utilizada pelo paper do _benchmark_ BoxingVI @boxingVI para detecção de objetos (pessoas). Entendeu-se importante compreender o formato dos dados de anotação e dos esqueletos criados, a fim de estudar a viabilidade dos dados para utilização do _dataset_.