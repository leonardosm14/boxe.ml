= Resultados do Treinamento

O treinamento do modelo descrito na seção anterior foi realizado utilizando o método `fit` do TensorFlow. Foi definido um máximo de 100 épocas, considerando a utilização dos _callbacks_ apresentados anteriormente, e um tamanho de _batch_ de 32 amostras. O treinamento foi interrompido antecipadamente na época 48 pelo mecanismo de _Early Stopping_, após 15 épocas consecutivas sem melhoria na acurácia de validação (`val_accuracy`), totalizando aproximadamente 32 minutos de execução. A @tab-training apresenta, para a última época de treinamento e para o melhor modelo restaurado, as métricas de acurácia e perdas (`loss`) dos conjuntos de treino e validação. A acurácia representa a proporção de amostras classificadas corretamente, enquanto a `loss` corresponde ao valor da função de custo de entropia cruzada categórica, sendo desejáveis valores menores. As métricas obtidas no conjunto de validação são as mais relevantes para avaliar a capacidade de generalização do modelo, pois refletem seu desempenho em dados não utilizados durante o treinamento. O melhor modelo foi obtido na época 33, alcançando 85,16% de acurácia de validação e perda de 44,70%. Embora a última época tenha apresentado acurácia de treino ligeiramente superior (81,38%), os pesos restaurados da época 33 foram mantidos por apresentarem o melhor desempenho no conjunto de validação, indicando boa capacidade de generalização e ausência de sinais significativos de _overfitting_.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    table.header(
      [*Época*],
      [*Acc. Treino*],
      [*Loss Treino*],
      [*Acc. Val.*],
      [*Loss Val.*],
    ),
    [1],  [0,2809], [2,0910], [0,2792], [1,4372],
    [2],  [0,3211], [1,9537], [0,3900], [1,2142],
    [⋮],  [⋮],      [⋮],      [⋮],      [⋮],
    [*33*], [0,7939], [0,8769], [*0,8516*], [0,4470],
    [⋮],  [⋮],      [⋮],      [⋮],      [⋮],
    [47], [0,8086], [0,8088], [0,8442], [0,4553],
    [48], [0,8138], [0,8162], [0,8353], [0,4687],
  ),
  caption: [Evolução das métricas de treino e validação ao longo do treinamento.],
) <tab-training>

Após a validação do modelo com os vídeos $V_5$, $V_9$ e $V_10$, os resultados se mostraram consistentes, atingindo uma acurácia de 83% sobre 899 amostras. A @tab-resultados detalha as métricas por classe: precisão, revocação e F1-score. A *precisão* mede a proporção de predições positivas que estão corretas; a *revocação* mede a proporção de amostras reais de cada classe que foram corretamente identificadas; e o *F1-score* é a média harmônica entre ambas, sendo a métrica mais equilibrada quando há desbalanceamento entre classes. A média macro, que trata todas as classes com peso igual, atingiu 78%, enquanto a média ponderada, que considera o número de amostras de cada classe, acompanhou a acurácia geral de 83%, refletindo o bom desempenho na classe majoritária (Straight). O menor F1-score foi observado na classe Hook (0,68), o que era esperado dado o menor volume de amostras e a maior similaridade visual desse golpe com as demais categorias.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    align: center,
    table.header(
      [*Classe*],
      rotate(-90deg, reflow: true)[*Precisão*],
      rotate(-90deg, reflow: true)[*Revocação*],
      rotate(-90deg, reflow: true)[*F1-score*],
      rotate(-90deg, reflow: true)[*Suporte*],
    ),
    [Hook],              [0,68], [0,69], [0,68], [172],
    [Straight],          [0,91], [0,91], [0,91], [519],
    [Uppercut],          [0,76], [0,76], [0,76], [208],
    [*Média macro*],       [0,78], [0,78], [0,78], [899],
    [*Média ponderada*],   [0,83], [0,83], [0,83], [899],
  ),
  caption: [Métricas de classificação por classe no conjunto de teste.],
) <tab-resultados>

A @fig-matriz apresenta a matriz de confusão obtida no conjunto de teste.
A diagonal principal concentra a grande maioria das predições corretas:
118 de 172 Hooks (69%), 471 de 519 Straights (91%) e 158 de 208 Uppercuts
(76%). Os erros mais frequentes ocorrem na classe Hook, que confunde 25
amostras com Straight e 29 com Uppercut -- o que é esperado, dado que o
Hook é biomecânicamente mais próximo dos demais golpes quando observado
apenas pelo esqueleto 2D. Straight e Uppercut apresentam confusão mútua
mais baixa (21 amostras em cada direção), refletindo a maior distinção
visual entre esses movimentos.

#figure(
  image("../img/matrix.pdf", width: 94.2%),
  caption: [Matriz de confusão -- $V_5$, $V_9$ e $V_10$.],
) <fig-matriz>