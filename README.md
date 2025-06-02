# Relatório do Tech Challenge - Fase 1: Modelo Preditivo de Custos Médicos

## 1. Introdução

Este relatório detalha o desenvolvimento de um modelo preditivo de regressão para estimar os custos médicos individuais cobrados por seguros de saúde, conforme solicitado no Tech Challenge da Fase 1. O objetivo principal é construir um modelo capaz de fazer previsões confiáveis com base nas características dos beneficiários.

O projeto seguiu um pipeline padrão de ciência de dados, utilizando Python e bibliotecas como Pandas, Matplotlib, Seaborn, Scikit-learn e Statsmodels.

## 2. Fonte de Dados

Utilizou-se o dataset "Medical Cost Personal Datasets", originalmente disponibilizado no Kaggle. O ficheiro `insurance.csv` contém 1338 registos e 7 colunas:

-   `age`: Idade do beneficiário principal (numérica).
-   `sex`: Género do contratante do seguro (categórica: female, male).
-   `bmi`: Índice de Massa Corporal (IMC) (numérica).
-   `children`: Número de filhos/dependentes cobertos pelo seguro (numérica).
-   `smoker`: Se o beneficiário é fumador (categórica: yes, no).
-   `region`: Área residencial do beneficiário nos EUA (categórica: northeast, southeast, southwest, northwest).
-   `charges`: Custos médicos individuais cobrados pelo seguro (numérica - variável alvo).

Referência: [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## 3. Exploração e Visualização de Dados

A fase inicial envolveu o carregamento e a exploração do dataset para compreender as suas características.

**Resumo Inicial:**

-   O dataset possui 1338 linhas e 7 colunas.
-   Não foram encontrados valores ausentes.
-   As variáveis  `sex`,  `smoker`  e  `region`  são do tipo 'object' (categóricas) e as restantes são numéricas (`int64`  ou  `float64`).

**Estatísticas Descritivas (Variáveis Numéricas):**

-   **Age:**  Varia entre 18 e 64 anos, com média de 39.2 anos.
-   **BMI:**  Varia entre 15.96 e 53.13, com média de 30.66. A média está acima da faixa considerada ideal (18.5 a 24.9).
-   **Children:**  Varia entre 0 e 5, com média de 1.1 filhos/dependentes.
-   **Charges:**  Varia significativamente, de $1,121.87 a $63,770.43, com média de $13,270.42 e um desvio padrão elevado ($12,110.01), indicando grande dispersão nos custos.

_(Consulte `data_exploration_summary.txt` para detalhes)_

**Visualizações:** Foram gerados diversos gráficos para visualizar as distribuições e relações entre as variáveis (encontram-se na pasta `plots`):

-   **Histogramas (`numerical_histograms.png`):**  Mostraram a distribuição das variáveis numéricas.  `charges`  apresenta uma forte assimetria à direita (muitos custos baixos, poucos custos muito altos).
-   **Countplots (`categorical_countplots.png`):**  Revelaram a distribuição das variáveis categóricas. Há um número ligeiramente maior de homens, uma maioria de não-fumadores e uma distribuição relativamente equilibrada entre as regiões.
-   **Boxplots (`charges_vs_categorical_boxplots.png`):**  Indicaram que fumadores (`smoker=yes`) têm custos significativamente mais elevados. O género (`sex`) e a região (`region`) parecem ter um impacto menor nos custos médios.
-   **Scatterplots (`charges_vs_numerical_scatterplots.png`):**  O gráfico de  `charges`  vs.  `age`  mostrou uma tendência geral de aumento dos custos com a idade, com uma clara separação entre fumadores e não-fumadores (fumadores têm custos mais altos em todas as idades). O gráfico de  `charges`  vs.  `bmi`  também sugere custos mais elevados para IMCs maiores, especialmente entre fumadores.
-   **Matriz de Correlação (`correlation_matrix.png`):**  Confirmou a forte correlação positiva entre  `charges`  e  `smoker`  (após conversão para numérico).  `age`  e  `bmi`  também mostraram correlação positiva com  `charges`, embora mais fraca que  `smoker`.

## 4. Pré-processamento de Dados

Com base na exploração, realizou-se o seguinte pré-processamento:

1.  **Valores Ausentes:**  Confirmou-se que não havia valores ausentes, não sendo necessária imputação.
2.  **Conversão de Variáveis Categóricas:**  As variáveis  `sex`,  `smoker`  e  `region`  foram convertidas em formato numérico utilizando One-Hot Encoding. A opção  `drop_first=True`  foi usada para evitar multicolinearidade, resultando nas seguintes novas colunas:  `sex_male`,  `smoker_yes`,  `region_northwest`,  `region_southeast`,  `region_southwest`. A categoria base para cada variável original (female, no, northeast) é representada quando todas as colunas correspondentes são 0.

O dataset pré-processado foi guardado em `02_preprocessed_data.csv`.

## 5. Modelagem e Treinamento

**Escolha do Modelo:** Optou-se por um modelo de **Regressão Linear**, uma técnica comum para problemas de previsão de valores contínuos e que permite interpretar facilmente o impacto de cada variável.

**Divisão dos Dados:** O dataset pré-processado foi dividido em conjuntos de treino (80%, 1070 amostras) e teste (20%, 268 amostras) para permitir a avaliação do modelo em dados não vistos durante o treinamento.

**Treinamento:** O modelo de Regressão Linear (`LinearRegression` do Scikit-learn) foi treinado utilizando o conjunto de treino (`X_train`, `y_train`).

## 6. Avaliação do Modelo

O desempenho do modelo treinado foi avaliado no conjunto de teste (`X_test`, `y_test`).

**Métricas de Avaliação:**

-   **Mean Absolute Error (MAE):**  $4,181.19
-   **Mean Squared Error (MSE):**  33,596,915.85
-   **Root Mean Squared Error (RMSE):**  $5,796.28
-   **R-squared (R²):**  0.7836

_(Consulte `results/evaluation_metrics.txt` para detalhes)_

**Interpretação:**

-   O  **R² de 0.7836**  indica que aproximadamente 78.36% da variabilidade nos custos médicos no conjunto de teste é explicada pelo modelo. Este é um valor razoável, sugerindo que o modelo tem um bom poder preditivo.
-   O  **RMSE de $5,796.28**  representa o erro médio das previsões em termos da unidade original (dólares). Considerando a grande variação nos custos (desvio padrão de ~$12,110), este erro pode ser considerado moderado.

## 7. Validação Estatística

Para uma análise mais aprofundada da significância estatística das variáveis e dos intervalos de confiança, utilizou-se a biblioteca `statsmodels` para ajustar um modelo OLS (Ordinary Least Squares) no conjunto de treino.

**Sumário do Modelo OLS (Statsmodels):** _(Consulte `results/model_summary.txt` para o sumário completo)_

**Principais Insights:**

-   **Significância das Variáveis:**  A maioria das variáveis (`age`,  `bmi`,  `children`,  `smoker_yes`) apresentou um  **p-value muito baixo (próximo de 0.000)**, indicando que são estatisticamente significativas para prever os custos médicos (rejeita-se a hipótese nula de que o coeficiente é zero).
-   A variável  `sex_male`  e as variáveis de  `region`  apresentaram p-values mais altos, sugerindo que o género e a região (considerando  `northeast`  como base) têm um impacto menos significativo nos custos quando comparados com idade, IMC e, principalmente, o facto de ser fumador.
-   **Coeficientes:**
    -   `smoker_yes`: O coeficiente positivo e muito elevado (~23650) confirma que ser fumador é o fator com maior impacto no aumento dos custos.
    -   `age`: Cada ano adicional de idade aumenta o custo em aproximadamente $257.
    -   `bmi`: Cada ponto adicional no IMC aumenta o custo em aproximadamente $337.
    -   `children`: Cada filho/dependente adicional aumenta o custo em aproximadamente $425.
-   **R² (Treino):**  O R² no conjunto de treino foi de 0.742, ligeiramente inferior ao R² no teste (0.784). Isto é um pouco invulgar (normalmente o R² de treino é maior), mas a diferença não é drástica e pode dever-se à aleatoriedade da divisão treino/teste.
-   **Intervalos de Confiança:**  Os intervalos de confiança (colunas  `[0.025`  e  `0.975]`) para os coeficientes significativos (`age`,  `bmi`,  `children`,  `smoker_yes`) não incluem zero, reforçando a sua significância estatística.

## 8. Análise Visual dos Resultados

Foram gerados gráficos para analisar a performance do modelo no conjunto de teste:

-   **Previsões vs. Valores Reais (`plots/predictions_vs_actual.png`):**  Este gráfico mostra os valores previstos pelo modelo em comparação com os valores reais. Idealmente, os pontos deveriam alinhar-se na linha diagonal tracejada (onde previsão = real). Observa-se uma tendência geral de alinhamento, mas com dispersão, especialmente para custos mais elevados. O modelo parece ter mais dificuldade em prever com precisão os custos muito altos.
-   **Distribuição dos Resíduos (`plots/residuals_plot.png`):**  Os resíduos (diferença entre valores reais e previstos) devem idealmente seguir uma distribuição normal centrada em zero. O histograma mostra que a distribuição dos resíduos está razoavelmente centrada perto de zero, mas com alguma assimetria à direita, indicando que o modelo tende a subestimar mais frequentemente os custos do que a sobrestimá-los, especialmente para os casos de custos mais elevados.

## 9. Conclusão

O modelo de Regressão Linear desenvolvido demonstrou ser capaz de explicar uma parte significativa (aproximadamente 78%) da variação nos custos médicos individuais, com base nas características fornecidas. As variáveis mais impactantes foram ser fumador, idade e IMC.

O modelo apresenta um desempenho razoável, mas a análise dos resíduos e do gráfico de previsões vs. reais sugere que há espaço para melhorias, especialmente na previsão de custos mais elevados. Possíveis próximos passos poderiam incluir:

-   Engenharia de Features: Criar novas variáveis (ex: interação entre  `bmi`  e  `smoker`).
-   Transformação de Variáveis: Aplicar transformações (ex: logarítmica) na variável  `charges`  para lidar com a assimetria.
-   Modelos Alternativos: Experimentar outros algoritmos de regressão (ex: Árvores de Decisão, Random Forest, Gradient Boosting) que podem capturar relações não-lineares de forma mais eficaz.

## 10. Instruções para Entregáveis Adicionais

Conforme solicitado no desafio:

1.  **Código-Fonte:**  O código Python desenvolvido está organizado no arquivo   `load_explore_data.py`,  `visualize_preprocess.py`  e  `model_train_evaluate.py`.
2.  **Repositório GitHub:**  O código, juntamente com este relatório, os gráficos e os resultados, deve ser carregado para um repositório no GitHub.
3.  **Vídeo de Apresentação:**  Um vídeo (máximo 10 minutos) deve ser gravado e disponibilizado (ex: YouTube), apresentando o passo a passo do projeto, a fonte de dados, a criação do modelo e a análise dos resultados. O link do vídeo e do repositório GitHub devem ser enviados como parte da entrega final.
