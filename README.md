# Classificadores Interativos

Apps para aprender classificadores arrastando pontos num gráfico. A cada movimento o classificador é recalculado e as regiões de decisão, a fronteira e os números na tela se atualizam na hora.

**Acesse:** <https://hugocarvalho.mat.br/demo-classificadores/>

Cada guia tem endereço próprio, útil para mandar à turma: por exemplo, [`#knn`](https://hugocarvalho.mat.br/demo-classificadores/#knn) ou [`#svm-kernels`](https://hugocarvalho.mat.br/demo-classificadores/#svm-kernels).

## Os apps

| Guia | O que mostra | Reproduz | Página |
|---|---|---|---|
| Naive Bayes | Naive Bayes gaussiano, com médias, variâncias por classe e probabilidades a posteriori | `GaussianNB` | [`gnb.html`](gnb.html) |
| LDA | Análise discriminante linear, com a covariância comum e as fronteiras retas | `LinearDiscriminantAnalysis` | [`lda.html`](lda.html) |
| QDA | Análise discriminante quadrática, com uma covariância por classe | `QuadraticDiscriminantAnalysis` | [`qda.html`](qda.html) |
| Margem máxima | Hiperplano separador, margem e vetores suporte (solução exata, pelas envoltórias convexas) | | [`margem-maxima.html`](margem-maxima.html) |
| Vetor suporte | Margem suave com orçamento `C`, folgas `ϵ` e `cost` equivalente | | [`vetor-suporte.html`](vetor-suporte.html) |
| SVM | Kernels linear, polinomial, radial e sigmoide, com orçamento `C` | `SVC`, `e1071::svm` | [`svm-kernels.html`](svm-kernels.html) |
| Árvore | Árvore de decisão com partição exata, diagrama e poda por α; passar o mouse num nó realça a região dele, e passar o mouse no gráfico acende o ramo até a folha | `DecisionTreeClassifier` | [`arvore.html`](arvore.html) |
| Floresta | Floresta aleatória e bagging, com erro OOB explicado, profundidade máxima e mínimo por folha | `RandomForestClassifier` | [`floresta.html`](floresta.html) |
| KNN | K vizinhos mais próximos, com pesos iguais ou 1/d, distância com p = 1, 2 ou 3 e a vizinhança sob o cursor | `KNeighborsClassifier` | [`knn.html`](knn.html) |

A coluna "Reproduz" indica a implementação de referência (scikit-learn ou R) cujo resultado o app replica.

## O que dá para fazer

- **Arrastar, acrescentar e remover pontos** com mouse, toque ou teclado (setas movem o ponto focado, Shift dá passo maior, Delete remove). Durante o arraste, a fronteira anterior aparece em cinza para comparação.
- **Passar o mouse numa área vazia** para ver Pr(Y = k | X = x) naquele lugar. A narração embaixo do gráfico explica o ponto sob o cursor e avisa quando não há solução, a matriz é singular ou o otimizador não convergiu.
- **Escolher o conjunto de dados.** Os apps multiclasse (Naive Bayes, LDA, QDA, Árvore, Floresta e KNN) aceitam até 4 classes e compartilham a mesma lista, com o n de cada conjunto:
  - *Duas classes:* anel, parábola, XOR, duas luas, espirais, tabuleiro, Fig. 9.3 do livro, nuvens correlacionadas, duas nuvens com a mesma covariância ou com covariâncias diferentes (como na Fig. 4.9), nuvens muito sobrepostas, mistura de gaussianas (como na Fig. 2.13) e classes desbalanceadas.
  - *Três ou mais classes:* mistura de gaussianas, três nuvens (como na Fig. 4.6), três nuvens com covariâncias diferentes, três faixas diagonais, alvo e quatro quadrantes.

  Os dados vêm de geradores com semente fixa: a turma inteira vê sempre os mesmos pontos.
- **Aumentar o ruído σ**, para sobrepor as classes aos poucos.
- **Ligar um conjunto de teste**, outra amostra da mesma distribuição que não entra no ajuste, e comparar o erro de treino com o de teste.
- Com o teste ligado, ver a **matriz de confusão** (previsto nas linhas, verdadeiro nas colunas, como na Tabela 4.4) e as medidas da Tabela 4.7, com a classe positiva escolhível, mais o F1 e, com três ou mais classes, as médias macro e micro.
- Nos conjuntos de mistura de gaussianas, ver a **fronteira de Bayes** tracejada e o erro de Bayes.
- **Copiar o código equivalente** em R e em Python, com todos os parâmetros explícitos. Quando a função de R não dá o mesmo resultado, a própria tela diz por quê.
- Abrir **Desempates**, que explica como o app, o scikit-learn e o R resolvem empates.

O ruído, o conjunto de teste e a mistura de gaussianas também existem na guia SVM. O tema claro ou escuro é escolhido no cabeçalho e vale para todas as guias.

## Como funciona

O site é estático e não tem etapa de compilação. Cada app é um único arquivo HTML com CSS e JavaScript embutidos, e todo o cálculo roda no navegador, rápido o bastante para refazer o ajuste a cada quadro do arraste. A única dependência externa são as fontes do Google Fonts, porém os apps funcionam sem elas.

Onde existe solução exata, o app a usa (margem máxima, árvores, KNN). As SVMs com orçamento `C` resolvem o dual por pontos interiores, com critério de convergência verificável. A floresta usa uma semente fixa por árvore, para que o arraste não sorteie tudo de novo; o botão "Sortear de novo" troca a semente.

Para usar sem internet, baixe o repositório e abra o `index.html` no navegador. Cada app também funciona aberto sozinho.

## Verificação

A pasta [`testes/`](testes/) tem os scripts usados para conferir os apps (veja [`testes/LEIAME.md`](testes/LEIAME.md)):

- o solver de cada app é comparado, em 1000 conjuntos de dados aleatórios com variedade planejada, com o scikit-learn e, quando possível, com o R (`MASS`, `e1071`, `class`, `kknn`, `rpart`, `randomForest`) e com otimizadores genéricos (cvxopt, HiGHS);
- com Playwright, testes ponta a ponta leem da tela os pontos, as cores das regiões, os números e o código mostrado, e executam literalmente as linhas de R e Python exibidas;
- centenas de interações aleatórias por app procuram `NaN`, erros no console e estouro de layout;
- axe-core confere a acessibilidade de todas as páginas, nos temas claro e escuro;
- tudo isso roda servido por HTTP e aberto direto do disco, dentro das guias e com cada app sozinho, no computador e numa tela de celular de 390 px com toque simulado.

## Estrutura do repositório

```
index.html          página principal com as 9 guias
gnb.html            Naive Bayes gaussiano
lda.html            análise discriminante linear
qda.html            análise discriminante quadrática
margem-maxima.html  classificador de margem máxima
vetor-suporte.html  classificador de vetor suporte
svm-kernels.html    SVM com kernels
arvore.html         árvore de decisão
floresta.html       floresta aleatória
knn.html            K vizinhos mais próximos
testes/             scripts de verificação (Python, R e Playwright)
```

## Sobre

Estes apps foram desenvolvidos por motivação pedagógica, a partir de perguntas dos alunos em sala de aula, para visualizar melhor os classificadores, usando IA: Claude, Opus 5.5 no esforço alto.

Disciplina Aprendizagem de Máquina I, da UFRJ, ministrada pelo Prof. Hugo Carvalho. Os materiais didáticos do curso estão em <https://github.com/HugoCarvalhoUFRJ/ap-maq>.

A notação é a mesma do livro [*An Introduction to Statistical Learning*](https://www.statlearning.com/). As Eqs. citadas nos apps usam a numeração do livro (edição com aplicações em Python).

Os pontos da opção "Fig. 9.3 do livro" foram estimados a partir da Fig. 9.3 de *An Introduction to Statistical Learning*. Os conjuntos "como na Fig. 2.13", "como na Fig. 4.6" e "como na Fig. 4.9" são gerados com a mesma ideia das figuras, não com os dados do livro.

O kernel sigmoide e seu parâmetro `r` não estão no capítulo 9 do livro. O `C` das guias Vetor suporte e SVM é o orçamento das Eqs. (9.12)–(9.15). Ele não é o argumento `cost` do e1071 nem o argumento `C` do `SVC` do scikit-learn: a guia mostra o `cost` equivalente.

## Licença

© 2026 Hugo Carvalho. Este trabalho está licenciado sob a [Creative Commons Atribuição-NãoComercial 4.0 Internacional (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.pt-br). Você pode usar, copiar e adaptar os apps, inclusive em outras aulas, desde que cite a fonte e não os use para fins comerciais. O texto completo está em [`LICENSE`](LICENSE).
