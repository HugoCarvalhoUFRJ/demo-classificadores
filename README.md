# Classificadores Interativos

Site com apps para aprender classificadores arrastando pontos. Cada mudança recalcula o classificador na hora.

| Guia | O que mostra | Página |
|---|---|---|
| Naive Bayes | Naive Bayes gaussiano, com médias, variâncias por classe e probabilidades a posteriori | `gnb.html` |
| LDA | Análise discriminante linear, com a covariância comum e as fronteiras retas | `lda.html` |
| QDA | Análise discriminante quadrática, com uma covariância por classe | `qda.html` |
| Margem máxima | Hiperplano separador, margem e vetores suporte | `margem-maxima.html` |
| Vetor suporte | Margem suave com orçamento `C`, folgas `ϵ` e `cost` equivalente | `vetor-suporte.html` |
| SVM | Kernels linear, polinomial, radial e sigmoide, com orçamento `C` e a chamada equivalente em R e Python | `svm-kernels.html` |
| Árvore | Árvore de decisão com partição exata, diagrama e poda | `arvore.html` |
| Floresta | Floresta aleatória e bagging, com erro OOB explicado e limite de folhas | `floresta.html` |
| KNN | K vizinhos mais próximos, com a vizinhança sob o cursor | `knn.html` |

Os apps multiclasse (Naive Bayes, LDA, QDA, Árvore, Floresta e KNN) aceitam até 4 classes e compartilham a mesma lista de conjuntos de dados.

O site é estático e não tem etapa de compilação. Todo o cálculo roda no navegador. A única dependência externa são as fontes do Google Fonts, e os apps funcionam sem elas.

Os pontos da opção "Fig. 9.3 do livro" foram estimados a partir da Fig. 9.3 de *An Introduction to Statistical Learning*. Os conjuntos "como na Fig. 4.6" e "como na Fig. 4.9" são gerados com a mesma ideia das figuras, não com os dados do livro. As Eqs. citadas nos apps usam a numeração do livro (edição com aplicações em Python).

## Sobre

Estes apps foram desenvolvidos por motivação pedagógica, a partir de perguntas dos alunos em sala de aula, para visualizar melhor os classificadores, usando IA: Claude, Opus 5.5 no esforço alto.

Disciplina Aprendizagem de Máquina I, da UFRJ, ministrada pelo Prof. Hugo Carvalho. Os materiais didáticos do curso estão em <https://github.com/HugoCarvalhoUFRJ/ap-maq>.

A notação é a mesma do livro [*An Introduction to Statistical Learning*](https://www.statlearning.com/).

O kernel sigmoide e seu parâmetro `r` não estão no capítulo 9 do livro. O `C` das guias Vetor suporte e SVM é o orçamento das Eqs. (9.12)–(9.15). Ele não é o argumento `cost` do e1071 nem o argumento `C` do `SVC` do scikit-learn: a guia mostra o `cost` equivalente.
