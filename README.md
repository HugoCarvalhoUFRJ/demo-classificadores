# SVM Interativo

Site com três apps para aprender máquinas de vetor suporte arrastando pontos. Cada mudança recalcula a solução na hora.

| Guia | O que mostra | Página |
|---|---|---|
| Margem máxima | Hiperplano separador, margem e vetores suporte | `margem-maxima.html` |
| Vetor suporte | Margem suave com orçamento `C`, folgas `ϵ` e `cost` equivalente | `vetor-suporte.html` |
| SVM com kernels | Kernels linear, polinomial, radial e sigmoide, com a chamada equivalente em R e Python | `svm-kernels.html` |

O site é estático e não tem etapa de compilação. Todo o cálculo roda no navegador. A única dependência externa são as fontes do Google Fonts, e os apps funcionam sem elas.

Os pontos iniciais da opção "figura" foram estimados a partir de a Figura 9.3 de *An Introduction to Statistical Learning*. As equações citadas nos apps usam a numeração do capítulo 9 desse livro.

## Sobre

Estes apps foram desenvolvidos em sala de aula, a pedido dos alunos, para visualizar melhor os classificadores, usando IA: Claude, Opus 5.5 no esforço alto.

Disciplina Aprendizagem de Máquina I, da UFRJ, ministrada pelo Prof. Hugo Carvalho. Os materiais didáticos do curso estão em <https://github.com/HugoCarvalhoUFRJ/ap-maq>.

A notação é a mesma do livro [*An Introduction to Statistical Learning*](https://www.statlearning.com/).

O kernel sigmoide e seu parâmetro `r` não estão no capítulo 9 do livro. O `cost` é o argumento `cost` do e1071 e o argumento `C` do `SVC` do scikit-learn, diferente do orçamento `C` das equações (9.12)–(9.15).
