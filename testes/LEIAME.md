# Testes

Os scripts ficam aqui. Tudo o que eles instalam ou gravam fica em `C:\Users\hugom\AppData\Local\classificadores-testes`, fora do OneDrive. Para remover, basta apagar essa pasta.

- `venv/`: Python criado com `--system-site-packages` a partir do conda base. Herda o scikit-learn 1.7.2 sem modificar o base. Instalados só aqui: `playwright`, `cvxopt` e `axe-playwright-python`.
- `Rlib/`: `e1071`, `proxy`, `randomForest` e `kknn` (com as dependências `igraph`, `cli`, `rlang` etc.) para o R 4.5.1. `MASS`, `class` e `rpart` são os do próprio R.
- `resultados/`: relatórios `.txt` e capturas de tela.
- O Node que roda os solvers é o que vem dentro do pacote `playwright`. O navegador é o Chrome já instalado, aberto com perfil temporário.

## Como rodar (Git Bash)

```bash
cd testes
source env.sh               # aponta TEMP, caches e bibliotecas para a pasta de testes
"$PY" t1_svm.py             # SVM com orçamento C: 1000 conjuntos (cvxopt, HiGHS, SVC)
"$PY" t1_gerativos.py       # GNB, LDA, QDA: 1000 conjuntos cada (scikit-learn, MASS, e1071)
"$PY" t1_arvores.py knn     # também: arvore, floresta, todos (KNN com p e pesos; class::knn e kknn)
"$PY" t_desempates.py       # empates forçados: o que scikit-learn e R fazem (base dos textos "Desempates")
"$PY" t_casca.py            # a casca dos seis apps multiclasse é idêntica? ("aplicar" copia a do knn.html)
"$PY" t_nav.py todos        # ou e2e, aleatorio, axe, contextos, desempenho, capturas
"$PY" t_rapido.py knn       # verificação rápida durante o desenvolvimento
```

- `dados.py` gera os 1000 conjuntos (separáveis, sobrepostos, aleatórios, grade com empates e duplicatas, casos extremos).
- `comum.py` extrai o bloco `/*SOLVER-START*/ … /*SOLVER-END*/` de cada app e o executa no Node.
- `ref_r.R` roda as referências em R, e `desempates.R` os casos de empate em R.
- `t_investiga_rpart.py` compara nó a nó a árvore do app com a do `rpart` e diz a causa de cada diferença.
- `t_investiga_rf.py` refaz os casos da floresta que fogem da folga (Monte Carlo com B = 500) e mostra os de estrutura diferente.
