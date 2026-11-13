"""Como scikit-learn e R desempatam: casos feitos à mão, com empates forçados.

Base dos textos "Desempates" dos apps. Gera resultados/t_desempates.txt (scikit-learn) + a saída de desempates.R.
"""
import os
import subprocess
import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from comum import TMP, relatorio

warnings.filterwarnings('ignore')
AQUI = os.path.dirname(os.path.abspath(__file__))
Rs = r'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'


def raiz(m):
    t = m.tree_
    if t.node_count == 1:
        return 'sem divisão'
    return f'X{t.feature[0] + 1} <= {t.threshold[0]:.4g}'


def conta(xs):
    v, c = np.unique(np.asarray(xs).astype(str), return_counts=True)
    return ' '.join(f'{a}:{b}' for a, b in zip(v, c))


def main():
    L = ['## scikit-learn: DecisionTreeClassifier']
    P = np.array([[0, 0], [.1, .1], [1, 1], [1.1, 1.1]]); y = np.array([1, 1, 2, 2])
    L.append('atributos empatados, random_state 0..199: ' + conta([raiz(DecisionTreeClassifier(random_state=s).fit(P, y)) for s in range(200)]))
    P = np.array([[0, 5], [1, 5], [2, 5], [3, 5.]]); y = np.array([1, 2, 2, 1])
    L.append('limiares empatados (0,5 ou 2,5), max_depth=1: ' + conta([raiz(DecisionTreeClassifier(max_depth=1, random_state=s).fit(P, y)) for s in range(50)]))
    P = np.array([[0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5.]]); y = np.array([1, 1, 2, 2, 1, 1])
    L.append('limiares empatados 1,5 e 3,5: ' + conta([raiz(DecisionTreeClassifier(max_depth=1, random_state=s).fit(P, y)) for s in range(50)]))
    P = np.zeros((2, 2)); y = np.array([1, 2])
    L.append(f'folha 1×1: prevê {DecisionTreeClassifier().fit(P, y).predict(P[:1])[0]}')
    y = np.array([2, 1])
    L.append(f'folha 1×1, ordem das linhas trocada: prevê {DecisionTreeClassifier().fit(P, y).predict(P[:1])[0]}')
    P = np.zeros((3, 2)); y = np.array([3, 2, 1])
    L.append(f'folha 1×1×1: prevê {DecisionTreeClassifier().fit(P, y).predict(P[:1])[0]}')
    P = np.array([[0, 0], [1, 1], [0, 1], [1, 0.]]); y = np.array([1, 1, 2, 2])
    L.append(f'XOR de 4 pontos (ganho zero na raiz): {DecisionTreeClassifier(random_state=0).fit(P, y).tree_.node_count} nós')

    L.append('## scikit-learn: RandomForestClassifier')
    P = np.array([[0, 0], [1, 1], [.1, .1], [1.1, 1.1]]); y = np.array([1, 2, 1, 2])
    res = []
    for s in range(400):
        m = RandomForestClassifier(n_estimators=2, max_features=2, random_state=s).fit(P, y)
        pr = m.predict_proba([[.5, .5]])[0]
        if abs(pr[0] - pr[1]) < 1e-12:
            res.append(m.predict([[.5, .5]])[0])
    L.append(f'média de probabilidades empatada (de 400 florestas de 2 árvores): {conta(res)}')

    L.append('## scikit-learn: KNeighborsClassifier (algorithm="brute")')
    tr = np.array([[1, 0], [-1, 0], [5, 0.]]); te = np.array([[0, 0.]])
    for w in ('uniform', 'distance'):
        for yy in ([1, 2, 1], [2, 1, 2]):
            for p in (1, 2, 3):
                m = KNeighborsClassifier(n_neighbors=2, weights=w, algorithm='brute', p=p).fit(tr, yy)
                L.append(f'voto 1×1, weights={w}, p={p}, y={yy}: prevê {m.predict(te)[0]}')
    tr = np.array([[1, 0], [-1, 0], [0, 1], [3, 3.]]); y = np.array([1, 2, 2, 1])
    for o in ([0, 1, 2, 3], [1, 0, 2, 3], [2, 1, 0, 3]):
        for p in (1, 2, 3):
            m = KNeighborsClassifier(n_neighbors=1, algorithm='brute', p=p).fit(tr[o], y[o])
            d, i = m.kneighbors(te)
            L.append(f'três à mesma distância, k=1, p={p}, ordem {o}: vizinho = linha {o[i[0, 0]]}')
    # com mais pontos: a ordem dos empatados segue o índice?
    rng = np.random.default_rng(5)
    ok = tot = 0
    for _ in range(300):
        n = int(rng.integers(5, 40))
        T = rng.integers(-3, 4, size=(n, 2)).astype(float) * 0.5
        q = rng.integers(-3, 4, size=(4, 2)).astype(float) * 0.5
        for p in (1, 2, 3):
            k = int(rng.integers(1, n + 1))
            _, I = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=p).fit(T, np.arange(n) % 2).kneighbors(q)
            D = (np.abs(q[:, None, :] - T[None, :, :]) ** p).sum(2) ** (1 / p)
            for a in range(len(q)):
                ordem = np.lexsort((np.arange(n), D[a]))[:k]
                tot += 1
                ok += set(ordem) == set(I[a])
    L.append(f'grade com muitos empates: conjunto dos K vizinhos = os K primeiros por (distância, índice) em {ok} de {tot}')
    tr = np.array([[0, 0], [1, 0], [-2, 0.]]); y = np.array([2, 1, 1])
    m = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='brute').fit(tr, y)
    L.append(f'distância zero, weights=distance: prevê {m.predict(te)[0]}, probabilidades {m.predict_proba(te)[0]}')
    tr = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [9, 9.]]); y = np.array([1, 2, 1, 2, 1])
    m = KNeighborsClassifier(n_neighbors=4, weights='distance', algorithm='brute').fit(tr, y)
    L.append(f'pesos empatados 2×2 (d = 1): prevê {m.predict(te)[0]}, probabilidades {m.predict_proba(te)[0]}')
    m = KNeighborsClassifier(n_neighbors=4, weights='distance', algorithm='brute').fit(tr, 3 - y)
    L.append(f'pesos empatados 2×2, rótulos trocados: prevê {m.predict(te)[0]}')
    # posição exatamente sobre um ponto do treino: a expansão |x|² − 2x·y + |y|² dá distância exatamente 0?
    rng = np.random.default_rng(9)
    T = rng.normal(size=(30, 2)); yy = np.arange(30) % 3 + 1
    m = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='brute', p=2).fit(T, yy)
    d, _ = m.kneighbors(T)
    pr = m.predict_proba(T)
    L.append(f'consulta sobre os próprios pontos (p=2): menor distância máx = {d[:, 0].max():.3e}; probabilidade 1 em {int((pr.max(1) == 1).sum())} de 30')

    L.append('## scikit-learn: GNB, LDA, QDA no ponto equidistante')
    P = np.array([[-1, 0], [-1.2, .3], [1, 0], [1.2, .3]]); y = np.array([1, 1, 2, 2]); g = np.array([[0, .15]])
    for M in (GaussianNB(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(reg_param=0.1)):
        m = M.fit(P, y)
        L.append(f'{type(M).__name__}: probabilidades {m.predict_proba(g)[0]}, prevê {m.predict(g)[0]}; com y trocado prevê {M.fit(P, 3 - y).predict(g)[0]}')

    L.append('## scikit-learn: SVC, f(x) = 0')
    P = np.array([[-1, 0], [1, 0.]])
    for yy in ([-1, 1], [1, -1]):
        m = SVC(kernel='linear', C=1).fit(P, yy)
        L.append(f'y = {yy}: f(0) = {m.decision_function([[0, 0]])[0]:.3g}, classes_ = {m.classes_}, prevê {m.predict([[0, 0]])[0]}')

    # R
    out = TMP / 'desempates_r.txt'
    r = subprocess.run([Rs, '--vanilla', os.path.join(AQUI, 'desempates.R'), str(out)], capture_output=True)
    if r.returncode != 0:
        L.append('R falhou: ' + r.stderr.decode('latin-1')[-800:])
    else:
        L += out.read_text(encoding='utf-8').splitlines()
    relatorio('t_desempates', L)


if __name__ == '__main__':
    main()
