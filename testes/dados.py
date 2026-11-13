"""Gera os conjuntos de dados dos testes: exatamente 1000 por app, com variedade planejada.

Tipos: 'separavel' (250), 'sobreposto' (250), 'aleatorio' (200), 'grade' (200, com empates e duplicatas),
'extremo' (100, casos feitos à mão). Rótulos 1..K (1 = azul, 2 = roxo, 3 = laranja, 4 = verde).
"""
import numpy as np

X0, X1, Y0, Y1 = -1.73, 3.52, -1.77, 3.75
CX, CY = (X0 + X1) / 2, (Y0 + Y1) / 2


def _clip(P):
    P = np.array(P, float)
    P[:, 0] = np.clip(P[:, 0], X0 + 0.03, X1 - 0.03)
    P[:, 1] = np.clip(P[:, 1], Y0 + 0.03, Y1 - 0.03)
    return P


def _centros(rng, K, raio):
    t0 = rng.uniform(0, 2 * np.pi)
    return [(CX + raio * np.cos(t0 + 2 * np.pi * k / K), CY + raio * np.sin(t0 + 2 * np.pi * k / K)) for k in range(K)]


def _nuvens(rng, K, raio, sd, nmin=4, nmax=18):
    P, y = [], []
    for k, (a, b) in enumerate(_centros(rng, K, raio)):
        m = rng.integers(nmin, nmax + 1)
        th = rng.uniform(0, np.pi)
        s1, s2 = sd * rng.uniform(0.5, 1.4), sd * rng.uniform(0.5, 1.4)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Z = rng.normal(size=(m, 2)) * [s1, s2]
        P.append(Z @ R.T + [a, b])
        y += [k + 1] * m
    return _clip(np.vstack(P)), np.array(y)


def _garante_classes(rng, P, y, K):
    """Toda classe fica com pelo menos um ponto."""
    for k in range(1, K + 1):
        if not (y == k).any():
            y[rng.integers(len(y))] = k
    for k in range(1, K + 1):
        if not (y == k).any():
            # sorteios acima podem ter roubado o único ponto de outra classe
            P = np.vstack([P, [rng.uniform(X0, X1), rng.uniform(Y0, Y1)]])
            y = np.append(y, k)
    return P, y


def extremo(i, rng, K):
    """Casos degenerados feitos à mão; i escolhe o padrão, rng só desloca."""
    d = rng.uniform(-0.4, 0.4, 2)
    ks = np.arange(1, K + 1)
    tipo = i % 12
    if tipo == 0:  # um ponto por classe
        P = np.array([[0.2 + 1.0 * k, 1.0 + 0.7 * ((-1) ** k)] for k in range(K)]) + d
        y = ks.copy()
    elif tipo == 1:  # dois pontos por classe
        P = np.array([[0.2 + k, 0.2 + 0.5 * j + 0.9 * k] for k in range(K) for j in range(2)]) + d
        y = np.repeat(ks, 2)
    elif tipo == 2:  # todos colineares na horizontal
        P = np.column_stack([np.linspace(-1, 3, 12), np.full(12, 1.0)]) + d
        y = (np.arange(12) % K) + 1
    elif tipo == 3:  # colineares na diagonal
        t = np.linspace(-1, 2.5, 12)
        P = np.column_stack([t, t]) + d
        y = (np.arange(12) % K) + 1
    elif tipo == 4:  # duplicatas com rótulos diferentes
        Q = np.array([[0.5, 0.5], [1.5, 1.5], [0.5, 2.0], [2.0, 0.5]]) + d
        P = np.vstack([Q] * K)
        y = np.concatenate([[k] * 4 for k in ks])
    elif tipo == 5:  # todos os pontos no mesmo lugar menos um
        P = np.tile([[1.0, 1.0]], (10, 1)) + d
        P[0] = [2.5, 2.5]
        y = (np.arange(10) % K) + 1
    elif tipo == 6:  # uma classe com um só ponto
        P = np.vstack([rng.normal(size=(14, 2)) * 0.5 + [1, 1]])
        y = (np.arange(14) % max(K - 1, 1)) + 2 if K > 1 else np.ones(14, int)
        y[y > K] = K
        y[0] = 1
    elif tipo == 7:  # atributo constante em uma classe
        P = rng.normal(size=(16, 2)) * 0.6 + [1, 1]
        y = (np.arange(16) % K) + 1
        P[y == 1, 1] = 1.2
    elif tipo == 8:  # atributo constante em todas
        P = np.column_stack([rng.uniform(-1, 3, 16), np.full(16, 0.7)])
        y = (np.arange(16) % K) + 1
    elif tipo == 9:  # pontos quase coincidentes
        P = np.array([[1.0, 1.0]]) + rng.normal(size=(12, 2)) * 1e-6
        y = (np.arange(12) % K) + 1
    elif tipo == 10:  # nos cantos do gráfico
        P = np.array([[X0 + 0.04, Y0 + 0.04], [X1 - 0.04, Y0 + 0.04], [X0 + 0.04, Y1 - 0.04], [X1 - 0.04, Y1 - 0.04]] * 3, float)
        y = (np.arange(12) % K) + 1
    else:  # classes muito desbalanceadas
        P = np.vstack([rng.normal(size=(24, 2)) * 0.7 + [1, 1], rng.normal(size=(K - 1, 2)) * 0.2 + [2.5, 2.5]])
        y = np.concatenate([np.ones(24, int), np.arange(2, K + 1)])
    return _clip(P), np.asarray(y, int)


def gera(semente=2026, K_de=(2, 3, 4), pesos=(0.5, 0.3, 0.2)):
    """Devolve lista com exatamente 1000 dicts: {tipo, P (n×2), y, K}."""
    casos = []
    r = np.random.default_rng(semente)
    plano = ['separavel'] * 250 + ['sobreposto'] * 250 + ['aleatorio'] * 200 + ['grade'] * 200 + ['extremo'] * 100
    for j, tipo in enumerate(plano):
        rng = np.random.default_rng(r.integers(2**63))
        K = int(rng.choice(K_de, p=pesos))
        if tipo == 'separavel':
            P, y = _nuvens(rng, K, rng.uniform(1.8, 2.3), rng.uniform(0.12, 0.3))
        elif tipo == 'sobreposto':
            P, y = _nuvens(rng, K, rng.uniform(0.3, 1.2), rng.uniform(0.5, 1.0))
        elif tipo == 'aleatorio':
            n = rng.integers(max(2 * K, 6), 45)
            P = np.column_stack([rng.uniform(X0 + 0.1, X1 - 0.1, n), rng.uniform(Y0 + 0.1, Y1 - 0.1, n)])
            y = rng.integers(1, K + 1, n)
        elif tipo == 'grade':
            n = rng.integers(max(2 * K, 8), 40)
            P = np.column_stack([rng.integers(-1, 4, n), rng.integers(-1, 4, n)]).astype(float) * rng.choice([0.5, 1.0])
            P = P + [0.5, 0.5]
            y = rng.integers(1, K + 1, n)
        else:
            P, y = extremo(j, rng, K)
        P, y = _garante_classes(rng, _clip(P), np.asarray(y, int), K)
        casos.append({'tipo': tipo, 'P': P, 'y': y, 'K': K})
    assert len(casos) == 1000
    return casos


def grade_teste(nx=23, ny=24):
    gx = np.linspace(X0, X1, nx)
    gy = np.linspace(Y0, Y1, ny)
    A, B = np.meshgrid(gx, gy)
    return np.column_stack([A.ravel(), B.ravel()])
