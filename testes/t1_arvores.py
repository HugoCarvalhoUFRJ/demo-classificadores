"""KNN, árvore e floresta: 1000 conjuntos por app contra scikit-learn e R."""
import subprocess
import sys
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from comum import TMP, relatorio, roda_js
from dados import gera, grade_teste

warnings.filterwarnings('ignore')
Rs = r'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
G = grade_teste()
AQUI = os.path.dirname(os.path.abspath(__file__))


def pontos(c):
    return [[float(a), float(b), int(t)] for (a, b), t in zip(c['P'], c['y'])]


def r_rodar(metodo, casos, extra):
    linhas = [(i, a, b, int(t)) for i, c in enumerate(casos) for (a, b), t in zip(c['P'], c['y'])]
    ent, gr, sai, ext = (TMP / f'r_{metodo}_{x}.csv' for x in ('ent', 'grade', 'sai', 'extra'))
    pd.DataFrame(linhas, columns=['caso', 'x1', 'x2', 'y']).to_csv(ent, index=False)
    pd.DataFrame(G, columns=['x1', 'x2']).to_csv(gr, index=False)
    pd.DataFrame(extra).to_csv(ext, index=False)
    r = subprocess.run([Rs, '--vanilla', os.path.join(AQUI, 'ref_r.R'), metodo, str(ent), str(gr), str(sai), str(ext)], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode('latin-1')[-1500:])
    return dict(tuple(pd.read_csv(sai, dtype={'erro': str}).groupby('caso')))


# ---------------- KNN ----------------
JS_KNN = """
const pts = caso.pts.map(p => ({x: p[0], y: p[1], c: p[2]}));
const fast = Solver.makeGrid(pts, caso.k, caso.K, caso.p, caso.w);
const P = caso.grid.map(g => Solver.posterior(pts, g[0], g[1], caso.k, caso.K, caso.p, caso.w));
const lab = caso.grid.map(g => Solver.predict(pts, g[0], g[1], caso.k, caso.K, caso.p, caso.w));
const labF = caso.grid.map(g => fast(g[0], g[1]));
return {P, lab, labF};
"""


def dist_p(G, P, p):
    """Distâncias de Minkowski exatas (sem a expansão |x|² − 2x·y + |y|² do scikit-learn)."""
    return (np.abs(G[:, None, :] - P[None, :, :]) ** p).sum(2) ** (1 / p)


def t_knn(linhas):
    r = np.random.default_rng(21)
    casos = gera(semente=31)
    jobs = []
    for c in casos:
        n = len(c['y'])
        u = r.random()
        k = 1 if u < 0.2 else n if u < 0.3 else int(r.integers(1, n + 1))
        jobs.append({'pts': pontos(c), 'K': c['K'], 'k': k, 'p': int(r.choice([1, 2, 3])), 'w': str(r.choice(['uniform', 'distance'])), 'grid': G.tolist()})
    out, ms = roda_js('knn', JS_KNN, jobs, 't1_knn')
    n = dict(total=len(jobs), erroJS=0, comparados=0, div_prob=0, div_classe=0, div_rapido=0, nos_tie_dist=0, nos_tie_voto=0, ref_falhou=0)
    combos = {}
    maior = 0.0
    ex = []
    extra = {'caso': [], 'K': [], 'p': [], 'kernel': []}
    livres = []
    for i, (c, jb, s) in enumerate(zip(casos, jobs, out)):
        extra['caso'].append(i); extra['K'].append(jb['k']); extra['p'].append(jb['p']); extra['kernel'].append('inv' if jb['w'] == 'distance' else 'rectangular')
        if 'erro' in s:
            n['erroJS'] += 1
            ex.append(s['erro'][:200])
            livres.append(None)
            continue
        P, y, k, p = c['P'], c['y'], jb['k'], jb['p']
        D = np.sort(dist_p(G, P, p), axis=1)
        # nós com empate (relativo 1e-9) na distância do K-ésimo vizinho: a ordem depende da implementação
        tie = np.zeros(len(G), bool)
        if k < len(y):
            tie = (D[:, k] - D[:, k - 1]) <= 1e-9 * np.maximum(D[:, k], 1e-300)
        n['nos_tie_dist'] += int(tie.sum())
        try:
            m = KNeighborsClassifier(n_neighbors=k, weights=jb['w'], algorithm='brute', p=p, metric='minkowski').fit(P, y)
            Pr = m.predict_proba(G)
            Lr = m.predict(G) - 1
        except Exception:
            n['ref_falhou'] += 1
            livres.append(None)
            continue
        n['comparados'] += 1
        pa = np.array(s['P'])
        top = np.sort(pa, axis=1)
        voto = (top[:, -1] - top[:, -2]) < 1e-9
        n['nos_tie_voto'] += int((voto & ~tie).sum())
        livre = ~tie & ~voto
        livres.append(livre)
        d = float(np.abs(pa[~tie] - Pr[~tie]).max()) if (~tie).any() else 0.0
        maior = max(maior, d)
        dif = int(((np.array(s['lab']) != Lr) & livre).sum())
        chave = f"{jb['w']},p={p}"
        cb = combos.setdefault(chave, [0, 0, 0.0])
        cb[0] += 1
        cb[2] = max(cb[2], d)
        # pesos 1/d: a expansão do scikit-learn erra a distância em ~1e-8 (relativo), daí a tolerância
        if d > (1e-12 if jb['w'] == 'uniform' else 1e-6):
            n['div_prob'] += 1
            ex.append(('prob', c['tipo'], k, chave, d))
        if dif:
            n['div_classe'] += 1
            cb[1] += 1
            ex.append(('classe', c['tipo'], k, chave, dif))
        # grade rápida contra a regra exata, inclusive nos empates de distância (os dois desempatam pelo índice)
        if int(((np.array(s['lab']) != np.array(s['labF'])) & ~voto).sum()):
            n['div_rapido'] += 1
    linhas += [f'knn sklearn: {n}', f'  maior |Δp| = {maior:.3e}; JS {ms:.0f} ms ({ms/len(jobs):.2f} ms/caso)',
               '  por combinação [casos, casos com classe divergente, maior |Δp|]: ' + ', '.join(f'{a}: [{b[0]}, {b[1]}, {b[2]:.1e}]' for a, b in sorted(combos.items()))] + [f'  ex: {e}' for e in ex[:8]]
    # R class::knn (pesos iguais, p = 2): empates são sorteados; compara só nós sem empate de distância nem de voto
    so_classe = [i for i, jb in enumerate(jobs) if jb['w'] == 'uniform' and jb['p'] == 2]
    gr = r_rodar('knn', [casos[i] for i in so_classe], {'caso': list(range(len(so_classe))), 'K': [jobs[i]['k'] for i in so_classe]})
    nr = dict(casos=len(so_classe), comparados=0, r_falhou=0, nos_divergentes=0, casos_div=0)
    for ii, i in enumerate(so_classe):
        c, s = casos[i], out[i]
        if 'erro' in s or livres[i] is None:
            continue
        d = gr.get(ii)
        if d is None or d['erro'].notna().any():
            nr['r_falhou'] += 1
            continue
        pred = d.sort_values('linha')['pred'].to_numpy() - 1
        livre = livres[i].copy()
        # class::knn considera empatadas distâncias a menos de 1e-4 (relativo) da K-ésima e inclui todas
        k_ = jobs[i]['k']
        if k_ < len(c['y']):
            dd = np.sort(dist_p(G, c['P'], 2), axis=1)
            livre &= dd[:, k_] > dd[:, k_ - 1] * (1 + 1e-4)
        dif = int(((pred != np.array(s['lab'])) & livre).sum())
        nr['comparados'] += 1
        if dif:
            nr['casos_div'] += 1
            nr['nos_divergentes'] += dif
    linhas.append(f'knn R class::knn (pesos iguais, p = 2; sem nós empatados): {nr}')
    # R kknn: todas as combinações
    gk = r_rodar('kknn', casos, extra)
    nk = dict(comparados=0, r_falhou=0, casos_div=0, nos_divergentes=0, atributo_constante=0, nos_quase_empate=0)
    exk = []
    for i, (c, jb, s) in enumerate(zip(casos, jobs, out)):
        if 'erro' in s or livres[i] is None:
            continue
        d = gk.get(i)
        if d is None or d['erro'].notna().any():
            nk['r_falhou'] += 1
            if len(exk) < 4:
                exk.append(('falhou', c['tipo'], jb['k'], len(c['y']), str(d['erro'].iloc[0])[:100] if d is not None else ''))
            continue
        P, y, k, p = c['P'], c['y'], jb['k'], jb['p']
        # o kknn ignora atributos constantes no treino (peso zero na distância)
        if (np.ptp(P, axis=0) == 0).any():
            nk['atributo_constante'] += 1
            continue
        pred = d.sort_values('linha')['pred'].to_numpy() - 1
        # com k = n não existe o (K+1)-ésimo vizinho que o kknn usa: relatado à parte
        if k == len(y):
            kn = nk.setdefault('k_igual_n', {}).setdefault(f"{jb['w']},p={p}", [0, 0])
            kn[0] += 1
            kn[1] += int(((pred != np.array(s['lab'])) & livres[i]).sum() > 0)
            continue
        livre = livres[i].copy()
        Ds = np.sort(dist_p(G, P, p), axis=1)
        pa = np.array(s['P'])
        top = np.sort(pa, axis=1)
        if jb['w'] == 'distance':
            # o kknn limita d/d₍K+1₎ a [1e-6, 1 − 1e-6]: perto de empates de peso, a classe pode mudar
            quase = (top[:, -1] - top[:, -2]) < 1e-5
            if k < len(y):
                quase |= Ds[:, k - 1] >= Ds[:, k] * (1 - 1e-6)
            nk['nos_quase_empate'] += int((quase & livre).sum())
            livre &= ~quase
        dif = int(((pred != np.array(s['lab'])) & livre).sum())
        nk['comparados'] += 1
        if dif:
            nk['casos_div'] += 1
            nk['nos_divergentes'] += dif
            if len(exk) < 8:
                exk.append((c['tipo'], k, len(y), jb['w'], p, dif))
    linhas.append(f'knn R kknn (todas as combinações; sem nós empatados): {nk}')
    linhas += [f'  ex kknn: {e}' for e in exk]


# ---------------- árvore ----------------
JS_TREE = """
const pts = caso.pts.map(p => ({x: p[0], y: p[1], c: p[2]}));
const m = Solver.fit(pts, caso.K, {crit: caso.crit, maxDepth: caso.maxDepth, minLeaf: caso.minLeaf, alpha: caso.alpha});
const lab = caso.grid.map(g => Solver.predictTree(m.tree, g[0], g[1]));
const R = Solver.regions([m.tree], caso.K);
const labR = caso.grid.map(g => R.cells[Solver.cellIndex(R.ys, g[1])*R.nx + Solver.cellIndex(R.xs, g[0])]);
return {nodes: m.tree.nodes.map(n => [n.f, n.t, n.l, n.r, n.n]), lab, labR};
"""


def mesma_arvore(js_nodes, tree):
    """Compara nó a nó, percorrendo as duas árvores a partir da raiz (a numeração interna pode diferir)."""
    if tree.node_count != len(js_nodes):
        return False

    def igual(i, j):
        f, t, l, r_, n_ = js_nodes[i]
        if int(tree.n_node_samples[j]) != n_:
            return False
        cl = int(tree.children_left[j])
        if (cl < 0) != (l < 0):
            return False
        if cl < 0:
            return True
        if int(tree.feature[j]) != f or abs(float(tree.threshold[j]) - t) > 1e-9:
            return False
        return igual(l, cl) and igual(r_, int(tree.children_right[j]))

    return igual(0, 0)


def ajusta_igual(P, y, jb, js_nodes, sw=None, n_seeds=20, **kw):
    achou = None
    primeira = None
    for rs in range(n_seeds):
        m = DecisionTreeClassifier(criterion=jb['crit'], max_depth=jb['maxDepth'], min_samples_split=2, min_samples_leaf=jb['minLeaf'], ccp_alpha=jb['alpha'], random_state=rs, **kw).fit(P, y, sample_weight=sw)
        if primeira is None:
            primeira = m
        if mesma_arvore(js_nodes, m.tree_):
            achou = (rs, m)
            break
    return achou, primeira


def proxy(P, y, K, crit, f, t):
    """Melhoria (forma do scikit-learn) de uma divisão, recalculada do zero em float32."""
    x = np.asarray(P[:, f], np.float32).astype(float)
    def imp(lbl):
        n_ = len(lbl)
        c = np.bincount(lbl, minlength=K + 1)[1:] / n_
        return float(1 - (c ** 2).sum()) if crit == 'gini' else float(-sum(v * np.log2(v) for v in c if v > 0))
    L, R = y[x <= t], y[x > t]
    return -len(R) * imp(R) - len(L) * imp(L)


def so_empates(js_nodes, tree, P, y, K, crit):
    """Percorre as duas árvores; onde a divisão difere, confirma que as duas têm a mesma melhoria (empate genuíno)."""
    def v(i, j, idx):
        f, t, l, r_, n_ = js_nodes[i]
        cl = int(tree.children_left[j])
        if (l < 0) != (cl < 0) or int(tree.n_node_samples[j]) != n_:
            return False
        if l < 0:
            return True
        fs, ts = int(tree.feature[j]), float(tree.threshold[j])
        if fs != f or abs(ts - t) > 1e-9:
            return abs(proxy(P[idx], y[idx], K, crit, f, t) - proxy(P[idx], y[idx], K, crit, fs, ts)) < 1e-9
        x = np.asarray(P[idx, f], np.float32).astype(float)
        return v(l, cl, idx[x <= t]) and v(r_, int(tree.children_right[j]), idx[x > t])
    return v(0, 0, np.arange(len(y)))


def t_arvore(linhas):
    r = np.random.default_rng(41)
    casos = gera(semente=51)
    jobs = []
    for c in casos:
        u = r.random()
        depth = None if u < 0.5 else int(r.integers(1, 9))
        leaf = int(r.choice([1, 1, 1, 2, 3, 5]))
        alpha = 0.0
        v = r.random()
        if v > 0.4:
            alpha = float(r.choice([0.002, 0.01, 0.03, 0.08, 0.15]))
        jobs.append({'pts': pontos(c), 'K': c['K'], 'crit': str(r.choice(['gini', 'entropy'])), 'maxDepth': depth, 'minLeaf': leaf, 'alpha': alpha, 'grid': G.tolist()})
    # Parte dos casos usa exatamente um α efetivo da sequência de poda do scikit-learn
    for c, jb in zip(casos, jobs):
        if r.random() < 0.2:
            path = DecisionTreeClassifier(criterion=jb['crit'], max_depth=jb['maxDepth'], min_samples_leaf=jb['minLeaf'], random_state=0).cost_complexity_pruning_path(c['P'], c['y'])
            if len(path.ccp_alphas) > 1:
                # um pouco acima do α efetivo: exatamente em cima, o desfecho depende do arredondamento
                jb['alpha'] = float(path.ccp_alphas[r.integers(len(path.ccp_alphas))]) * (1 + 1e-8)
    out, ms = roda_js('arvore', JS_TREE, jobs, 't1_arvore')
    n = dict(total=len(jobs), erroJS=0, comparados=0, igual_semente0=0, igual_outra_semente=0, div_estrutura=0, div_pred=0, div_regioes=0)
    ex = []
    extra = {'caso': [], 'L': [], 'prof': [], 'crit': []}
    for i, (c, jb, s) in enumerate(zip(casos, jobs, out)):
        extra['caso'].append(i); extra['L'].append(jb['minLeaf']); extra['prof'].append(jb['maxDepth'] or 30); extra['crit'].append('information' if jb['crit'] == 'entropy' else 'gini')
        if 'erro' in s:
            n['erroJS'] += 1
            ex.append(s['erro'][:300])
            continue
        n['comparados'] += 1
        (achou, primeira) = ajusta_igual(c['P'], c['y'], jb, s['nodes'])
        if achou is None:
            # muitos empates: pode faltar semente que escolha o mesmo atributo em todos
            achou, primeira = ajusta_igual(c['P'], c['y'], jb, s['nodes'], n_seeds=1500)
        if achou is None and so_empates(s['nodes'], primeira.tree_, c['P'], c['y'], c['K'], jb['crit']):
            n['empate_confirmado'] = n.get('empate_confirmado', 0) + 1
            continue
        if achou is None:
            n['div_estrutura'] += 1
            if len(ex) < 12:
                ex.append(('estrutura', c['tipo'], c['K'], len(c['y']), {k: jb[k] for k in ('crit', 'maxDepth', 'minLeaf', 'alpha')}, len(s['nodes']), primeira.tree_.node_count))
            continue
        rs, m = achou
        n['igual_semente0' if rs == 0 else 'igual_outra_semente'] += 1
        Lr = m.predict(G) - 1
        if int((np.array(s['lab']) != Lr).sum()):
            n['div_pred'] += 1
        if int((np.array(s['labR']) != Lr).sum()):
            n['div_regioes'] += 1
    linhas += [f'árvore sklearn: {n}', f'  JS {ms:.0f} ms ({ms/len(jobs):.2f} ms/caso)'] + [f'  ex: {e}' for e in ex[:8]]
    # rpart: cp = -1 e mesmos mínimos; só casos sem poda. Regras diferem em empates e em divisões sem ganho, então é só um relatório.
    sem_poda = [i for i, jb in enumerate(jobs) if jb['alpha'] == 0.0]
    cs = [casos[i] for i in sem_poda]
    ex2 = {k: [extra[k][i] for i in sem_poda] for k in extra}
    ex2['caso'] = list(range(len(cs)))
    gr = r_rodar('rpart', cs, ex2)
    nr = dict(sem_poda=len(cs), r_falhou=0, comparados=0, identicos=0, casos_div=0, nos_divergentes=0, nos_total=0)
    for ii, i in enumerate(sem_poda):
        s = out[i]
        if 'erro' in s:
            continue
        d = gr.get(ii)
        if d is None or d['erro'].notna().any():
            nr['r_falhou'] += 1
            continue
        pred = d.sort_values('linha')['pred'].to_numpy() - 1
        dif = int((pred != np.array(s['lab'])).sum())
        nr['comparados'] += 1
        nr['nos_total'] += len(G)
        if dif == 0:
            nr['identicos'] += 1
        else:
            nr['casos_div'] += 1
            nr['nos_divergentes'] += dif
            nr.setdefault('tipos', {})
            nr['tipos'][casos[i]['tipo']] = nr['tipos'].get(casos[i]['tipo'], 0) + 1
    linhas.append(f'árvore R rpart: {nr}')
    linhas.append('  causas das árvores diferentes: veja t_investiga_rpart.py (ganho zero, que o rpart não divide, e empates de ganho desfeitos pelo arredondamento)')


# ---------------- floresta ----------------
JS_RF_EXATA = """
const pts = caso.pts.map(p => ({x: p[0], y: p[1], c: p[2]}));
const f = Solver.fit(pts, caso.K, {B: caso.B, m: 2, maxDepth: caso.maxDepth, maxLeaves: caso.maxLeaves, minLeaf: caso.minLeaf, seed: caso.seed});
const W = [];
for (let t = 0; t < caso.B; t++) W.push(Solver.bootstrapWeights(pts.length, caso.seed, t).w);
const R = Solver.regions(f.trees, caso.K);
const lab = caso.grid.map(g => Solver.predict(f, g[0], g[1]));
const labR = caso.grid.map(g => R.cells[Solver.cellIndex(R.ys, g[1])*R.nx + Solver.cellIndex(R.xs, g[0])]);
return {W, trees: f.trees.map(t => t.nodes.map(n => [n.f, n.t, n.l, n.r, n.n])), P: caso.grid.map(g => Solver.probs(f, g[0], g[1])), lab, labR, oob: f.oob, oobV: f.oobVotes, oobN: Array.from(f.oobN)};
"""

JS_RF_M1 = """
const pts = caso.pts.map(p => ({x: p[0], y: p[1], c: p[2]}));
const f = Solver.fit(pts, caso.K, {B: caso.B, m: 1, maxDepth: null, minLeaf: caso.minLeaf, seed: caso.seed});
return {P: caso.grid.map(g => Solver.probs(f, g[0], g[1])), lab: caso.grid.map(g => Solver.predict(f, g[0], g[1]))};
"""


def so_empates_w(js_nodes, tree, P, y, w, K):
    """Como so_empates, com pesos de bootstrap (Gini ponderado)."""
    def prox(idx, f, t):
        x = np.asarray(P[idx, f], np.float32).astype(float)
        def imp(m):
            ww = w[idx][m]; s_ = ww.sum()
            c = np.array([ww[y[idx][m] == k].sum() for k in range(1, K + 1)]) / s_
            return s_ * (1 - (c ** 2).sum())
        return -imp(x > t) - imp(x <= t)
    def v(i, j, idx):
        f, t, l, r_, n_ = js_nodes[i]
        cl = int(tree.children_left[j])
        if (l < 0) != (cl < 0):
            return abs(0) < 0  # folha num e não no outro: só aceitável se a diferença vier de empate acima
        if l < 0:
            return True
        fs, ts = int(tree.feature[j]), float(tree.threshold[j])
        if fs != f or abs(ts - t) > 1e-9:
            return abs(prox(idx, f, t) - prox(idx, fs, ts)) < 1e-9
        x = np.asarray(P[idx, f], np.float32).astype(float)
        return v(l, cl, idx[x <= t]) and v(r_, int(tree.children_right[j]), idx[x > t])
    return v(0, 0, np.where(w > 0)[0])


def t_floresta(linhas):
    r = np.random.default_rng(61)
    casos = gera(semente=71)
    jobs = [{'pts': pontos(c), 'K': c['K'], 'B': 5, 'maxDepth': None if r.random() < 0.7 else int(r.integers(1, 6)), 'maxLeaves': None if r.random() < 0.5 else int(r.integers(2, 12)), 'seed': int(r.integers(1, 100000)), 'crit': 'gini', 'minLeaf': int(r.choice([1, 1, 2, 3, 5])), 'alpha': 0.0, 'grid': G.tolist()} for c in casos]
    out, ms = roda_js('floresta', JS_RF_EXATA, jobs, 't1_rf')
    n = dict(total=len(jobs), erroJS=0, comparados=0, arvores=0, igual_semente0=0, igual_outra=0, div_estrutura=0, div_prob=0, div_regioes=0, div_oob=0)
    ex = []
    maior = 0.0
    for c, jb, s in zip(casos, jobs, out):
        if 'erro' in s:
            n['erroJS'] += 1
            ex.append(s['erro'][:300])
            continue
        n['comparados'] += 1
        P, y = c['P'], c['y']
        probs = np.zeros((len(G), jb['K']))
        ok = True
        oobv = np.zeros((len(y), jb['K']))
        oobn = np.zeros(len(y), int)
        for w, nodes in zip(s['W'], s['trees']):
            n['arvores'] += 1
            w = np.array(w, float)
            achou, prim = ajusta_igual(P, y, jb, nodes, sw=w, max_features=2, max_leaf_nodes=jb['maxLeaves'])
            if achou is None:
                achou, prim = ajusta_igual(P, y, jb, nodes, sw=w, n_seeds=600, max_features=2, max_leaf_nodes=jb['maxLeaves'])
            if achou is None:
                ok = False
                if so_empates_w(nodes, prim.tree_, P, y, w, jb['K']):
                    n['empate_confirmado'] = n.get('empate_confirmado', 0) + 1
                    break
                n['div_estrutura'] += 1
                if len(ex) < 8:
                    ex.append(('estrutura', c['tipo'], jb['K'], len(y)))
                break
            rs, m = achou
            n['igual_semente0' if rs == 0 else 'igual_outra'] += 1
            # classes ausentes na amostra de bootstrap: alinha as colunas às K classes
            pp = np.zeros((len(G), jb['K']))
            pp[:, m.classes_.astype(int) - 1] = m.predict_proba(G)
            probs += pp
            mask = w == 0
            if mask.any():
                q = np.zeros((int(mask.sum()), jb['K']))
                q[:, m.classes_.astype(int) - 1] = m.predict_proba(P[mask])
                oobv[mask] += q
                oobn[mask] += 1
        if not ok:
            continue
        probs /= jb['B']
        d = float(np.abs(probs - np.array(s['P'])).max())
        maior = max(maior, d)
        if d > 1e-9:
            n['div_prob'] += 1
        Lr = probs.argmax(1)
        top = np.sort(probs, axis=1)
        tie = (top[:, -1] - top[:, -2]) < 1e-9
        if int(((np.array(s['labR']) != Lr) & ~tie).sum()):
            n['div_regioes'] += 1
        if np.abs(oobv - np.array(s['oobV'])).max() > 1e-9 or (oobn != np.array(s['oobN'])).any():
            n['div_oob'] += 1
    print('etapa m=2 concluída', flush=True)
    linhas += [f'floresta m=2 exata (mesmo bootstrap, 5 árvores): {n}', f'  maior |Δp| = {maior:.3e}; JS {ms:.0f} ms ({ms/len(jobs):.2f} ms/caso)'] + [f'  ex: {e}' for e in ex[:6]]

    # m = 1: o sorteio é outro; compara a distância JS–sklearn com a distância sklearn–sklearn (outra semente)
    B = 100
    jobs2 = [{'pts': pontos(c), 'K': c['K'], 'B': B, 'seed': int(r.integers(1, 100000)), 'minLeaf': int(r.choice([1, 1, 2, 3])), 'grid': G.tolist()} for c in casos]
    out2, ms2 = roda_js('floresta', JS_RF_M1, jobs2, 't1_rf1')
    soma_js, soma_sk = [], []
    n2 = dict(total=len(jobs2), erroJS=0, comparados=0, casos_fora=0, pior_razao=0.0)
    ex2 = []
    extra = {'caso': [], 'B': [], 'm': [], 'nodesize': []}
    for i, (c, jb, s) in enumerate(zip(casos, jobs2, out2)):
        extra['caso'].append(i); extra['B'].append(B); extra['m'].append(1); extra['nodesize'].append(2 * jb['minLeaf'] - 1)
        if 'erro' in s:
            n2['erroJS'] += 1
            continue
        P, y = c['P'], c['y']
        ps = [RandomForestClassifier(n_estimators=B, max_features=1, min_samples_leaf=jb['minLeaf'], bootstrap=True, random_state=rs).fit(P, y).predict_proba(G) for rs in (1, 2, 3)]
        a = np.array(s['P'])
        # distância do app à média de três florestas do sklearn, contra a de cada floresta do sklearn à média das outras duas
        d_js = float(np.abs(a - (ps[0] + ps[1] + ps[2]) / 3).mean())
        d_sk = max(float(np.abs(ps[i] - (ps[(i + 1) % 3] + ps[(i + 2) % 3]) / 2).mean()) for i in range(3))
        soma_js.append(d_js)
        soma_sk.append(d_sk)
        n2['comparados'] += 1
        # folga: ruído de Monte Carlo com B = 100
        if d_js > 1.5 * d_sk + 0.01:
            n2['casos_fora'] += 1
            ex2.append((c['tipo'], c['K'], len(y), round(d_js, 4), round(d_sk, 4)))
        if d_sk > 0:
            n2['pior_razao'] = max(n2['pior_razao'], d_js / d_sk if d_sk > 1e-3 else 0)
    linhas += [f'floresta m=1 (B=100): {n2}', f'  média |Δp| app–(média de 3 sklearn) = {np.mean(soma_js):.4f}; pior sklearn–(média das outras 2) = {np.mean(soma_sk):.4f}; JS {ms2:.0f} ms ({ms2/len(jobs2):.1f} ms/caso)'] + [f'  ex: {e}' for e in ex2[:8]]
    # R randomForest com voto majoritário: concordância de classes com o app (B=300)
    gr = r_rodar('rf', casos, extra)
    conc, concSk = [], []
    nf = dict(r_falhou=0, comparados=0)
    for i, (c, s) in enumerate(zip(casos, out2)):
        if 'erro' in s:
            continue
        d = gr.get(i)
        if d is None or d['erro'].notna().any():
            nf['r_falhou'] += 1
            continue
        nf['comparados'] += 1
        pred = d.sort_values('linha')['pred'].to_numpy() - 1
        conc.append((pred == np.array(s['lab'])).mean())
        rf1 = RandomForestClassifier(n_estimators=B, max_features=1, min_samples_leaf=jobs2[i]['minLeaf'], random_state=1).fit(c['P'], c['y'])
        concSk.append((pred == (rf1.predict(G) - 1)).mean())
    linhas.append(f'floresta R randomForest (B=100): {nf}; concordância média de classes app–R {np.mean(conc):.4f}; sklearn–R {np.mean(concSk):.4f}')


def main(qual='todos'):
    linhas = []
    if qual in ('todos', 'knn'):
        t_knn(linhas)
    if qual in ('todos', 'arvore'):
        t_arvore(linhas)
    if qual in ('todos', 'floresta'):
        t_floresta(linhas)
    relatorio('t1_' + qual, linhas)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'todos')
