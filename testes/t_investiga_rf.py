"""Investigação pontual da floresta: (1) casos de estrutura divergente no teste exato; (2) casos 'fora' do teste m = 1, refeitos com B = 500."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import t1_arvores as T
from comum import roda_js
from dados import gera

G = T.G


def exata():
    r = np.random.default_rng(61)
    casos = gera(semente=71)
    jobs = [{'pts': T.pontos(c), 'K': c['K'], 'B': 5, 'maxDepth': None if r.random() < 0.7 else int(r.integers(1, 6)), 'maxLeaves': None if r.random() < 0.5 else int(r.integers(2, 12)), 'seed': int(r.integers(1, 100000)), 'crit': 'gini', 'minLeaf': int(r.choice([1, 1, 2, 3, 5])), 'alpha': 0.0, 'grid': G.tolist()} for c in casos]
    sel = [i for i, c in enumerate(casos) if c['tipo'] == 'extremo']
    out, _ = roda_js('floresta', T.JS_RF_EXATA, [jobs[i] for i in sel], 'inv_rf')
    for i, s in zip(sel, out):
        c, jb = casos[i], jobs[i]
        for w, nodes in zip(s['W'], s['trees']):
            w = np.array(w, float)
            achou, prim = T.ajusta_igual(c['P'], c['y'], jb, nodes, sw=w, n_seeds=600, max_features=2, max_leaf_nodes=jb['maxLeaves'])
            if achou is None:
                t = prim.tree_
                print('caso', i, 'K', c['K'], 'n', len(c['y']), {k: jb[k] for k in ('maxDepth', 'maxLeaves', 'minLeaf')})
                print('  P float32 distintos:', len(np.unique(np.asarray(c['P'], np.float32), axis=0)), 'de', len(c['P']), '; pesos', w.astype(int).tolist())
                print('  app:', [(f, round(tt, 6), n_) for f, tt, l, r_, n_ in nodes])
                print('  sk :', [(int(t.feature[j]), round(float(t.threshold[j]), 6), int(t.n_node_samples[j])) for j in range(t.node_count)])
                print('  sk impureza:', np.round(t.impurity, 6).tolist())
                break


def fora():
    r = np.random.default_rng(61)
    casos = gera(semente=71)
    # mesma sequência de sorteios do teste: primeiro os jobs exatos, depois os de m = 1
    _ = [(None if r.random() < 0.7 else int(r.integers(1, 6)), None if r.random() < 0.5 else int(r.integers(2, 12)), int(r.integers(1, 100000)), int(r.choice([1, 1, 2, 3, 5]))) for c in casos]
    jobs2 = [{'pts': T.pontos(c), 'K': c['K'], 'B': 100, 'seed': int(r.integers(1, 100000)), 'minLeaf': int(r.choice([1, 1, 2, 3])), 'grid': G.tolist()} for c in casos]
    out, _ = roda_js('floresta', T.JS_RF_M1, jobs2, 'inv_rf1')
    fora = []
    for i, (c, jb, s) in enumerate(zip(casos, jobs2, out)):
        ps = [RandomForestClassifier(n_estimators=100, max_features=1, min_samples_leaf=jb['minLeaf'], random_state=rs).fit(c['P'], c['y']).predict_proba(G) for rs in (1, 2, 3)]
        a = np.array(s['P'])
        d_js = float(np.abs(a - sum(ps) / 3).mean())
        d_sk = max(float(np.abs(ps[k] - (ps[(k + 1) % 3] + ps[(k + 2) % 3]) / 2).mean()) for k in range(3))
        if d_js > 1.5 * d_sk + 0.01:
            fora.append(i)
    print('fora com B = 100:', len(fora), 'minLeaf:', [jobs2[i]['minLeaf'] for i in fora])
    # refaz com B = 500 e outras sementes
    jobs3 = [dict(jobs2[i], B=500, seed=jobs2[i]['seed'] + 7) for i in fora]
    out3, _ = roda_js('floresta', T.JS_RF_M1, jobs3, 'inv_rf3')
    ainda = 0
    for i, jb, s in zip(fora, jobs3, out3):
        c = casos[i]
        ps = [RandomForestClassifier(n_estimators=500, max_features=1, min_samples_leaf=jb['minLeaf'], random_state=rs).fit(c['P'], c['y']).predict_proba(G) for rs in (11, 12, 13)]
        a = np.array(s['P'])
        d_js = float(np.abs(a - sum(ps) / 3).mean())
        d_sk = max(float(np.abs(ps[k] - (ps[(k + 1) % 3] + ps[(k + 2) % 3]) / 2).mean()) for k in range(3))
        ainda += d_js > 1.5 * d_sk + 0.01
        print(f'  caso {i}: B=500 app–sk {d_js:.4f}, sk–sk {d_sk:.4f}')
    print('ainda fora com B = 500:', ainda, 'de', len(fora))


if __name__ == '__main__':
    exata()
    fora()
