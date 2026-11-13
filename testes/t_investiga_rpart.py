"""Por que o rpart dá outra árvore em alguns casos? Compara nó a nó a árvore do app com a do rpart (mesmos parâmetros, sem poda)."""
import os
import subprocess

import numpy as np
import pandas as pd

import t1_arvores as T
from comum import TMP, roda_js
from dados import gera

Rs = T.Rs


def main():
    r = np.random.default_rng(41)
    casos = gera(semente=51)
    jobs = []
    for c in casos:
        u = r.random()
        depth = None if u < 0.5 else int(r.integers(1, 9))
        leaf = int(r.choice([1, 1, 1, 2, 3, 5]))
        alpha = 0.0
        if r.random() > 0.4:
            alpha = float(r.choice([0.002, 0.01, 0.03, 0.08, 0.15]))
        jobs.append({'pts': T.pontos(c), 'K': c['K'], 'crit': str(r.choice(['gini', 'entropy'])), 'maxDepth': depth, 'minLeaf': leaf, 'alpha': alpha, 'grid': T.G.tolist()})
    sel = [i for i, jb in enumerate(jobs) if jb['alpha'] == 0.0]
    out, _ = roda_js('arvore', T.JS_TREE, [jobs[i] for i in sel], 'inv_rpart')
    # rpart: guarda a estrutura (nó, variável, limiar, n) de cada caso
    linhas = [(ii, a, b, int(t)) for ii, i in enumerate(sel) for (a, b), t in zip(casos[i]['P'], casos[i]['y'])]
    ent, sai = TMP / 'inv_rpart_ent.csv', TMP / 'inv_rpart_sai.csv'
    pd.DataFrame(linhas, columns=['caso', 'x1', 'x2', 'y']).to_csv(ent, index=False)
    ext = TMP / 'inv_rpart_ext.csv'
    pd.DataFrame({'caso': range(len(sel)), 'L': [jobs[i]['minLeaf'] for i in sel], 'prof': [jobs[i]['maxDepth'] or 30 for i in sel], 'crit': ['information' if jobs[i]['crit'] == 'entropy' else 'gini' for i in sel]}).to_csv(ext, index=False)
    scr = TMP / 'inv_rpart.R'
    scr.write_text(r'''
args <- commandArgs(TRUE); ent <- read.csv(args[1]); ext <- read.csv(args[2])
suppressPackageStartupMessages(library(rpart))
out <- file(args[3], "w"); writeLines("caso,no,var,thr,n,esq", out)
for (cs in unique(ent$caso)) {
  d <- ent[ent$caso == cs, c("x1", "x2", "y")]; d$y <- factor(d$y); e <- ext[ext$caso == cs, ]
  m <- rpart(y ~ x1 + x2, d, method = "class", parms = list(split = as.character(e$crit)), control = rpart.control(minsplit = 2 * e$L, minbucket = e$L, cp = -1, maxdepth = e$prof, maxcompete = 0, maxsurrogate = 0, xval = 0))
  f <- m$frame; nos <- as.integer(rownames(f)); sp <- m$splits; k <- 0
  for (j in seq_len(nrow(f))) {
    if (f$var[j] == "<leaf>") { writeLines(sprintf("%d,%d,leaf,0,%d,0", cs, nos[j], f$n[j]), out) }
    else { k <- k + 1; while (sp[k, "count"] != f$n[j] || rownames(sp)[k] != as.character(f$var[j])) k <- k + 1
           writeLines(sprintf("%d,%d,%s,%.17g,%d,%d", cs, nos[j], as.character(f$var[j]), sp[k, "index"], f$n[j], as.integer(sp[k, "ncat"] < 0)), out) }
  }
}
close(out)
''', encoding='utf-8')
    res = subprocess.run([Rs, '--vanilla', str(scr), str(ent), str(ext), str(sai)], capture_output=True)
    if res.returncode:
        print(res.stderr.decode('latin-1')[-800:])
        return
    rp = dict(tuple(pd.read_csv(sai).groupby('caso')))
    causas = {}
    for ii, i in enumerate(sel):
        s, c, jb = out[ii], casos[i], jobs[i]
        R = rp[ii].set_index('no')
        N = s['nodes']
        P, y = c['P'], c['y']
        x32 = np.asarray(P, np.float32).astype(float)

        def imp(lbl):
            if len(lbl) == 0:
                return 0.0
            q = np.bincount(lbl, minlength=c['K'] + 1)[1:] / len(lbl)
            return float(1 - (q ** 2).sum()) if jb['crit'] == 'gini' else float(-sum(v * np.log2(v) for v in q if v > 0))

        def ganho(idx, f, t):
            L, Rr = idx[x32[idx, f] <= t], idx[x32[idx, f] > t]
            return len(idx) * imp(y[idx]) - len(L) * imp(y[L]) - len(Rr) * imp(y[Rr])

        # percorre: nó do app (id) e nó do rpart (número 1, 2, 3 ... com filhos 2k, 2k+1)
        def anda(a, rno, idx):
            f, t, l, r_, n_ = N[a]
            if rno not in R.index:
                return ('rpart sem o nó', n_)
            rr = R.loc[rno]
            if (l < 0) != (rr['var'] == 'leaf'):
                if l < 0:
                    return ('rpart divide, app não', n_, len(idx))
                return ('app divide, rpart não', round(ganho(idx, f, t), 12), n_, jb['minLeaf'], jb['maxDepth'])
            if l < 0:
                return None
            fr = 0 if rr['var'] == 'x1' else 1
            tr = float(rr['thr'])
            if fr != f or abs(tr - t) > 1e-6:
                ga, gr = ganho(idx, f, t), ganho(idx, fr, tr)
                return ('divisão diferente', 'empate' if abs(ga - gr) < 1e-9 else f'ganho app {ga:.6g} rpart {gr:.6g}', n_)
            # filhos: no rpart, o da esquerda pode ser o lado ">="
            esq_menor = bool(rr['esq'])
            L, Rr = idx[x32[idx, f] <= t], idx[x32[idx, f] > t]
            a1 = anda(l, 2 * rno if esq_menor else 2 * rno + 1, L)
            return a1 if a1 else anda(r_, 2 * rno + 1 if esq_menor else 2 * rno, Rr)

        d = anda(0, 1, np.arange(len(y)))
        if d:
            causas.setdefault(d[0] + (' (' + d[1] + ')' if d[0] == 'divisão diferente' and d[1] == 'empate' else ''), []).append((i, c['tipo'], d))
    for k, v in causas.items():
        print(k, len(v))
        for e in v[:4]:
            print('   ', e)


if __name__ == '__main__':
    main()
