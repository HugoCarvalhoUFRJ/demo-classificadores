"""GNB, LDA e QDA: 1000 conjuntos por app contra scikit-learn (probabilidades e classes) e R (classes)."""
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB

from comum import RES, TMP, relatorio, roda_js
import os
from dados import gera, grade_teste

warnings.filterwarnings('ignore')
Rs = r'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
G = grade_teste()

JS = """
const pts = caso.pts.map(p => ({x: p[0], y: p[1], c: p[2]}));
const m = Solver.fit(pts, caso.K, {priors: caso.priors, ddof: caso.ddof || 0});
if (!m.ok) return {ok: false};
const P = caso.grid.map(g => Solver.posterior(m, g[0], g[1]));
const lab = caso.grid.map(g => Solver.argmax(m, g[0], g[1]));
return {ok: true, P, lab, singular: m.singular ? true : false, pred: m.pred};
"""


def compara(nome, casos, jobs, modelo_ref, cond_excluir=None):
    out, ms = roda_js(nome, JS, jobs, 't1_' + nome)
    n = dict(total=len(jobs), erroJS=0, app_sem_solucao=0, ref_falhou=0, comparados=0, div_prob=0, div_classe=0, empates=0)
    maior = 0.0
    ex = []
    resumo_tipos = {}
    for c, jb, s in zip(casos, jobs, out):
        if 'erro' in s:
            n['erroJS'] += 1; ex.append(s['erro'][:200]); continue
        P = c['P']; y = c['y']
        try:
            ref = modelo_ref(jb)
            ref.fit(P, y)
            Pr = ref.predict_proba(G)
            Lr = ref.predict(G) - 1
            ok_ref = np.isfinite(Pr).all()
        except Exception as e:
            ok_ref = False
            Pr = None
        if not s['ok']:
            n['app_sem_solucao'] += 1
            if ok_ref and cond_excluir and not cond_excluir(c, ref):
                # sklearn ajustou algo; só é divergência se a matriz não for de fato singular
                ex.append(('app_sem_solucao_mas_ref_ok', c['tipo'], c['K'], len(y)))
            continue
        if not ok_ref:
            n['ref_falhou'] += 1
            continue
        n['comparados'] += 1
        pa = np.array(s['P'])
        d = float(np.abs(pa - Pr).max())
        maior = max(maior, d) if not s.get('singular') else maior
        la = np.array(s['lab'])
        top2 = np.sort(Pr, axis=1)
        emp = (top2[:, -1] - top2[:, -2]) < 1e-7
        n['empates'] += int(emp.sum())
        dif = int(((la != Lr) & ~emp).sum())
        if s.get('singular') and (d > 1e-6 or dif):
            # Σ̂ singular: a solução do scikit-learn depende de ruído de arredondamento; o app avisa na tela
            n['div_em_singular'] = n.get('div_em_singular', 0) + 1
            ex.append(('singular', c['tipo'], c['K'], len(y), d))
            continue
        if d > 1e-6:
            n['div_prob'] += 1; ex.append(('prob', c['tipo'], c['K'], len(y), jb['priors'], d))
        if dif:
            n['div_classe'] += 1; ex.append(('classe', c['tipo'], c['K'], len(y), dif))
    return n, maior, ex, ms


def r_rodar(metodo, casos, chave=None):
    pasta = TMP
    linhas = []
    for i, c in enumerate(casos):
        for (a, b), t in zip(c['P'], c['y']):
            linhas.append((i, a, b, int(t)))
    ent = pasta / f'r_{metodo}_ent.csv'
    pd.DataFrame(linhas, columns=['caso', 'x1', 'x2', 'y']).to_csv(ent, index=False)
    gr = pasta / f'r_grade.csv'
    pd.DataFrame(G, columns=['x1', 'x2']).to_csv(gr, index=False)
    sai = pasta / f'r_{metodo}_sai.csv'
    r = subprocess.run([Rs, '--vanilla', str(RES.parent.parent / 'nada')] if False else [Rs, '--vanilla', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_r.R'), metodo, str(ent), str(gr), str(sai)], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    df = pd.read_csv(sai, dtype={'erro': str})
    return df


def compara_r(metodo, casos, jobs, nome):
    df = r_rodar(metodo, casos)
    out, _ = roda_js(nome, JS, jobs, 't1r_' + nome)
    n = dict(total=len(casos), r_falhou=0, app_sem=0, comparados=0, div_casos=0, nos_divergentes=0, empates_sem=0)
    ex = []
    grupos = dict(tuple(df.groupby('caso')))
    for i, (c, s) in enumerate(zip(casos, out)):
        if not s['ok']:
            n['app_sem'] += 1; continue
        d = grupos.get(i)
        if d is None or (d['erro'].notna().any()):
            n['r_falhou'] += 1; continue
        n['comparados'] += 1
        pred = d.sort_values('linha')['pred'].to_numpy() - 1
        la = np.array(s['lab'])
        P = np.array(s['P'])
        top2 = np.sort(P, axis=1)
        # max.col do R (usado em predict.lda/qda) trata diferenças < 1e-5 como empate e sorteia
        emp = (top2[:, -1] - top2[:, -2]) < 1e-5
        n['empates_sem'] += int(emp.sum())
        dv = (pred != la) & ~emp
        if metodo == 'nb':
            # e1071 troca densidade 0 por threshold = 0.001: longe dos dados a resposta dele muda. Esses nós são separados.
            X, y = c['P'], c['y']
            zero = np.zeros(len(G), bool)
            for k in range(1, c['K'] + 1):
                A = X[y == k]
                sd = A.std(axis=0, ddof=1)
                if (sd <= 0).any():
                    zero[:] = True
                    continue
                dens = np.exp(-0.5 * ((G - A.mean(axis=0)) / sd) ** 2)
                zero |= (dens < 1e-300).any(axis=1)
            n['nos_efeito_threshold'] = n.get('nos_efeito_threshold', 0) + int((dv & zero).sum())
            dv = dv & ~zero
        m = int(dv.sum())
        if m:
            gap = float((top2[:, -1] - top2[:, -2])[dv].min())
            n['div_casos'] += 1; n['nos_divergentes'] += m; ex.append((c['tipo'], c['K'], len(c['y']), m, f'menor lacuna {gap:.2e}'))
    return n, ex


def main():
    r = np.random.default_rng(5)
    casos = gera(semente=11)
    jobs = []
    for c in casos:
        jobs.append({'pts': [[float(a), float(b), int(t)] for (a, b), t in zip(c['P'], c['y'])], 'K': c['K'], 'priors': 'eq' if r.random() < 0.3 else 'emp', 'grid': G.tolist()})

    def pr(jb):
        return None if jb['priors'] == 'emp' else [1 / jb['K']] * jb['K']

    def excl_qda(c, ref):
        # sklearn "ajusta" classes singulares com variância numérica; não é solução de verdade
        return any(np.min(s) < 1e-8 * max(np.max(s), 1e-300) for s in ref.scalings_)

    linhas = []
    for nome, mdl, exc in [('gnb', lambda jb: GaussianNB(priors=pr(jb)), None),
                           ('lda', lambda jb: LDA(solver='svd', priors=pr(jb)), None),
                           ('qda', lambda jb: QDA(priors=pr(jb), reg_param=0.0), excl_qda)]:
        n, maior, ex, ms = compara(nome, casos, jobs, mdl, exc)
        linhas += [f'{nome} sklearn: {n}', f'  maior |Δp| = {maior:.3e}; JS {ms:.0f} ms ({ms/len(jobs):.2f} ms/caso)'] + [f'  ex: {e}' for e in ex[:8]]

    # R: só priors estimadas (o padrão das funções de R) e GNB com ddof = 1, como o e1071
    jr = [dict(j, priors='emp') for j in jobs]
    for nome, metodo in [('lda', 'lda'), ('qda', 'qda')]:
        n, ex = compara_r(metodo, casos, jr, nome)
        linhas += [f'{nome} R: {n}'] + [f'  ex: {e}' for e in ex[:6]]
    jb2 = [dict(j, priors='emp', ddof=1) for j in jobs]
    n, ex = compara_r('nb', casos, jb2, 'gnb')
    linhas += [f'gnb R (ddof=1): {n}'] + [f'  ex: {e}' for e in ex[:6]]
    relatorio('t1_gerativos', linhas)


if __name__ == '__main__':
    main()
