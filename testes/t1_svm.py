"""SVM com orçamento C: 1000 conjuntos contra o QP direto (cvxopt), SVC(C=λ) e o LP do orçamento mínimo (HiGHS)."""
import warnings

import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import linprog
from sklearn.svm import SVC

from comum import relatorio, roda_js
from dados import gera

solvers.options.update(show_progress=False, abstol=1e-10, reltol=1e-10, feastol=1e-10, maxiters=400)
warnings.filterwarnings('ignore')


def gram(P, Q, k):
    D = P @ Q.T
    if k['type'] == 'linear':
        return D
    if k['type'] == 'poly':
        return (1 + D) ** k['d']
    if k['type'] == 'rbf':
        s = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(2)
        return np.exp(-k['gamma'] * s)
    return np.tanh(k['gamma'] * D + k['r'])


def qp_orcamento(K, y, C):
    """min ½α'Qα − Σα + Cλ  s.a. y'α = 0, 0 ≤ α ≤ λ. Devolve (α, λ) ou None."""
    n = len(y)
    kap = max(np.abs(np.diag(K)).max(), 1e-12)
    Q = np.zeros((n + 1, n + 1))
    Q[:n, :n] = np.outer(y, y) * K / kap
    Q += 1e-11 * np.eye(n + 1)
    q = np.r_[-np.ones(n), C]
    G = np.zeros((2 * n, n + 1))
    G[:n, :n] = -np.eye(n)
    G[n:, :n] = np.eye(n)
    G[n:, n] = -1
    A = matrix(np.r_[y, 0.0].reshape(1, n + 1))
    r = solvers.qp(matrix(Q), matrix(q), matrix(G), matrix(np.zeros(2 * n)), A, matrix(0.0))
    if r['status'] != 'optimal':
        return None
    x = np.array(r['x']).ravel()
    return x[:n] / kap, x[n] / kap


def qp_cmin_limitado(K, y, cost=1e5):
    """Σϵ da solução com penalidade fixa (o que o app chama de orçamento mínimo alcançável)."""
    n = len(y)
    kap = max(np.abs(np.diag(K)).max(), 1e-12)
    Q = np.outer(y, y) * K / kap + 1e-11 * np.eye(n)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.r_[np.zeros(n), np.full(n, cost * kap)]
    r = solvers.qp(matrix(Q), matrix(-np.ones(n)), matrix(G), matrix(h), matrix(y.reshape(1, n)), matrix(0.0))
    if r['status'] != 'optimal' and not (r['relative gap'] is not None and r['relative gap'] < 1e-6):
        return None
    al = np.array(r['x']).ravel() / kap
    g = (al * y) @ K
    b = min((y[i] - g[i] for i in range(n)), key=lambda t: np.maximum(0, 1 - y * (g + t)).sum())
    return float(np.maximum(0, 1 - y * (g + b)).sum())


def primal(K, y, al_y, b, cost):
    f = al_y @ K + b
    return 0.5 * al_y @ K @ al_y + cost * np.maximum(0, 1 - y * f).sum()


def lp_cmin(K, y):
    """Orçamento mínimo: min Σϵ com y f ≥ 1 − ϵ, f no espaço do kernel (fatoração espectral de K)."""
    n = len(y)
    v, U = np.linalg.eigh(K)
    keep = v > 1e-10 * max(v.max(), 1e-12)
    L = U[:, keep] * np.sqrt(v[keep])
    r_ = L.shape[1]
    c = np.r_[np.zeros(r_ + 1), np.ones(n)]
    A = np.hstack([-(y[:, None] * L), -y[:, None], -np.eye(n)])
    b = -np.ones(n)
    bounds = [(None, None)] * (r_ + 1) + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    return r.fun if r.status == 0 else None


def _dual(m, n):
    v = np.zeros(n)
    v[m.support_] = m.dual_coef_.ravel()
    return v


JS = """
const pts = caso.pts.map(p => ({x: p[0], y: p[1], c: p[2]}));
const s = Solver.solve(pts, caso.k, caso.C, 0);
return {kind: s.kind, Cmin: s.Cmin, Cmax: s.Cmax, cost: s.cost, hard: s.hard, alpha: s.alpha, b: s.b, norm2: s.norm2, sumEps: s.sumEps, converged: s.converged, approximate: s.approximate, vals: s.vals};
"""


def main():
    r = np.random.default_rng(99)
    casos = gera(semente=7, K_de=(2,), pesos=(1.0,))
    jobs = []
    for c in casos:
        y = np.where(c['y'] == 1, 1, -1)
        u = r.random()
        if u < 0.2:
            k = {'type': 'linear'}
        elif u < 0.4:
            k = {'type': 'poly', 'd': int(r.integers(1, 6))}
        elif u < 0.8:
            k = {'type': 'rbf', 'gamma': float(10 ** r.uniform(-2, 2))}
        else:
            k = {'type': 'sigmoid', 'gamma': float(10 ** r.uniform(-2, 0)), 'r': float(r.uniform(-2, 1))}
        Cmax = 2 * min((y == 1).sum(), (y == -1).sum())
        v = r.random()
        C = 0.0 if v < 0.05 else float(Cmax) if v < 0.1 else float(Cmax * r.random() ** 2)
        jobs.append({'pts': [[float(a), float(b), int(t)] for (a, b), t in zip(c['P'], y)], 'k': k, 'C': C})
    out, ms = roda_js('svm-kernels', JS, jobs, 't1_svm')

    n = dict(total=len(jobs), erroJS=0, ok=0, flat=0, infeasible=0, ref_falhou=0, naoconv=0, approx=0, naoconv_psd=0,
             div_qp=0, div_kind=0, div_svc=0, svc_parou_cedo=0, svc_falhou=0, div_cmin_lp=0, div_cmin_1e5=0,
             sig=0, sig_comparados=0, sig_div=0, sig_ref_cedo=0)
    maior = dict(fobj=0.0, f=0.0, spent=0.0, cmin=0.0, svc=0.0, sigf=0.0)
    ex = []
    for c, jb, s in zip(casos, jobs, out):
        if 'erro' in s:
            n['erroJS'] += 1
            ex.append(('erroJS', s['erro'][:300]))
            continue
        P = np.array([p[:2] for p in jb['pts']])
        y = np.array([p[2] for p in jb['pts']], float)
        k = jb['k']
        C = jb['C']
        K = gram(P, P, k)
        n[s['kind']] = n.get(s['kind'], 0) + 1
        if not s['converged']:
            n['naoconv'] += 1
            if k['type'] != 'sigmoid':
                n['naoconv_psd'] += 1
                ex.append(('naoconv', c['tipo'], k, C, s['kind'], s['Cmin'], s['cost']))
        if s.get('approximate'):
            n['approx'] += 1
        if k['type'] == 'sigmoid':
            n['sig'] += 1
            if s['kind'] != 'ok' or s['cost'] is None:
                continue
            try:
                m = SVC(C=s['cost'], kernel='sigmoid', gamma=k['gamma'], coef0=k['r'], tol=1e-5, max_iter=2000000).fit(P, y)
                f = m.decision_function(P)
            except Exception:
                n['svc_falhou'] += 1
                continue
            if not s['converged'] or s.get('approximate'):
                continue
            n['sig_comparados'] += 1
            if m.fit_status_ != 0:
                n['sig_ref_cedo'] += 1
                continue
            d = np.abs(f - (np.array(s['vals']) * y)).max()
            maior['sigf'] = max(maior['sigf'], d)
            if d > 1e-2 * (1 + np.abs(f).max()):
                n['sig_div'] += 1
                ex.append(('sig', c['tipo'], k, C, s['cost'], d))
            continue

        cmin = lp_cmin(K, y)
        c5 = qp_cmin_limitado(K, y)
        if cmin is None:
            n['ref_falhou'] += 1
            continue
        Cmax = s['Cmax']
        cmin = min(cmin, Cmax)
        # o app nunca pode dizer que cabe abaixo do mínimo matemático
        if s['Cmin'] < cmin - 1e-4 * (1 + cmin):
            n['div_cmin_lp'] += 1
            ex.append(('cmin_lp', c['tipo'], k, C, s['Cmin'], cmin))
        if c5 is None:
            n['ref_cmin_falhou'] = n.get('ref_cmin_falhou', 0) + 1
        else:
            c5 = min(c5, Cmax)
            d = abs(s['Cmin'] - c5)
            maior['cmin'] = max(maior['cmin'], d)
            if d > 1e-2 * (1 + c5):
                # Se o app gasta menos que a referência com a mesma penalidade (e nunca abaixo do LP), a referência é que parou longe do ótimo.
                if s['Cmin'] < c5 and s['Cmin'] >= cmin - 1e-4 * (1 + cmin):
                    n['ref_cmin_pior'] = n.get('ref_cmin_pior', 0) + 1
                else:
                    n['div_cmin_1e5'] += 1
                    ex.append(('cmin_1e5', c['tipo'], k, C, s['Cmin'], c5))
        ref_min = s['Cmin']
        esperado = 'flat' if C >= Cmax - 1e-9 else ('infeasible' if C < ref_min - 1e-6 * (1 + ref_min) else 'ok')
        if esperado != s['kind'] and not (esperado == 'ok' and s['kind'] == 'flat'):
            n['div_kind'] += 1
            ex.append(('kind', c['tipo'], k, C, s['kind'], esperado, ref_min))
            continue
        if s['kind'] != 'ok':
            continue
        if s['hard'] or C <= s['Cmin'] + 1e-4 * (1 + s['Cmin']):
            # margem rígida ou orçamento colado no mínimo: a forma com λ não tem ótimo finito estável; compara só o gasto e f via SVC
            pass
        else:
            q = None
            try:
                q = qp_orcamento(K, y, C)
            except Exception:
                q = None
            if q is None:
                n['ref_falhou'] += 1
            else:
                al, lam = q
                g = (al * y) @ K
                w2_ref = float((al * y) @ K @ (al * y))
                w2 = s['norm2']
                fobj = abs(w2 - w2_ref) / (1e-6 + abs(w2_ref))
                maior['fobj'] = max(maior['fobj'], fobj)
                maior['spent'] = max(maior['spent'], abs(s['sumEps'] - C))
                fref = g
                fapp = (np.array(s['alpha']) * y) @ K
                d = np.abs((fapp - fapp.mean()) - (fref - fref.mean())).max()
                maior['f'] = max(maior['f'], d)
                if fobj > 1e-3 or abs(s['sumEps'] - C) > 1e-3 * (1 + C):
                    n['div_qp'] += 1
                    ex.append(('qp', c['tipo'], k, C, w2, w2_ref, s['sumEps']))
        # SVC com o C equivalente
        try:
            m = SVC(C=s['cost'], kernel={'linear': 'linear', 'poly': 'poly', 'rbf': 'rbf'}[k['type']], gamma=k.get('gamma', 1), coef0=1, degree=k.get('d', 3), tol=1e-6, max_iter=5000000).fit(P, y)
            f = m.decision_function(P)
        except Exception:
            n['svc_falhou'] += 1
            continue
        d = np.abs(f - (np.array(s['vals']) * y)).max()
        maior['svc'] = max(maior['svc'], d)
        if d > 5e-3 * (1 + np.abs(f).max()):
            pa = primal(K, y, np.array(s['alpha']) * y, s['b'], s['cost'])
            ps = primal(K, y, np.bincount(m.support_, minlength=len(y)).astype(float) * 0 + (lambda v: v)(_dual(m, len(y))), float(m.intercept_[0]), s['cost'])
            if m.fit_status_ != 0 or pa <= ps * (1 + 1e-7):
                n['svc_parou_cedo'] += 1
            else:
                n['div_svc'] += 1
                ex.append(('svc', c['tipo'], k, C, s['cost'], d, pa, ps))
    linhas = [f'SVM orçamento: {n}', f'maior: {maior}', f'tempo JS total {ms:.0f} ms ({ms/len(jobs):.2f} ms/caso)']
    linhas += [f'  ex: {e}' for e in ex[:25]]
    relatorio('t1_svm', linhas)


if __name__ == '__main__':
    main()
