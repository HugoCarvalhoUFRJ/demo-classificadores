"""Testes no navegador (Chrome já instalado, perfil temporário): ponta a ponta, interação aleatória, acessibilidade, contextos, desempenho e capturas.
Uso: python t_nav.py [e2e|aleatorio|axe|contextos|desempenho|capturas|todos]
"""
import http.server
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import warnings
from functools import partial

import numpy as np
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import sync_playwright
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from comum import PROJ, RES, relatorio

warnings.filterwarnings('ignore')
Rs = r'C:\Program Files\R\R-4.5.1\bin\Rscript.exe'
APPS = ['gnb', 'lda', 'qda', 'margem-maxima', 'vetor-suporte', 'svm-kernels', 'arvore', 'floresta', 'knn']
NOVOS = ['gnb', 'lda', 'qda', 'arvore', 'floresta', 'knn']
PORTA = 8765
BASE = f'http://127.0.0.1:{PORTA}'
X0, Y1, ML, MT, S = -1.73, 3.75, 54, 8, 100


class Quieto(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def servidor():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', PORTA), partial(Quieto, directory=str(PROJ)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def novo_navegador(p):
    return p.chromium.launch(channel='chrome', headless=True)


def coleta_erros(page, lista):
    def console(m):
        if m.type == 'error' and 'fonts.g' not in m.text and 'ERR_' not in m.text and 'favicon' not in (m.location or {}).get('url', ''):
            lista.append(m.text)
    page.on('console', console)
    page.on('pageerror', lambda e: lista.append(str(e)))


LE_PONTOS = """() => [...document.querySelectorAll('#layer-points .pt')].map(n => {
  const d = n.querySelector('.dot'), c = [...n.classList].find(x => x.length === 1);
  return {cx: +d.getAttribute('cx'), cy: +d.getAttribute('cy'), c};
})"""
LE_DOTS = """() => {
  const cv = document.getElementById('cv'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  const cs = getComputedStyle(document.documentElement), hex = n => { const h = cs.getPropertyValue(n).trim().replace('#',''); return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; };
  const cores = ['--blue-dot','--pink-dot','--orange-dot','--green-dot'].map(hex);
  const out = [], NX = 106, NY = 111, img = ctx.getImageData(0,0,W,H).data;
  for (let j = 1; j < NY; j += 4) for (let i = 1; i < NX; i += 4) {
    const x = Math.round(i/(NX-1)*W), y = Math.round(H - j/(NY-1)*H), q = (Math.min(y, H-1))*W*4 + Math.min(x, W-1)*4;
    const c = [img[q], img[q+1], img[q+2]];
    let b = -1, bd = 900;
    cores.forEach((v, k) => { const d = Math.abs(v[0]-c[0])+Math.abs(v[1]-c[1])+Math.abs(v[2]-c[2]); if (d < bd) { bd = d; b = k; } });
    out.push({i, j, k: b});
  }
  return out;
}"""


def pontos_da_tela(page):
    bruto = page.evaluate(LE_PONTOS)
    letra = {'b': 1, 'p': 2, 'o': 3, 'g': 4}
    P = np.array([[X0 + (b['cx'] - ML) / S, Y1 - (b['cy'] - MT) / S] for b in bruto])
    y = np.array([letra[b['c']] for b in bruto])
    return P, y


def grade_dots(dots):
    HX = 5.25 / 105
    HY = 5.52 / 110
    G = np.array([[X0 + d['i'] * HX, -1.77 + d['j'] * HY] for d in dots])
    k = np.array([d['k'] for d in dots])
    return G, k


def espera_quadro(page):
    page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    page.wait_for_timeout(30)


def arrasta(page, de, para):
    box = page.locator('#plot').bounding_box()
    f = box['width'] / 602
    a = (box['x'] + de[0] * f, box['y'] + de[1] * f)
    b = (box['x'] + para[0] * f, box['y'] + para[1] * f)
    page.mouse.move(*a)
    page.mouse.down()
    for t in np.linspace(0, 1, 6)[1:]:
        page.mouse.move(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        espera_quadro(page)
    page.mouse.up()
    espera_quadro(page)


def r_executa(linha, P, y, G, svm=False):
    """Executa literalmente a linha de R mostrada na tela."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as d:
        np.savetxt(os.path.join(d, 'P.csv'), np.column_stack([P, y]), delimiter=',')
        np.savetxt(os.path.join(d, 'G.csv'), G, delimiter=',')
        with open(os.path.join(d, 'linha.txt'), 'w', encoding='utf-8') as f:
            f.write(linha)
        scr = r"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({library(MASS); library(e1071); library(class); library(rpart); library(randomForest); suppressWarnings(library(kknn))})
args <- commandArgs(TRUE); d <- args[1]
A <- as.matrix(read.csv(file.path(d, "P.csv"), header = FALSE)); G <- as.matrix(read.csv(file.path(d, "G.csv"), header = FALSE))
dados <- data.frame(X1 = A[, 1], X2 = A[, 2], y = factor(A[, 3]))
X <- dados[, c("X1", "X2")]; y <- dados$y
x0 <- data.frame(X1 = G[, 1], X2 = G[, 2]); novos <- x0
l <- readLines(file.path(d, "linha.txt"), encoding = "UTF-8")
exprs <- parse(text = l)
for (e in exprs) r <- eval(e, envir = environment())
fit <- if (exists("m")) m else r
out <- if (inherits(fit, "factor")) as.character(fit) else if (inherits(fit, "kknn")) as.character(fitted(fit)) else if (inherits(fit, "svm") || inherits(fit, "randomForest")) as.character(predict(fit, x0)) else if (inherits(fit, "rpart")) as.character(predict(fit, x0, type = "class")) else if (inherits(fit, "naiveBayes")) as.character(predict(fit, x0, type = "class")) else as.character(predict(fit, x0)$class)
cat(out, sep = "\n")
"""
        arq = os.path.join(d, 'r.R')
        with open(arq, 'w', encoding='utf-8') as f:
            f.write(scr)
        r = subprocess.run([Rs, '--vanilla', arq, d], capture_output=True)
        if r.returncode != 0:
            return None, r.stderr.decode('latin-1')[-400:]
        return np.array([float(v) for v in r.stdout.decode('latin-1').split()]), ''


# ------------------------------------------------------------------ e2e
ESTADOS = {
    'gnb': [('tresdif', None), ('corr', 'eq'), ('quad', None), ('duasig', None), ('duasdif', None), ('mix3', {'ruido': 0.3, 'teste': 1})],
    'lda': [('tresig', None), ('figure', 'eq'), ('quad', None), ('mix2', {'ruido': 0.2, 'teste': 1})],
    'qda': [('tresdif', None), ('corr', 'eq'), ('xor', None), ('moons', {'ruido': 0.45, 'teste': 1})],
    'arvore': [('xor', {'p-leaf': 2}), ('tresdif', {'p-alpha': 0.02}), ('moons', {'p-depth': 3}), ('mix2', {'ruido': 0.25, 'teste': 1, 'p-leaf': 4})],
    'floresta': [('moons', {}), ('tresig', {'mseg': '2'}), ('quad', {'p-b': 30}), ('duasdif', {'mseg': '2', 'p-leaf': 3}), ('mix3', {'ruido': 0.3, 'teste': 1, 'p-leaf': 2, 'p-depth': 4})],
    'knn': [('moons', {'p-k': 5}), ('tresdif', {'p-k': 1}), ('ring', {'p-k': 8}), ('moons', {'p-k': 6, 'wseg': 'distance', 'pseg': '1'}),
            ('mix2', {'p-k': 9, 'pseg': '3', 'ruido': 0.3, 'teste': 1}), ('mix3', {'p-k': 4, 'wseg': 'distance', 'pseg': '2', 'teste': 1}), ('board', {'p-k': 7, 'pseg': '1', 'ruido': 0.2})],
}


def set_range(page, id_, v):
    page.evaluate("([i,v]) => { const e = document.getElementById(i); e.value = v; e.dispatchEvent(new Event('input', {bubbles:true})); }", [id_, v])


LE_TESTE = """() => [...document.querySelectorAll('#layer-test .tpt')].map(n => {
  const c = n.querySelector('circle'), x = n.querySelector('path');
  return {cx: +c.getAttribute('cx'), cy: +c.getAttribute('cy'), c: +n.dataset.c, errado: (x.getAttribute('d') || '') !== ''};
})"""


LE_CONF = """() => {
  const t = document.querySelector('#v-conf table'); if (!t || document.getElementById('s-conf').hidden) return null;
  const linhas = [...t.tBodies[0].rows].map(r => [...r.cells].slice(1).map(c => +c.textContent));
  const med = [...document.querySelectorAll('#v-metr tbody tr')].map(r => [r.cells[0].textContent, ...[...r.cells].slice(1).map(c => c.textContent)]);
  const pos = document.querySelector('#posseg button[aria-pressed="true"]');
  return {linhas, med, pos: pos ? +pos.dataset.c : null};
}"""


def confere_conf(page, pred, yt, classes, exato=True):
    """Matriz de confusão do teste (previsto × verdadeiro, com totais) e medidas da Tabela 4.7, recalculadas aqui."""
    c = page.evaluate(LE_CONF)
    if c is None:
        return 'matriz: ausente'
    K = len(classes)
    M = np.array([r[:K] for r in c['linhas'][:K]])
    ok_tot = all(r[K] == sum(r[:K]) for r in c['linhas'][:K]) and c['linhas'][K][:K] == M.sum(0).tolist() and c['linhas'][K][K] == len(yt)
    esp = np.array([[int(((pred == a) & (yt == b)).sum()) for b in classes] for a in classes])
    ok_ref = (M == esp).all() if exato else True
    ok_x = int(M.sum() - np.trace(M)) == int((pred != yt).sum()) if exato else True
    # medidas: refaz cada fração a partir da própria matriz exibida
    def frac(txt):
        import re as _re
        m = _re.search(r'(\d+)/(\d+)', txt)
        return (int(m.group(1)), int(m.group(2))) if m else None
    ruins = 0
    erro = frac(c['med'][0][-1])
    if erro != (int(M.sum() - np.trace(M)), int(M.sum())):
        ruins += 1
    from sklearn.metrics import precision_recall_fscore_support as prf

    def pct(txt):
        import re as _re
        m = _re.search(r'([\d,]+)%', txt)
        return float(m.group(1).replace(',', '.')) / 100 if m else None
    # referência das medidas: a própria matriz exibida, convertida de volta em pares (previsto, verdadeiro)
    pv = np.concatenate([[classes[i]] * int(M[i, j]) for i in range(K) for j in range(K)]) if M.sum() else np.array([])
    vv = np.concatenate([[classes[j]] * int(M[i, j]) for i in range(K) for j in range(K)]) if M.sum() else np.array([])
    if K == 2:
        p = classes.index(c['pos']); q = 1 - p
        TP, FN, FP, TN = M[p, p], M[q, p], M[p, q], M[q, q]
        for linha, (a_, b_) in zip(c['med'][1:5], [(FP, FP + TN), (TP, TP + FN), (TP, TP + FP), (TN, TN + FN)]):
            if frac(linha[-1]) != (int(a_), int(b_)):
                ruins += 1
        _, _, f1, _ = prf(vv, pv, labels=[c['pos']], average=None, zero_division=0)
        if frac(c['med'][5][-1]) != (int(2 * TP), int(2 * TP + FP + FN)) or abs(pct(c['med'][5][-1]) - f1[0]) > 5.1e-4:
            ruins += 1
    else:
        P_, R_, F_, _ = prf(vv, pv, labels=classes, average=None, zero_division=0)
        for j, linha in enumerate(c['med'][1:K + 1]):
            for txt, ref_v in ((linha[1], R_[j]), (linha[2], P_[j]), (linha[3], F_[j])):
                v = pct(txt)
                if (v is None and not (txt.startswith('—') and ref_v == 0)) or (v is not None and abs(v - ref_v) > 5.1e-4):
                    ruins += 1
        for linha, av in ((c['med'][K + 1], 'macro'), (c['med'][K + 2], 'micro')):
            Pm, Rm, Fm, _ = prf(vv, pv, labels=classes, average=av, zero_division=0)
            if any(abs(pct(t) - r_) > 5.1e-4 for t, r_ in zip(linha[1:4], (Rm, Pm, Fm))):
                ruins += 1
    return f'matriz {K}×{K}: totais ok={ok_tot}, igual à referência={ok_ref}, fora da diagonal = erros={ok_x}, medidas com problema={ruins}' + ('' if exato else ' (referência aleatória: só coerência)')


def confere_dados(page, ref, P, y):
    """n por classe e erros de treino e teste exibidos, contra o modelo de referência."""
    txt = page.inner_text('#v-data').replace('\n', ' ')
    cont = [int(v) for v in re.findall(r'(\d+) (?:azu|rox|lar|verd)', txt)]
    ok_n = sum(cont) == len(y) and cont == [int((y == k).sum()) for k in range(1, len(cont) + 1)]
    out = f'n por classe {cont} ok={ok_n}'
    T = page.evaluate(LE_TESTE)
    if T:
        Q = np.array([[X0 + (t['cx'] - ML) / S, Y1 - (t['cy'] - MT) / S] for t in T])
        yt = np.array([t['c'] for t in T])
        err_ref = ref.predict(Q) != yt
        err_tela = np.array([t['errado'] for t in T])
        m = re.search(r'de teste ([\d,]+)%', txt)
        pct = float(m.group(1).replace(',', '.')) if m else float('nan')
        out += f'; teste m={len(T)}: × na tela={int(err_tela.sum())}, referência={int(err_ref.sum())}, pontos diferentes={int((err_ref != err_tela).sum())}, % exibido={pct} (esperado {100 * err_tela.mean():.1f})'
        classes = sorted(set(int(v) for v in y) | set(int(v) for v in yt))
        exato = type(ref).__name__ != 'RandomForestClassifier'
        out += '; ' + confere_conf(page, ref.predict(Q), yt, classes, exato)
        if len(classes) == 2 and page.locator('#posseg button').count() == 2:
            page.locator('#posseg button').nth(1).click()
            espera_quadro(page)
            out += ' | + = segunda classe: ' + confere_conf(page, ref.predict(Q), yt, classes, exato).split(': ', 1)[1]
            page.locator('#posseg button').nth(0).click()
            espera_quadro(page)
    return out


LE_NO = """() => {
  const p = document.querySelector('#layer-extra .nodebox'); if (!p) return null;
  const v = p.getAttribute('d').match(/-?[\\d.]+/g).map(Number);
  return {x0: v[0], y0: v[1], x1: v[2], y1: v[3], rings: [...document.querySelectorAll('#layer-points .ring')].filter(r => r.style.display !== 'none').length, say: document.getElementById('say').innerText};
}"""


def confere_nos(page, ref):
    """Passa o mouse em cada nó do diagrama e compara a região realçada com os limiares do tree_ do scikit-learn."""
    t = ref.tree_
    n = page.locator('#v-tree .tn').count()
    if n != t.node_count:
        return f'nós: app {n}, sklearn {t.node_count} (árvores diferentes; sem comparação)'
    lim = [None] * n

    def rec(j, lo, hi):
        lim[j] = (lo, hi)
        if t.children_left[j] < 0:
            return
        f_, th = int(t.feature[j]), float(t.threshold[j])
        h2 = list(hi); h2[f_] = min(hi[f_], th)
        l2 = list(lo); l2[f_] = max(lo[f_], th)
        rec(int(t.children_left[j]), lo, h2)
        rec(int(t.children_right[j]), l2, hi)
    rec(0, [-np.inf, -np.inf], [np.inf, np.inf])
    ruins, maior = 0, 0.0
    for j in range(n):
        page.locator('#v-tree .tn').nth(j).hover()
        espera_quadro(page)
        b = page.evaluate(LE_NO)
        if b is None:
            ruins += 1
            continue
        lo, hi = lim[j]
        cl = lambda v, a, z: max(a, min(z, v))
        esp = (cl(lo[0], X0, 3.52), cl(hi[1], -1.77, Y1), cl(hi[0], X0, 3.52), cl(lo[1], -1.77, Y1))
        tela = (X0 + (b['x0'] - ML) / S, Y1 - (b['y0'] - MT) / S, X0 + (b['x1'] - ML) / S, Y1 - (b['y1'] - MT) / S)
        d = max(abs(a - c) for a, c in zip(esp, tela))
        maior = max(maior, d)
        if d > 2e-3 or b['rings'] != int(t.n_node_samples[j]) or f'{int(t.n_node_samples[j])} ponto' not in b['say']:
            ruins += 1
    page.mouse.move(5, 5)
    return f'realce de nós: {n} nós, {ruins} com problema, maior diferença do retângulo {maior:.1e}'


def confere_ramo(page, ref, n_pos=25):
    """Mouse em posições do plano: o ramo aceso no diagrama tem de ser o decision_path do scikit-learn, e o contorno, a região da folha."""
    t = ref.tree_
    if page.locator('#v-tree .tn').count() != t.node_count:
        return 'ramo: árvores diferentes; sem comparação'
    rnd = random.Random(7)
    box = page.locator('#plot').bounding_box()
    f = box['width'] / 602
    pts = page.evaluate(LE_PONTOS)
    ruins = testadas = 0
    while testadas < n_pos:
        sx, sy = rnd.uniform(60, 575), rnd.uniform(14, 555)
        if any(abs(sx - q['cx']) < 20 and abs(sy - q['cy']) < 20 for q in pts):
            continue  # perto de um ponto o cursor seleciona o ponto, não a posição
        testadas += 1
        page.mouse.move(box['x'] + sx * f, box['y'] + sy * f)
        espera_quadro(page)
        x, y = X0 + (sx - ML) / S, Y1 - (sy - MT) / S
        tela = page.evaluate("() => ({on: [...document.querySelectorAll('#v-tree .tn.on')].map(g => +g.dataset.id).sort((a, b) => a - b), arestas: document.querySelectorAll('#v-tree .te.on').length, d: (document.getElementById('leafbox') || {getAttribute: () => ''}).getAttribute('d')})")
        # a posição do mouse na tela não é exatamente (x, y): usa a que o app narrou
        m = re.search(r'\(([−\-\d,]+); ([−\-\d,]+)\)', page.inner_text('#say'))
        if m:
            x, y = (float(v.replace('−', '-').replace(',', '.')) for v in m.groups())
        esp = sorted(int(j) for j in ref.decision_path(np.array([[x, y]])).indices)
        if tela['on'] != esp or tela['arestas'] != len(esp) - 1 or not tela['d']:
            ruins += 1
    page.mouse.move(5, 5)
    return f'ramo sob o mouse: {testadas} posições, {ruins} com problema'


def e2e_novos(page, app, linhas):
    for k, (ds, ajuste) in enumerate(ESTADOS[app]):
        page.select_option('#dataset', ds)
        espera_quadro(page)
        if app in ('gnb', 'lda', 'qda') and ajuste == 'eq':
            page.click('#priors button[data-pri="eq"]')
        if isinstance(ajuste, dict):
            for id_, v in ajuste.items():
                if id_ == 'mseg':
                    page.click(f'#mseg button[data-m="{v}"]')
                elif id_ == 'wseg':
                    page.click(f'#wseg button[data-w="{v}"]')
                elif id_ == 'pseg':
                    page.click(f'#pseg button[data-p="{v}"]')
                elif id_ == 'ruido':
                    page.fill('#p-noise-n', str(v))
                elif id_ == 'teste':
                    page.check('#o-test')
                else:
                    set_range(page, id_, v)
                espera_quadro(page)
        espera_quadro(page)
        # arrasta um ponto de verdade
        pts = page.evaluate(LE_PONTOS)
        p0 = pts[3]
        arrasta(page, (p0['cx'], p0['cy']), (p0['cx'] + 70, p0['cy'] - 40))
        espera_quadro(page)
        P, y = pontos_da_tela(page)
        code_py = page.inner_text('#code-py')
        code_r = page.inner_text('#code-r')
        pill = page.inner_text('#pill')
        dots = page.evaluate(LE_DOTS)
        G, kk = grade_dots(dots)
        ns = {c.__name__: c for c in (GaussianNB, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier)}
        ref = eval(code_py, ns).fit(P, y)
        Lr = ref.predict(G) - 1
        tem = kk >= 0
        conc = float((Lr[tem] == kk[tem]).mean()) if tem.any() else float('nan')
        nerr = int(re.search(r'(\d+) de', pill).group(1))
        pt_err = int((ref.predict(P) != y).sum())
        # R: literal
        lin_r = re.sub(r'#.*$', '', code_r).strip()
        if app == 'knn':
            ra, er = r_executa(lin_r, P, y, G)
        else:
            ra, er = r_executa(lin_r, P, y, G)
        if ra is None:
            conc_r = f'erro R: {er}'
        else:
            mk = tem & np.isfinite(ra[:len(G)])
            conc_r = f'{float((ra[mk].astype(int) - 1 == kk[mk]).mean()):.4f}'
        linhas.append(f'{app}[{k}] {ds}: n={len(y)} dots={int(tem.sum())}/{len(tem)} sklearn(concordância)={conc:.4f} erros tela={nerr} sklearn={pt_err} R(literal)={conc_r}')
        linhas.append(f'   py: {code_py}')
        linhas.append('   ' + confere_dados(page, ref, P, y))
        if app == 'arvore':
            linhas.append('   ' + confere_nos(page, ref))
            linhas.append('   ' + confere_ramo(page, ref))
        # o n do texto da opção é o do gerador (sem edições)
        page.evaluate("() => document.getElementById('reset').click()")
        espera_quadro(page)
        n_op = int(re.search(r'n = (\d+)', page.evaluate("() => { const s = document.getElementById('dataset'); return s.options[s.selectedIndex].textContent; }")).group(1))
        n_tela = len(page.evaluate(LE_PONTOS))
        if n_op != n_tela:
            linhas.append(f'   PROBLEMA: opção diz n = {n_op}, tela tem {n_tela}')
        page.fill('#p-noise-n', '0')
        if page.is_checked('#o-test'):
            page.uncheck('#o-test')
        espera_quadro(page)


def e2e_svm(page, linhas):
    def verifica(rot):
        P, y = pontos_da_tela(page)
        Y = np.where(y == 1, 1, -1)
        cpy = page.inner_text('#code-py')
        cr = page.inner_text('#code-r')
        pill = page.inner_text('#pill')
        say = page.inner_text('#say')
        spent = page.inner_text('#v-spendlab')
        if cpy.startswith('#'):
            linhas.append(f'svm {rot}: {cpy} | say: {say[:90]}')
            return
        dots = page.evaluate(LE_DOTS)
        G, kk = grade_dots(dots)
        m = eval(cpy, {'SVC': SVC}).fit(P, Y)
        Lr = np.where(m.predict(G) == 1, 0, 1)
        tem = kk >= 0
        conc = float((Lr[tem] == kk[tem]).mean())
        Yr = np.where(Y == 1, 2, 1)  # rótulos do fator em R; o fator só precisa separar as classes
        ra, er = r_executa(cr, P, np.where(Y == 1, 1, 0), G)
        conc_r = 'erro ' + er if ra is None else f'{float((((1 - ra.astype(int)) == kk)[tem]).mean()):.4f}'
        n_err = int(re.search(r'(\d+) de', pill).group(1))
        linhas.append(f'svm {rot}: {spent.strip()} | erros tela={n_err} SVC={int((m.predict(P) != Y).sum())} concordância SVC={conc:.4f} R literal={conc_r} | {cpy}')

    def fix_range(id_, v):
        set_range(page, id_, v)
        espera_quadro(page)

    page.select_option('#dataset', 'mix2')
    page.check('#o-test')
    espera_quadro(page)
    P, y = pontos_da_tela(page)
    Y = np.where(y == 1, 1, -1)
    m = eval(page.inner_text('#code-py'), {'SVC': SVC}).fit(P, Y)
    T = page.evaluate(LE_TESTE)
    Q = np.array([[X0 + (t['cx'] - ML) / S, Y1 - (t['cy'] - MT) / S] for t in T])
    yt = np.array([t['c'] for t in T])
    linhas.append('svm mistura com teste: ' + confere_conf(page, m.predict(Q), yt, [1, -1]))
    page.uncheck('#o-test')
    page.select_option('#dataset', 'ring')
    espera_quadro(page)
    verifica('anel/rbf')
    pts = page.evaluate(LE_PONTOS)
    arrasta(page, (pts[0]['cx'], pts[0]['cy']), (pts[0]['cx'] - 70, pts[0]['cy'] + 30))
    verifica('anel/rbf arrastado')
    page.click('#k-linear')
    espera_quadro(page)
    verifica('anel/linear (C ajustado)')
    page.select_option('#dataset', 'figure')
    espera_quadro(page)
    fix_range('c-range', 300)
    verifica('figura/linear')
    page.click('#k-poly')
    espera_quadro(page)
    verifica('figura/poly')
    page.click('#k-sigmoid')
    espera_quadro(page)
    verifica('figura/sigmoide')
    page.click('#k-rbf')
    espera_quadro(page)
    page.fill('#c-num', '100')
    espera_quadro(page)
    page.locator('#plot').hover()
    linhas.append(f'svm C=100 (≥ Cmax): say={page.inner_text("#say")[:110]} M={page.inner_text("#v-m")}')


def nos_toque_teclado(b, linhas):
    """Realce dos nós da árvore por toque (liga e desliga) e pelo teclado (Tab, setas, Enter, Esc)."""
    tem = "() => !!document.querySelector('#layer-extra .nodebox')"
    ctx = b.new_context(viewport={'width': 390, 'height': 800}, has_touch=True, is_mobile=True)
    page = ctx.new_page()
    page.goto(f'{BASE}/arvore.html')
    espera_quadro(page)
    page.locator('#v-tree .tn').nth(1).tap()
    espera_quadro(page)
    r1 = page.evaluate(tem)
    page.locator('#v-tree .tn').nth(1).tap()
    espera_quadro(page)
    r2 = page.evaluate(tem)
    ctx.close()
    ctx = b.new_context(viewport={'width': 1280, 'height': 950})
    page = ctx.new_page()
    page.goto(f'{BASE}/arvore.html')
    espera_quadro(page)
    page.focus('#v-tree')
    page.keyboard.press('Tab')
    espera_quadro(page)
    k1 = page.evaluate("() => [document.activeElement.dataset.id, !!document.querySelector('#layer-extra .nodebox')]")
    page.keyboard.press('ArrowLeft')
    espera_quadro(page)
    k2 = page.evaluate("() => document.activeElement.dataset.id")
    page.keyboard.press('ArrowUp')
    espera_quadro(page)
    k3 = page.evaluate("() => document.activeElement.dataset.id")
    page.keyboard.press('Enter')
    espera_quadro(page)
    k4 = page.evaluate(tem)
    page.keyboard.press('Enter')
    espera_quadro(page)
    page.keyboard.press('Escape')
    espera_quadro(page)
    k5 = page.evaluate(tem)
    ctx.close()
    ok = r1 and not r2 and k1 == ['0', True] and k2 == '1' and k3 == '0' and not k4 and not k5
    linhas.append(f'árvore, nós: toque liga={r1} toque de novo desliga={not r2}; teclado Tab={k1} ←={k2} ↑={k3} Enter desliga={not k4} Esc={not k5} -> {"ok" if ok else "PROBLEMA"}')


def e2e(linhas):
    with sync_playwright() as p:
        b = novo_navegador(p)
        ctx = b.new_context(viewport={'width': 1280, 'height': 950})
        page = ctx.new_page()
        erros = []
        coleta_erros(page, erros)
        for app in NOVOS:
            page.goto(f'{BASE}/{app}.html')
            espera_quadro(page)
            e2e_novos(page, app, linhas)
        page.goto(f'{BASE}/svm-kernels.html')
        espera_quadro(page)
        e2e_svm(page, linhas)
        nos_toque_teclado(b, linhas)
        linhas.append(f'erros no console: {erros[:5]}')
        b.close()


# ------------------------------------------------------------------ aleatório
def aleatorio(linhas, n=300):
    rnd = random.Random(1234)
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in APPS:
            ctx = b.new_context(viewport={'width': 1200, 'height': 900})
            page = ctx.new_page()
            erros = []
            coleta_erros(page, erros)
            page.goto(f'{BASE}/{app}.html')
            espera_quadro(page)
            ruins = set()
            acoes = 0
            for a in range(n):
                acao = rnd.choice(['drag', 'drag', 'dset', 'tool', 'add', 'rem', 'param', 'param', 'key', 'hover', 'chk', 'no'])
                try:
                    box = page.locator('#plot').bounding_box()
                    if acao == 'drag':
                        pts = page.evaluate(LE_PONTOS)
                        if pts:
                            q = rnd.choice(pts)
                            arrasta(page, (q['cx'], q['cy']), (rnd.uniform(70, 570), rnd.uniform(20, 550)))
                    elif acao == 'dset' and page.locator('#dataset').count():
                        ops = page.evaluate("() => [...document.getElementById('dataset').options].map(o => o.value)")
                        page.select_option('#dataset', rnd.choice(ops))
                    elif acao == 'tool':
                        bs = page.locator('.seg button:visible').all()
                        if bs:
                            rnd.choice(bs).click()
                    elif acao == 'add':
                        bs = page.locator('[id^=t-add]:visible').all()
                        if bs:
                            rnd.choice(bs).click()
                            page.mouse.click(box['x'] + box['width'] * rnd.uniform(.15, .9), box['y'] + box['height'] * rnd.uniform(.05, .85))
                            page.click('#t-move')
                    elif acao == 'rem' and page.locator('#t-rem').count():
                        page.click('#t-rem')
                        pts = page.evaluate(LE_PONTOS)
                        if pts:
                            q = rnd.choice(pts)
                            page.mouse.click(box['x'] + q['cx'] * box['width'] / 602, box['y'] + q['cy'] * box['width'] / 602)
                        page.click('#t-move')
                    elif acao == 'param':
                        rs = page.locator('input[type=range]:visible').all()
                        if rs:
                            r = rnd.choice(rs)
                            mi = float(r.get_attribute('min') or 0)
                            ma = float(r.get_attribute('max') or 100)
                            r.evaluate("(e, v) => { e.value = v; e.dispatchEvent(new Event('input', {bubbles:true})); }", rnd.uniform(mi, ma))
                    elif acao == 'chk':
                        cs = page.locator('.chk:visible input:enabled, .toggles input:visible').all()
                        if cs:
                            rnd.choice(cs).click()
                    elif acao == 'no':
                        ns = page.locator('#v-tree .tn').all()
                        if ns:
                            g = rnd.choice(ns)
                            if rnd.random() < .5:
                                g.hover()
                            else:
                                g.focus()
                                page.keyboard.press(rnd.choice(['ArrowUp', 'ArrowLeft', 'ArrowRight', 'Home', 'Escape']))
                    elif acao == 'key':
                        pts = page.locator('#layer-points .pt').all()
                        if pts:
                            rnd.choice(pts).focus()
                            page.keyboard.press(rnd.choice(['ArrowLeft', 'ArrowUp', 'Shift+ArrowRight', 'Delete']))
                    else:
                        page.mouse.move(box['x'] + box['width'] * rnd.random(), box['y'] + box['height'] * rnd.random())
                    espera_quadro(page)
                    acoes += 1
                except Exception as e:
                    erros.append(f'{acao}: {e}'[:200])
                st = page.evaluate("""() => {
                  const t = document.body.innerText, plotPts = [...document.querySelectorAll('#layer-points .dot, #layer-test circle')];
                  const fora = plotPts.some(d => { const x = +d.getAttribute('cx'), y = +d.getAttribute('cy'); return !(x >= 54 && x <= 579 && y >= 8 && y <= 560); });
                  const paths = [...document.querySelectorAll('svg path')].map(p => p.getAttribute('d') || '').join(' ');
                  return { nan: /NaN|undefined/.test(t) || /NaN|undefined|Infinity/.test(paths), inf: /Infinity/.test(t), fora, scroll: document.documentElement.scrollWidth > innerWidth + 1 };
                }""")
                for k, v in st.items():
                    if v:
                        ruins.add(k)
            linhas.append(f'{app}: {acoes}/{n} ações, problemas={sorted(ruins)}, erros de console/página={len(erros)} {erros[:3]}')
            ctx.close()
        b.close()


# ------------------------------------------------------------------ axe
def axe(linhas):
    with sync_playwright() as p:
        b = novo_navegador(p)
        ax = Axe()
        total = 0
        for tema in ['light', 'dark']:
            ctx = b.new_context(viewport={'width': 1200, 'height': 900}, color_scheme=tema)
            page = ctx.new_page()
            for app in ['index'] + APPS:
                page.goto(f'{BASE}/{app}.html' + ('' if app != 'index' else '#gnb'))
                espera_quadro(page)
                page.wait_for_timeout(100)
                r = ax.run(page)
                v = r.response['violations']
                total += len(v)
                det = [(x['id'], x['impact'], len(x['nodes']), x['nodes'][0]['html'][:80]) for x in v]
                linhas.append(f'axe {tema} {app}: {len(v)} violações {det[:4]}')
            ctx.close()
        linhas.append(f'axe total de violações: {total}')
        b.close()


# ------------------------------------------------------------------ contextos
def contextos(linhas):
    with sync_playwright() as p:
        b = novo_navegador(p)
        nprob = 0
        for proto in ['http', 'file']:
            for vp, toque in [((1280, 900), False), ((390, 800), True)]:
                for tema in ['light', 'dark']:
                    ctx = b.new_context(viewport={'width': vp[0], 'height': vp[1]}, has_touch=toque, is_mobile=toque, color_scheme=tema)
                    page = ctx.new_page()
                    erros = []
                    coleta_erros(page, erros)
                    base = BASE if proto == 'http' else PROJ.as_uri()
                    # sozinho
                    for app in APPS:
                        page.goto(f'{base}/{app}.html')
                        espera_quadro(page)
                        ok = page.evaluate("(W) => { const c = document.getElementById('cv'); return document.documentElement.scrollWidth <= W + 1 && (!c || (c.width > 10 && c.getContext('2d').getImageData(0,0,c.width,c.height).data.some(v => v))); }", vp[0])
                        if not ok:
                            nprob += 1
                            linhas.append(f'  problema (sozinho) {proto} {vp} {tema} {app}')
                    # dentro das guias
                    page.goto(f'{base}/index.html')
                    espera_quadro(page)
                    for app in APPS:
                        t = page.locator(f'#tab-{app}')
                        t.click()
                        page.wait_for_timeout(120)
                        fr = page.frame_locator(f'#frame-{app}')
                        ok = fr.locator('#say').count() == 1 and page.evaluate("(W) => document.documentElement.scrollWidth <= W + 1", vp[0]) and page.frame(name=None, url=re.compile(app)).evaluate("(W) => document.documentElement.scrollWidth <= W + 1", vp[0])
                        vis = t.is_visible()
                        if not ok or not vis:
                            nprob += 1
                            linhas.append(f'  problema (guias) {proto} {vp} {tema} {app} ok={ok} vis={vis}')
                    # tema chega nos apps
                    page.click('#theme')
                    page.wait_for_timeout(100)
                    fr = page.frame(url=re.compile('knn'))
                    tema_app = fr.evaluate("() => document.documentElement.getAttribute('data-theme')") if fr else None
                    tema_top = page.evaluate("() => document.documentElement.getAttribute('data-theme')")
                    if tema_app != tema_top:
                        nprob += 1
                        linhas.append(f'  tema não propagou {proto} {vp} {tema}: top={tema_top} app={tema_app}')
                    if toque:
                        # arraste por toque na floresta (CDP)
                        page.goto(f'{base}/floresta.html')
                        espera_quadro(page)
                        c = page.context.new_cdp_session(page)
                        pts = page.evaluate(LE_PONTOS)
                        box = page.locator('#plot').bounding_box()
                        f = box['width'] / 602
                        x0, y0 = box['x'] + pts[2]['cx'] * f, box['y'] + pts[2]['cy'] * f
                        antes = pts[2]
                        y0sc = page.evaluate('scrollY')
                        c.send('Input.dispatchTouchEvent', {'type': 'touchStart', 'touchPoints': [{'x': x0, 'y': y0}]})
                        for t in range(1, 6):
                            c.send('Input.dispatchTouchEvent', {'type': 'touchMove', 'touchPoints': [{'x': x0 + 8 * t, 'y': y0 + 6 * t}]})
                            page.wait_for_timeout(40)
                        c.send('Input.dispatchTouchEvent', {'type': 'touchEnd', 'touchPoints': []})
                        espera_quadro(page)
                        depois = page.evaluate(LE_PONTOS)[2]
                        moveu = abs(depois['cx'] - antes['cx']) > 5
                        rolou = page.evaluate('scrollY') != y0sc
                        if not moveu or rolou:
                            nprob += 1
                        linhas.append(f'  toque {proto} {tema}: arraste moveu={moveu} página rolou={rolou}')
                    if erros:
                        nprob += 1
                        linhas.append(f'  erros {proto} {vp} {tema}: {erros[:2]}')
                    ctx.close()
        linhas.append(f'contextos: {nprob} problemas')
        b.close()


# ------------------------------------------------------------------ desempenho
def desempenho(linhas):
    INST = """(() => {
      const raf = window.requestAnimationFrame.bind(window);
      window.__f = [];
      window.requestAnimationFrame = cb => raf(t => { const a = performance.now(); cb(t); window.__f.push(performance.now() - a); });
    })();"""
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in APPS:
            for n_alvo in [0, 60]:
                ctx = b.new_context(viewport={'width': 1280, 'height': 950})
                ctx.add_init_script(INST)
                page = ctx.new_page()
                page.goto(f'{BASE}/{app}.html')
                espera_quadro(page)
                if app == 'floresta':
                    pass
                box = page.locator('#plot').bounding_box()
                if n_alvo:
                    add = page.locator('[id^=t-add]:visible').first
                    add.click()
                    while page.evaluate("() => document.querySelectorAll('#layer-points .pt').length") < n_alvo:
                        page.mouse.click(box['x'] + box['width'] * random.uniform(.15, .9), box['y'] + box['height'] * random.uniform(.05, .85))
                    page.click('#t-move')
                    espera_quadro(page)
                n_real = page.evaluate("() => document.querySelectorAll('#layer-points .pt').length")
                page.evaluate('() => { window.__f = []; }')
                pts = page.evaluate(LE_PONTOS)
                arrasta(page, (pts[1]['cx'], pts[1]['cy']), (pts[1]['cx'] + 60, pts[1]['cy'] + 40))
                fr = page.evaluate('() => window.__f')
                fr = [v for v in fr if v > 0.05]
                linhas.append(f'{app} n={n_real}: {len(fr)} quadros; média {np.mean(fr) if fr else 0:.1f} ms; pior {max(fr) if fr else 0:.1f} ms' + ('' if fr else ' (desenha fora do requestAnimationFrame: não medido aqui)'))
                ctx.close()
            if app in NOVOS + ['svm-kernels']:
                ctx = b.new_context(viewport={'width': 1280, 'height': 950})
                ctx.add_init_script(INST)
                page = ctx.new_page()
                page.goto(f'{BASE}/{app}.html')
                espera_quadro(page)
                page.select_option('#dataset', 'mix2')
                page.check('#o-test')
                page.check('#o-bayes')
                espera_quadro(page)
                page.evaluate('() => { window.__f = []; }')
                pts = page.evaluate(LE_PONTOS)
                arrasta(page, (pts[1]['cx'], pts[1]['cy']), (pts[1]['cx'] + 60, pts[1]['cy'] + 40))
                fr = [v for v in page.evaluate('() => window.__f') if v > 0.05]
                linhas.append(f'{app} mistura n={len(pts)} + teste m={len(pts)}: {len(fr)} quadros; média {np.mean(fr) if fr else 0:.1f} ms; pior {max(fr) if fr else 0:.1f} ms' + ('' if fr else ' (desenha fora do requestAnimationFrame: não medido aqui)'))
                ctx.close()
            if app == 'floresta':
                ctx = b.new_context(viewport={'width': 1280, 'height': 950})
                ctx.add_init_script(INST)
                page = ctx.new_page()
                page.goto(f'{BASE}/floresta.html')
                espera_quadro(page)
                set_range(page, 'p-b', 500)
                espera_quadro(page)
                page.evaluate('() => { window.__f = []; }')
                pts = page.evaluate(LE_PONTOS)
                arrasta(page, (pts[1]['cx'], pts[1]['cy']), (pts[1]['cx'] + 60, pts[1]['cy'] + 40))
                fr = [v for v in page.evaluate('() => window.__f') if v > 0.05]
                linhas.append(f'floresta B=500 n={len(pts)}: {len(fr)} quadros; média {np.mean(fr) if fr else 0:.1f} ms; pior {max(fr) if fr else 0:.1f} ms' + ('' if fr else ' (desenha fora do requestAnimationFrame: não medido aqui)'))
                ctx.close()
        b.close()


# ------------------------------------------------------------------ capturas
def capturas(linhas):
    pasta = RES / 'capturas'
    pasta.mkdir(exist_ok=True)
    with sync_playwright() as p:
        b = novo_navegador(p)
        for tema in ['light', 'dark']:
            ctx = b.new_context(viewport={'width': 1366, 'height': 900}, color_scheme=tema)
            page = ctx.new_page()
            for app in APPS:
                page.goto(f'{BASE}/index.html#{app}')
                page.wait_for_timeout(250)
                page.screenshot(path=str(pasta / f'{app}_{tema}.png'))
            ctx.close()
        ctx = b.new_context(viewport={'width': 390, 'height': 800}, has_touch=True, is_mobile=True, color_scheme='light')
        page = ctx.new_page()
        for app in ['index#gnb', 'index#arvore', 'index#knn']:
            page.goto(f'{BASE}/{app.replace("#", ".html#")}')
            page.wait_for_timeout(250)
            page.screenshot(path=str(pasta / f'cel_{app.replace("#", "_")}.png'))
        for app in NOVOS + ['svm-kernels']:
            page.goto(f'{BASE}/{app}.html')
            page.wait_for_timeout(200)
            page.screenshot(path=str(pasta / f'cel_{app}.png'), full_page=True)
        ctx.close()
        # estados novos: mistura com ruído e teste, KNN com p = 1, nó da árvore em destaque
        for tema in ['light', 'dark']:
            ctx = b.new_context(viewport={'width': 1366, 'height': 900}, color_scheme=tema)
            page = ctx.new_page()
            for app in NOVOS + ['svm-kernels']:
                page.goto(f'{BASE}/{app}.html')
                espera_quadro(page)
                page.select_option('#dataset', 'mix2')
                page.fill('#p-noise-n', '0.3')
                page.check('#o-test')
                page.check('#o-bayes')
                if app == 'knn':
                    page.click('#pseg button[data-p="1"]')
                    page.click('#wseg button[data-w="distance"]')
                espera_quadro(page)
                if app == 'arvore':
                    page.locator('#v-tree .tn').nth(2).hover()
                elif app == 'knn':
                    box = page.locator('#plot').bounding_box()
                    page.mouse.move(box['x'] + box['width'] * .5, box['y'] + box['height'] * .45)
                espera_quadro(page)
                page.screenshot(path=str(pasta / f'novo_{app}_{tema}.png'))
            ctx.close()
        # mosaico de todos os conjuntos (KNN, K = 5), com e sem ruído
        from PIL import Image
        ctx = b.new_context(viewport={'width': 1366, 'height': 900}, color_scheme='light')
        page = ctx.new_page()
        page.goto(f'{BASE}/knn.html')
        espera_quadro(page)
        page.check('#o-test')
        ops = page.evaluate("() => [...document.getElementById('dataset').options].map(o => o.value)")
        for sigma in ('0', '0.3'):
            page.fill('#p-noise-n', sigma)
            ims = []
            for o in ops:
                page.select_option('#dataset', o)
                espera_quadro(page)
                arq = pasta / f'_m_{o}.png'
                page.locator('.plotwrap').screenshot(path=str(arq))
                ims.append(Image.open(arq).resize((300, 313)))
            cols = 5
            mos = Image.new('RGB', (300 * cols, 313 * ((len(ims) + cols - 1) // cols)), 'white')
            for i, im in enumerate(ims):
                mos.paste(im, (300 * (i % cols), 313 * (i // cols)))
            mos.save(pasta / f'mosaico_sigma{sigma}.png')
            for o in ops:
                (pasta / f'_m_{o}.png').unlink()
        ctx.close()
        b.close()
    linhas.append('capturas gravadas em ' + str(pasta))


def main():
    qual = sys.argv[1] if len(sys.argv) > 1 else 'todos'
    srv = servidor()
    linhas = []
    try:
        if qual in ('e2e', 'todos'):
            e2e(linhas)
        if qual in ('aleatorio', 'todos'):
            aleatorio(linhas)
        if qual in ('axe', 'todos'):
            axe(linhas)
        if qual in ('contextos', 'todos'):
            contextos(linhas)
        if qual in ('desempenho', 'todos'):
            desempenho(linhas)
        if qual in ('capturas', 'todos'):
            capturas(linhas)
    except Exception:
        linhas.append(traceback.format_exc())
    finally:
        srv.shutdown()
    relatorio('t_' + qual, linhas)


if __name__ == '__main__':
    main()
