"""A casca comum dos seis apps multiclasse tem de ser idêntica: CSS, barra de dados e o script da casca.

Uso: python t_casca.py            confere (relatório em resultados/t_casca.txt)
     python t_casca.py aplicar    copia a casca do knn.html para os outros cinco (o knn.html é a referência)
"""
import difflib
import re
import sys

from comum import PROJ, relatorio

APPS = ['gnb', 'lda', 'qda', 'arvore', 'floresta', 'knn']
REF = 'knn'


def regioes(s):
    """Trechos da casca: (início, fim) de cada um no texto."""
    out = {}
    m = re.search(r'<style>\n/\* Layout.*?</style>', s, re.S)
    out['css'] = m.span()
    a = s.index('    <div class="bar">')
    b = s.index('\n    <div class="params">', a)
    out['barra'] = (a, b)
    a = s.index('(() => {\n  const X0 = -1.73')
    out['script'] = (a, len(s))
    # pedaços pequenos que precisam existir em todos
    for nome, pad in [('svg', r'<svg id="plot".*?</svg>'), ('vdata', r'<p class="note" id="v-data"></p>')]:
        m = re.search(pad, s, re.S)
        out[nome] = m.span() if m else None
    return out


def confere():
    linhas = []
    ref = (PROJ / f'{REF}.html').read_text(encoding='utf-8')
    R = regioes(ref)
    falhas = 0
    for app in APPS:
        s = (PROJ / f'{app}.html').read_text(encoding='utf-8')
        r = regioes(s)
        for k in ('css', 'barra', 'script'):
            a = ref[R[k][0]:R[k][1]]
            b = s[r[k][0]:r[k][1]]
            if a != b:
                falhas += 1
                d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='', n=0))
                linhas.append(f'{app}: {k} difere ({len(d)} linhas de diff)')
                linhas += ['    ' + x for x in d[:12]]
        if r['vdata'] is None:
            falhas += 1
            linhas.append(f'{app}: falta <p id="v-data">')
        svg = s[r['svg'][0]:r['svg'][1]]
        for pedaco in ('id="bayes"', 'id="probe-p"', 'id="layer-test"'):
            if pedaco not in svg:
                falhas += 1
                linhas.append(f'{app}: SVG sem {pedaco}')
        if 'predict' not in s[s.index('const App = {'):s.index('(() => {\n  const X0 = -1.73')]:
            falhas += 1
            linhas.append(f'{app}: App sem predict (usado pelo conjunto de teste)')
    linhas.insert(0, f'casca comum dos {len(APPS)} apps (referência {REF}.html): {falhas} diferenças')
    relatorio('t_casca', linhas)
    return falhas


SVG_TROCAS = [
    ('          <g id="layer-extra"></g>\n          <path id="ghost"', '          <g id="layer-extra"></g>\n          <path id="bayes" class="curve bayes"></path>\n          <path id="ghost"'),
    ('          <circle id="probe-c" class="probe" style="display:none"></circle>\n        </g>', '          <circle id="probe-c" class="probe" style="display:none"></circle>\n          <path id="probe-p" class="probe" style="display:none"></path>\n        </g>'),
    ('        <g id="axes"></g>\n        <g id="layer-points"></g>', '        <g id="axes"></g>\n        <g id="layer-test"></g>\n        <g id="layer-points"></g>'),
]


def aplica():
    ref = (PROJ / f'{REF}.html').read_text(encoding='utf-8')
    R = regioes(ref)
    for app in APPS:
        if app == REF:
            continue
        f = PROJ / f'{app}.html'
        s = f.read_text(encoding='utf-8')
        # de trás para a frente, para os índices continuarem valendo
        for k in ('script', 'barra', 'css'):
            r = regioes(s)[k]
            s = s[:r[0]] + ref[R[k][0]:R[k][1]] + s[r[1]:]
        for a, b in SVG_TROCAS:
            if b not in s:
                assert s.count(a) == 1, (app, a)
                s = s.replace(a, b)
        if 'id="v-data"' not in s:
            a = '        <span id="pill" class="pill"></span>\n      </div>\n'
            assert s.count(a) == 1, app
            s = s.replace(a, a + '      <p class="note" id="v-data"></p>\n')
        f.write_text(s, encoding='utf-8')
        print('aplicado em', app)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'aplicar':
        aplica()
    sys.exit(1 if confere() else 0)
