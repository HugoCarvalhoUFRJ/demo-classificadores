"""Verificação rápida durante o desenvolvimento: abre um app, exercita os controles novos, junta erros e tira capturas.
Uso: python t_rapido.py <app> [<app> ...]
"""
import sys

from playwright.sync_api import sync_playwright

from comum import RES
from t_nav import BASE, coleta_erros, espera_quadro, novo_navegador, servidor

ACOES = {
    'comum': [
        ("select", '#dataset', 'mix2'), ("check", '#o-test'), ("check", '#o-bayes'), ("shot", 'mix2'),
        ("fill", '#p-noise-n', '0.4'), ("shot", 'mix2-ruido'),
        ("select", '#dataset', 'figure'), ("shot", 'figure'),
        ("select", '#dataset', 'mix3'), ("shot", 'mix3'),
        ("fill", '#p-noise-n', '0'),
        ("select", '#dataset', 'moons'), ("shot", 'moons'),
    ],
    'knn': [("click", '#pseg button[data-p="1"]'), ("click", '#wseg button[data-w="distance"]'), ("hover", (300, 300)), ("shot", 'knn-p1'),
            ("click", '#pseg button[data-p="3"]'), ("hover", (250, 250)), ("shot", 'knn-p3')],
    'arvore': [("hovernode", 3), ("shot", 'no3'), ("hovernode", 6), ("shot", 'no6')],
    'floresta': [("fill", '#p-leaf-n', '4'), ("shot", 'folha4')],
}


def main(apps):
    srv = servidor()
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in apps:
            pg = b.new_page(viewport={'width': 1300, 'height': 900})
            erros = []
            coleta_erros(pg, erros)
            pg.goto(f'{BASE}/{app}.html')
            espera_quadro(pg)
            for a in ACOES['comum'] + ACOES.get(app, []):
                kind, alvo = a[0], a[1]
                if kind == 'select':
                    if not pg.evaluate(f"() => !!document.querySelector('{alvo} option[value=\"{a[2]}\"]')"):
                        continue
                    pg.select_option(alvo, a[2])
                elif kind == 'check':
                    pg.check(alvo)
                elif kind == 'click':
                    pg.click(alvo)
                elif kind == 'fill':
                    pg.fill(alvo, a[2])
                elif kind == 'hover':
                    box = pg.locator('#plot').bounding_box()
                    pg.mouse.move(box['x'] + alvo[0] * box['width'] / 602, box['y'] + alvo[1] * box['width'] / 602)
                elif kind == 'hovernode':
                    pg.locator('#v-tree .tn').nth(alvo).hover()
                espera_quadro(pg)
                if kind == 'shot':
                    pg.screenshot(path=str(RES / 'capturas' / f'rapido_{app}_{alvo}.png'))
            txt = pg.evaluate("() => document.getElementById('v-data').innerText + ' | ' + document.getElementById('say').innerText")
            print(app, 'erros:', erros, '|', txt)
            pg.close()
        b.close()
    srv.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:] or ['knn'])
