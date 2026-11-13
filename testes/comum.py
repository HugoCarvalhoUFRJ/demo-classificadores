"""Roda o solver de cada app (bloco /*SOLVER-START*/ ... /*SOLVER-END*/) no Node que vem com o Playwright."""
import json
import os
import pathlib
import subprocess

PROJ = pathlib.Path(__file__).resolve().parent.parent
T = pathlib.Path(os.environ.get('TW', r'C:\Users\hugom\AppData\Local\classificadores-testes'))
NODE = T / 'venv' / 'Lib' / 'site-packages' / 'playwright' / 'driver' / 'node.exe'
TMP = T / 'tmp'
RES = T / 'resultados'
RES.mkdir(exist_ok=True)

RUNNER = r"""
const fs = require('fs');
const [,, html, inFile, outFile] = process.argv;
const src = fs.readFileSync(html, 'utf8');
const m = src.match(/\/\*SOLVER-START\*\/([\s\S]*?)\/\*SOLVER-END\*\//);
const Solver = new Function(m[1] + '\nreturn Solver;')();
const job = JSON.parse(fs.readFileSync(inFile, 'utf8'));
const fn = new Function('Solver', 'caso', job.body);
const t0 = process.hrtime.bigint();
const out = job.casos.map(c => { try { return fn(Solver, c); } catch (e) { return { erro: String(e && e.stack || e) }; } });
const ms = Number(process.hrtime.bigint() - t0) / 1e6;
fs.writeFileSync(outFile, JSON.stringify({ ms, out }));
"""


def roda_js(app, body, casos, nome=None):
    """Executa `body` (corpo de função JS com Solver e caso) para cada caso; devolve (saídas, ms)."""
    nome = nome or app
    TMP.mkdir(exist_ok=True)
    runner = TMP / 'runner.js'
    runner.write_text(RUNNER, encoding='utf-8')
    fin, fout = TMP / f'{nome}.in.json', TMP / f'{nome}.out.json'
    fin.write_text(json.dumps({'body': body, 'casos': casos}), encoding='utf-8')
    r = subprocess.run([str(NODE), str(runner), str(PROJ / f'{app}.html'), str(fin), str(fout)], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    d = json.loads(fout.read_text(encoding='utf-8'))
    return d['out'], d['ms']


def relatorio(nome, linhas):
    texto = '\n'.join(linhas)
    print(texto)
    (RES / f'{nome}.txt').write_text(texto + '\n', encoding='utf-8')
