# Fonte: source testes/env.sh
# Tudo que os testes instalam ou gravam fica em $T. Nada toca o conda base, a biblioteca do R ou o perfil do Chrome.
export T="/c/Users/hugom/AppData/Local/classificadores-testes"
export TW='C:\Users\hugom\AppData\Local\classificadores-testes'
export TEMP="$TW\tmp" TMP="$TW\tmp" TMPDIR="$TW\tmp"
export PIP_CACHE_DIR="$TW\pipcache" R_LIBS_USER="$TW\Rlib"
export PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export PLAYWRIGHT_BROWSERS_PATH="$TW\pw-browsers" PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
export CONDA=/c/Users/hugom/anaconda3
# DLLs do conda base (numpy/scipy/sklearn) só nesta sessão do shell
export PATH="$T/venv/Scripts:$CONDA:$CONDA/Library/bin:$CONDA/Library/mingw-w64/bin:$CONDA/Library/usr/bin:$PATH"
export PY="$T/venv/Scripts/python.exe"
export RS="/c/Program Files/R/R-4.5.1/bin/Rscript.exe"
export PROJ="/c/Users/hugom/OneDrive/Página/Demo Classificadores"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
# Evita que o loky (scikit-learn) chame o wmic para contar núcleos: a saída em cp850 gerava um UnicodeDecodeError inofensivo
export LOKY_MAX_CPU_COUNT=4
