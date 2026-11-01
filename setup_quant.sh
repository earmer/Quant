#!/usr/bin/env bash
# =============================================================================
# setup_quant.sh — CPU-only quantitative research environment
#
# Assumes: a Linux machine with python3 (>= 3.10) already installed.
# Creates:  $QUANT_HOME with an isolated virtualenv, project skeleton,
#           Jupyter kernel, lockfile and a smoke test.
#
# Usage:
#   bash setup_quant.sh
#   CN_MIRROR=1 WITH_TORCH=0 bash setup_quant.sh      # example with flags
#
# Flags (environment variables, all optional):
#   QUANT_HOME     install location                   (default: ~/quant)
#   PYTHON         interpreter to build the venv from (default: python3)
#   CN_MIRROR      1 = use TUNA PyPI mirror           (default: 0)
#   SKIP_SYSTEM    1 = skip OS package installation   (default: 0)
#   WITH_TORCH     1 = install CPU-only PyTorch       (default: 1)
#   WITH_BACKTEST  1 = install backtesting libraries  (default: 1)
#   WITH_CN_DATA   1 = install A-share data clients   (default: 1)
#   WITH_TALIB     1 = install TA-Lib bindings        (default: 1)
# =============================================================================
set -Eeuo pipefail
trap 'echo "[error] line $LINENO: $BASH_COMMAND" >&2' ERR

QUANT_HOME="${QUANT_HOME:-$HOME/quant}"
PYTHON="${PYTHON:-python3}"
CN_MIRROR="${CN_MIRROR:-0}"
SKIP_SYSTEM="${SKIP_SYSTEM:-0}"
WITH_TORCH="${WITH_TORCH:-1}"
WITH_BACKTEST="${WITH_BACKTEST:-1}"
WITH_CN_DATA="${WITH_CN_DATA:-1}"
WITH_TALIB="${WITH_TALIB:-1}"

TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
FAILED=()

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 0. Pre-flight: check the existing Python
# -----------------------------------------------------------------------------
command -v "$PYTHON" >/dev/null || die "$PYTHON not found; set PYTHON=/path/to/python3"
"$PYTHON" - <<'EOF' || die "Python >= 3.10 required (several quant libraries have dropped older versions)"
import sys
print(f"[setup] Found Python {sys.version.split()[0]} at {sys.executable}")
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

if [[ $EUID -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

# -----------------------------------------------------------------------------
# 1. OS packages: compilers (for any package without a wheel) and CLI tools
# -----------------------------------------------------------------------------
install_system() {
  if [[ -n "$SUDO" ]] && ! command -v sudo >/dev/null; then
    warn "sudo unavailable; skipping OS packages"; return
  fi
  if command -v apt-get >/dev/null; then
    $SUDO apt-get update -y
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
      build-essential pkg-config git curl wget ca-certificates \
      tmux htop unzip zip jq sqlite3
  elif command -v dnf >/dev/null; then
    $SUDO dnf install -y gcc gcc-c++ make pkgconf-pkg-config git curl wget \
      ca-certificates tmux htop unzip zip jq sqlite
  elif command -v pacman >/dev/null; then
    $SUDO pacman -Sy --noconfirm --needed base-devel pkgconf git curl wget \
      ca-certificates tmux htop unzip zip jq sqlite
  elif command -v zypper >/dev/null; then
    $SUDO zypper --non-interactive install -t pattern devel_basis
    $SUDO zypper --non-interactive install git curl wget tmux htop unzip zip jq sqlite3
  else
    warn "Unrecognised package manager; install a C/C++ toolchain and git manually"
  fi
}

if [[ "$SKIP_SYSTEM" == 1 ]]; then
  log "Skipping OS packages (SKIP_SYSTEM=1)"
else
  log "Installing OS packages"
  install_system
fi

# -----------------------------------------------------------------------------
# 2. Package index (optional mainland China mirror)
# -----------------------------------------------------------------------------
if [[ "$CN_MIRROR" == 1 ]]; then
  log "Using TUNA PyPI mirror"
  export UV_DEFAULT_INDEX="$TUNA"
  export PIP_INDEX_URL="$TUNA"
  mkdir -p "$HOME/.config/uv"
  cat > "$HOME/.config/uv/uv.toml" <<EOF
[[index]]
url = "$TUNA"
default = true
EOF
fi

# -----------------------------------------------------------------------------
# 3. uv (fast resolver/installer; does not need python3-venv or system pip)
# -----------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "Installing uv"
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    warn "uv installer unreachable; trying pip"
    "$PYTHON" -m pip install --user uv \
      || "$PYTHON" -m pip install --user --break-system-packages uv \
      || die "Could not install uv (no network to astral.sh and no working pip)"
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv installed but not on PATH"
log "uv $(uv --version | awk '{print $2}')"

# -----------------------------------------------------------------------------
# 4. Project skeleton and virtualenv built from the existing Python
# -----------------------------------------------------------------------------
log "Creating project at $QUANT_HOME"
mkdir -p "$QUANT_HOME"/{data/{raw,interim,processed},notebooks,src/quant,research,backtests,reports,logs}
cd "$QUANT_HOME"

if [[ ! -x .venv/bin/python ]]; then
  uv venv --python "$(command -v "$PYTHON")" .venv
fi
VPY="$QUANT_HOME/.venv/bin/python"

# Install a group in one resolve; on conflict, fall back to one-by-one so a
# single awkward package cannot sink the rest.
pip_group() {
  local name=$1; shift
  log "Group: $name"
  if uv pip install --python "$VPY" "$@"; then return 0; fi
  warn "Group '$name' failed as a whole; retrying individually"
  local p
  for p in "$@"; do
    uv pip install --python "$VPY" "$p" || FAILED+=("$p")
  done
}

# -----------------------------------------------------------------------------
# 5. Python packages
# -----------------------------------------------------------------------------
pip_group "core numerics & storage" \
  numpy pandas polars pyarrow scipy numba duckdb tables

pip_group "statistics & econometrics" \
  statsmodels arch linearmodels

pip_group "machine learning (CPU)" \
  scikit-learn lightgbm xgboost catboost optuna

pip_group "portfolio construction & pricing" \
  cvxpy PyPortfolioOpt riskfolio-lib QuantLib

pip_group "market data (global)" \
  yfinance ccxt fredapi requests

# Alpaca (broker + market data), QuantConnect (Lean CLI + IDE stubs)
pip_group "broker & platform APIs" \
  alpaca-py lean quantconnect-stubs

if [[ "$WITH_CN_DATA" == 1 ]]; then
  pip_group "market data (A-shares)" akshare tushare baostock
fi

pip_group "notebooks & visualisation" \
  jupyterlab ipykernel ipywidgets matplotlib seaborn plotly

pip_group "tooling" \
  ruff pytest mypy rich tqdm loguru python-dotenv joblib pydantic

if [[ "$WITH_BACKTEST" == 1 ]]; then
  # Installed late: these pin older numba/numpy ranges more often than most.
  pip_group "backtesting & performance analytics" \
    vectorbt backtrader quantstats empyrical-reloaded pyfolio-reloaded
fi

if [[ "$WITH_TALIB" == 1 ]]; then
  # Recent PyPI releases ship wheels with the C library bundled.
  pip_group "technical indicators" TA-Lib
fi

if [[ "$WITH_TORCH" == 1 ]]; then
  log "Group: PyTorch (CPU build)"
  uv pip install --python "$VPY" torch \
      --index-url https://download.pytorch.org/whl/cpu \
    || FAILED+=("torch (cpu)")
fi

# -----------------------------------------------------------------------------
# 6. Jupyter kernel, activation helper, git scaffolding
# -----------------------------------------------------------------------------
log "Registering Jupyter kernel 'quant'"
"$VPY" -m ipykernel install --user --name quant --display-name "Python (quant)" \
  || FAILED+=("ipykernel registration")

cat > "$QUANT_HOME/activate.sh" <<EOF
# source ~/quant/activate.sh   (bash)
source "$QUANT_HOME/.venv/bin/activate"
export QUANT_HOME="$QUANT_HOME"
export QUANT_DATA="$QUANT_HOME/data"
# Avoid oversubscription when mixing joblib/multiprocessing with BLAS threads;
# raise these for single-process heavy linear algebra.
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=\${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=\${MKL_NUM_THREADS:-1}
export NUMBA_CACHE_DIR="$QUANT_HOME/.numba_cache"
# Load .env without clobbering: empty values are ignored and variables already
# set in the environment (e.g. secrets injected by the host) take precedence.
if [ -f "$QUANT_HOME/.env" ]; then
  while IFS='=' read -r k v || [ -n "\$k" ]; do
    [[ \$k =~ ^[A-Za-z_][A-Za-z0-9_]*\$ && -n \$v && -z \${!k:-} ]] && export "\$k=\$v"
  done < "$QUANT_HOME/.env"
fi
true
EOF

[[ -f .env ]] || cat > .env <<'EOF'
# API keys — never commit this file. Format: NAME=value (no quotes, no spaces)
# Values already set in the environment take precedence over this file.

# Alpaca (Alpaca's standard variable names)
APCA_API_KEY_ID=
APCA_API_SECRET_KEY=
# true for paper keys (PK...), false for live keys (AK...)
ALPACA_PAPER=true

# FRED (fredapi reads this name automatically)
FRED_API_KEY=

# QuantConnect: Account page -> User ID and API token
QC_USER_ID=
QC_API_TOKEN=

# Tushare (optional, A-shares)
TUSHARE_TOKEN=
EOF
chmod 600 .env

mkdir -p scripts
cat > scripts/check_apis.py <<'EOF'
"""Verify the API keys in $QUANT_HOME/.env. Never prints secret values.

    python scripts/check_apis.py               # check all
    python scripts/check_apis.py --lean-login  # also log the Lean CLI in
"""
import argparse, base64, hashlib, os, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(os.environ.get("QUANT_HOME") or Path(__file__).resolve().parents[1])
load_dotenv(ROOT / ".env")


def env(*names):
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return ""


def check_fred():
    key = env("FRED_API_KEY", "FRED_API")
    if not key:
        return "skip", "FRED_API_KEY not set"
    from fredapi import Fred
    s = Fred(api_key=key).get_series("DGS10").dropna()
    return "ok", f"DGS10 (10y Treasury) {s.iloc[-1]:.2f}% on {s.index[-1].date()}"


def check_alpaca():
    key = env("APCA_API_KEY_ID", "ALPACA_KEY")
    secret = env("APCA_API_SECRET_KEY", "ALPACA_SECRET", "ALPACA_SERECT")
    if not (key and secret):
        return "skip", "APCA_API_KEY_ID / APCA_API_SECRET_KEY not set"
    paper = env("ALPACA_PAPER").lower() not in ("false", "0", "no")
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    try:
        acct = TradingClient(key, secret, paper=paper).get_account()
    except Exception as e:
        hint = " (PK... keys need ALPACA_PAPER=true, AK... keys need false)"
        return "fail", f"trading API: {type(e).__name__}: {str(e).replace(chr(10), ' ').strip()[:80]}{hint}"
    bars = StockHistoricalDataClient(key, secret).get_stock_bars(StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame.Day,
        start=datetime.now(timezone.utc) - timedelta(days=10),
        feed=DataFeed.IEX,  # free plan; SIP needs a paid subscription
    )).df
    return "ok", (f"{'paper' if paper else 'LIVE'} account, status {acct.status}; "
                  f"SPY daily bars (IEX): {len(bars)}")


def check_quantconnect():
    uid = env("QC_USER_ID", "QUANTCONNECT_USERID")
    tok = env("QC_API_TOKEN", "QUANTCONNECT_KEY")
    if not (uid and tok):
        return "skip", "QC_USER_ID / QC_API_TOKEN not set"
    # Documented scheme: Basic base64(uid:sha256(token:ts)) + Timestamp header
    ts = str(int(time.time()))
    h = hashlib.sha256(f"{tok}:{ts}".encode()).hexdigest()
    auth = base64.b64encode(f"{uid}:{h}".encode()).decode()
    r = requests.post("https://www.quantconnect.com/api/v2/authenticate",
                      headers={"Authorization": f"Basic {auth}", "Timestamp": ts},
                      json={}, timeout=20)
    if r.ok and r.json().get("success"):
        return "ok", "REST API authenticated"
    return "fail", f"HTTP {r.status_code}: {r.text[:100]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean-login", action="store_true",
                    help="store QuantConnect credentials for the Lean CLI")
    args = ap.parse_args()

    results = {}
    for name, fn in [("FRED", check_fred), ("Alpaca", check_alpaca),
                     ("QuantConnect", check_quantconnect)]:
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = ("fail", f"{type(e).__name__}: {str(e).replace(chr(10), ' ').strip()[:120]}")
        status, msg = results[name]
        print(f"  {name:<13} {status.upper():<5} {msg}")

    if args.lean_login:
        if results["QuantConnect"][0] != "ok":
            print("  Lean login skipped: QuantConnect check did not pass")
        else:
            subprocess.run(["lean", "login",
                            "-u", env("QC_USER_ID", "QUANTCONNECT_USERID"),
                            "-t", env("QC_API_TOKEN", "QUANTCONNECT_KEY")],
                           check=True, stdout=subprocess.DEVNULL)
            print("  Lean CLI logged in (credentials in ~/.lean/credentials)")

    sys.exit(1 if any(s == "fail" for s, _ in results.values()) else 0)


if __name__ == "__main__":
    main()
EOF

if ! command -v docker >/dev/null; then
  warn "Docker not found: 'lean backtest' (local) needs it; 'lean cloud backtest' does not"
fi

[[ -f .gitignore ]] || cat > .gitignore <<'EOF'
.venv/
.env
data/
logs/
.numba_cache/
__pycache__/
.ipynb_checkpoints/
*.parquet
*.h5
*.duckdb
EOF

touch src/quant/__init__.py
if command -v git >/dev/null && [[ ! -d .git ]]; then
  git init -q && log "Initialised git repository"
fi

# -----------------------------------------------------------------------------
# 7. Lockfile and smoke test
# -----------------------------------------------------------------------------
uv pip freeze --python "$VPY" > requirements.lock
log "Pinned $(wc -l < requirements.lock) packages to requirements.lock"

log "Smoke test"
"$VPY" - <<'EOF'
import importlib, os
mods = ["numpy", "pandas", "polars", "pyarrow", "scipy", "numba", "duckdb",
        "statsmodels", "sklearn", "lightgbm", "xgboost", "cvxpy",
        "QuantLib", "yfinance", "fredapi", "alpaca", "akshare", "vectorbt",
        "talib", "torch"]
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "ok")
        print(f"  {m:<12} {v}")
    except Exception as e:
        print(f"  {m:<12} -- {type(e).__name__}: {str(e).replace(chr(10), ' ').strip()[:60]}")

import numpy as np
from numba import njit
@njit(cache=False)
def ewma(x, a):
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, x.size):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out
print("  numba JIT   ", round(float(ewma(np.random.randn(10_000), 0.1)[-1]), 4))
print(f"  CPU cores    {os.cpu_count()}")
EOF

# -----------------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------------
echo
if ((${#FAILED[@]})); then
  warn "Finished with failures: ${FAILED[*]}"
  warn "Retry individually: uv pip install --python $VPY <package>"
else
  log "All groups installed cleanly"
fi
log "Activate with:  source $QUANT_HOME/activate.sh"
log "Launch Jupyter: jupyter lab --notebook-dir=$QUANT_HOME/notebooks"
log "Next: fill in $QUANT_HOME/.env (or set the keys as environment variables), then run"
log "      python scripts/check_apis.py --lean-login"
