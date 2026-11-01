# Quant

CPU-only quantitative research environment.

## Setup

Requires Linux and Python ≥ 3.10.

```bash
bash setup_quant.sh
```

This creates `~/quant` (override with `QUANT_HOME`) containing a virtualenv, a project skeleton, a Jupyter kernel and `requirements.lock`. Optional flags are documented at the top of `setup_quant.sh`.

## API keys

Set the keys as environment variables, or fill in `~/quant/.env`. Environment variables take precedence, and empty entries in `.env` are ignored.

| Service | Variables |
|---|---|
| Alpaca | `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `ALPACA_PAPER` |
| FRED | `FRED_API_KEY` |
| QuantConnect | `QC_USER_ID`, `QC_API_TOKEN` |
| Tushare (optional) | `TUSHARE_TOKEN` |

## Usage

```bash
source ~/quant/activate.sh
python ~/quant/scripts/check_apis.py --lean-login
jupyter lab --notebook-dir=~/quant/notebooks
```
