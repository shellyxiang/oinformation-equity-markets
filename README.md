# Beyond Mean Correlation: O-Information and the Organization of Dependence in International Equity Markets

Code and derived results for the paper of the same title (submitted to *Entropy*).

The analysis applies Gaussian-copula O-information to daily returns of 20 international equity
indices (2005–2025): system-level redundancy across system size, deviations from an equicorrelated
benchmark, regional systems, and local (daily) O-information across VIX states.

## Repository layout

| Folder | Contents |
|---|---|
| `code/` | All scripts used for the results, tables and figures of the paper |
| `run_all.sh` | Runs the full pipeline in dependency order |
| `raw_audit/` | Derived results (JSON/CSV/NPZ) and run logs written by the scripts |
| `figures/` | Figures 1–5 (PDF and PNG) |
| `data_raw/` | Place the raw index files here (not distributed, see below) |
| `data_external/` | Place the VIX file here (not distributed, see below) |
| `processed/` | Return panel built by `code/build_returns.py` (not distributed) |

## Data

**Equity indices.** Daily closing levels were downloaded from Investing.com ("Historical Data",
daily frequency, 2005-01-01 to 2025-12-31). Investing.com does not permit redistribution, so the
raw files are not included. Save each export under the file name below in `data_raw/`:

| Code | Index | File name |
|---|---|---|
| AEX | AEX (Netherlands) | `AEX Historical Data.csv` |
| CAC40 | CAC 40 (France) | `CAC 40 Historical Data.csv` |
| DAX | DAX (Germany) | `DAX Historical Data.csv` |
| FTSE100 | FTSE 100 (United Kingdom) | `FTSE 100 Historical Data.csv` |
| FTSEMIB | FTSE MIB (Italy) | `FTSE MIB Historical Data.csv` |
| IBEX35 | IBEX 35 (Spain) | `IBEX 35 Historical Data.csv` |
| OMXS30 | OMX Stockholm 30 (Sweden) | `OMX Stockholm 30 Historical Data.csv` |
| OBX | Oslo OBX (Norway) | `Oslo OBX Historical Data.csv` |
| SMI | SMI (Switzerland) | `SMI Historical Data.csv` |
| HangSeng | Hang Seng (Hong Kong) | `Hang Seng Historical Data (4).csv` |
| LQ45 | Jakarta LQ45 (Indonesia) | `Jakarta Stock Exchange LQ45 Historical Data (1).csv` |
| KOSPI200 | KOSPI 200 (South Korea) | `KOSPI 200 Historical Data_1.csv` |
| Nifty50 | Nifty 50 (India) | `Nifty 50 Historical Data (3).csv` |
| ASX200 | S&P/ASX 200 (Australia) | `S&P_ASX 200 Historical Data (1).csv` |
| CSI300 | CSI 300 (China) | `Shanghai Shenzhen CSI 300 Historical Data (2).csv` |
| TWSE | Taiwan Weighted (Taiwan) | `Taiwan Weighted Historical Data.csv` |
| SP500 | S&P 500 (United States) | `S&P 500 Historical Data (1).csv` |
| IPC | S&P/BMV IPC (Mexico) | `S&P_BMV IPC Historical Data (1).csv` |
| TSX60 | S&P/TSX 60 (Canada) | `S&P_TSX 60 Historical Data.csv` |
| SATop40 | South Africa Top 40 | `South Africa Top 40 Historical Data.csv` |

`code/build_returns.py` builds the complete-case panel (3,793 common trading dates, 3,792 daily
log returns, 2005-01-05 to 2025-12-30). Cleaning rules are documented in the script header.

**VIX.** CBOE VIX daily history, saved as `data_external/VIX_History_CBOE.csv`
(columns `DATE, OPEN, HIGH, LOW, CLOSE`), available at
<https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv>.

## Reproducing the results

Python 3.12 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
bash run_all.sh
```

`run_all.sh` lists which script produces which part of the paper. The bootstrap (B = 999) and
surrogate steps use fixed seeds; the full pipeline takes roughly one hour on a desktop CPU.
The derived results in `raw_audit/` and the figures in `figures/` are the outputs used in the paper,
so they can be inspected without rerunning anything.

## License

Code: MIT (see `LICENSE`).
