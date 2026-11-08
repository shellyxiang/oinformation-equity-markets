"""Build the 20-market complete-case daily log-return panel from Investing.com exports.

Input : data_raw/<file>.csv  (Investing.com "Historical Data" exports; not redistributed)
Output: processed/returns_daily_20markets.csv

Rules: parse Date and Price (thousands separators removed); drop rows without a valid date;
keep 2005-01-01..2025-12-31; for duplicated dates keep the first occurrence after sorting;
keep dates on which all 20 markets have a valid price (complete case); log returns are
differences of log prices between consecutive complete-case dates.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(ROOT, "processed", "returns_daily_20markets.csv")

FILES = {
    "AEX": "AEX Historical Data.csv",
    "CAC40": "CAC 40 Historical Data.csv",
    "DAX": "DAX Historical Data.csv",
    "FTSE100": "FTSE 100 Historical Data.csv",
    "FTSEMIB": "FTSE MIB Historical Data.csv",
    "HangSeng": "Hang Seng Historical Data (4).csv",
    "IBEX35": "IBEX 35 Historical Data.csv",
    "LQ45": "Jakarta Stock Exchange LQ45 Historical Data (1).csv",
    "KOSPI200": "KOSPI 200 Historical Data_1.csv",
    "Nifty50": "Nifty 50 Historical Data (3).csv",
    "OMXS30": "OMX Stockholm 30 Historical Data.csv",
    "OBX": "Oslo OBX Historical Data.csv",
    "SP500": "S&P 500 Historical Data (1).csv",
    "ASX200": "S&P_ASX 200 Historical Data (1).csv",
    "IPC": "S&P_BMV IPC Historical Data (1).csv",
    "TSX60": "S&P_TSX 60 Historical Data.csv",
    "SMI": "SMI Historical Data.csv",
    "CSI300": "Shanghai Shenzhen CSI 300 Historical Data (2).csv",
    "SATop40": "South Africa Top 40 Historical Data.csv",
    "TWSE": "Taiwan Weighted Historical Data.csv",
}


def load_market(path: str) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    date = pd.to_datetime(df["Date"].str.replace("/", "-", regex=False), format="%Y-%m-%d", errors="coerce")
    price = pd.to_numeric(df["Price"].str.replace(",", "", regex=False), errors="coerce")
    s = pd.DataFrame({"Date": date, "Price": price}).dropna(subset=["Date"])
    s = s[(s["Date"] >= "2005-01-01") & (s["Date"] <= "2025-12-31")]
    s = s.sort_values("Date", kind="mergesort").drop_duplicates("Date", keep="first")
    return s.set_index("Date")["Price"]


def main() -> None:
    prices = pd.concat({m: load_market(os.path.join(RAW, f)) for m, f in FILES.items()}, axis=1)
    prices = prices.sort_index().dropna(how="any")
    rets = np.log(prices).diff().iloc[1:]
    rets.index.name = "Date"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rets.to_csv(OUT, date_format="%Y-%m-%d")
    print(f"{len(prices)} complete-case dates, {len(rets)} returns -> {OUT}")


if __name__ == "__main__":
    main()
