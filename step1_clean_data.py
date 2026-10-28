"""
Step 1: Clean the LTE site traffic exports (voice traffic).

Input : the two Excel exports (stell_1.xlsx, stell_2.xlsx)
Output: one clean CSV per site plus a combined CSV, in the output folder.

Each export has two sheets ("0_Data Traffic", "1_Voice Traffic"). Each sheet has
two title rows, a header row (DATETIME, value) and a report footer. Only the
voice sheet ("LTE Erl") is used. The data volume column is skipped because its
units are not stated in the export.

Columns in the clean CSV:
    datetime   hourly timestamp (start of hour)
    site       short site name
    voice_erl  carried LTE voice traffic in Erlang (NaN where the export is empty)
    weekday    0 = Monday ... 6 = Sunday
    hour       0 ... 23
    is_weekend True on Saturday and Sunday
    split      "calibration" (before SPLIT_DATE) or "test" (from SPLIT_DATE)

Usage:
    python step1_clean_data.py stell_1.xlsx stell_2.xlsx --out data_clean
"""
import argparse
from pathlib import Path

import pandas as pd

VOICE_SHEET = "1_Voice Traffic"
SPLIT_DATE = pd.Timestamp("2026-09-20")  # calibration: 2 Aug - 19 Sep, test: 20 Sep - 8 Oct

SITE_NAMES = {
    "WES_34941": "Hoerskool",
    "WES_11201": "Square",
}


def site_from_title(title: str) -> tuple[str, str]:
    """Return (site_id, short_name) from the sheet title row."""
    for site_id, name in SITE_NAMES.items():
        if site_id in title:
            return site_id, name
    return "unknown", "unknown"


def clean_voice_sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=VOICE_SHEET, header=None)
    site_id, site = site_from_title(str(raw.iloc[0, 0]))

    # Header row is the one whose first cell is "DATETIME"
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "DATETIME"][0]
    df = raw.iloc[header_row + 1:, :2].copy()
    df.columns = ["datetime", "voice_erl"]

    # Footer rows do not parse as dates, so they drop out here
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df["voice_erl"] = pd.to_numeric(df["voice_erl"], errors="coerce")

    df = df.drop_duplicates(subset="datetime").sort_values("datetime")

    # Reindex to a full hourly range so any missing hours show up as NaN
    full = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full).rename_axis("datetime").reset_index()

    df.insert(1, "site", site)
    df.insert(2, "site_id", site_id)
    df["weekday"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["is_weekend"] = df["weekday"] >= 5
    df["split"] = (df["datetime"] >= SPLIT_DATE).map({False: "calibration", True: "test"})
    return df


def summarise(df: pd.DataFrame) -> None:
    site = df["site"].iloc[0]
    missing = df[df["voice_erl"].isna()]
    print(f"\n{site} ({df['site_id'].iloc[0]})")
    print(f"  period        : {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  hours         : {len(df)}")
    print(f"  empty values  : {len(missing)}")
    for t in missing["datetime"]:
        print(f"                  {t:%a %Y-%m-%d %H:%M}")
    print(f"  zero hours    : {(df['voice_erl'] == 0).sum()} (kept, valid quiet hours)")
    print(f"  negative      : {(df['voice_erl'] < 0).sum()}")
    for split, g in df.groupby("split"):
        print(f"  {split:<12}: {len(g)} hours, mean {g['voice_erl'].mean():.2f} E, max {g['voice_erl'].max():.2f} E")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data_clean"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in args.files:
        df = clean_voice_sheet(f)
        summarise(df)
        df.to_csv(args.out / f"voice_{df['site'].iloc[0].lower()}.csv", index=False)
        frames.append(df)

    pd.concat(frames).to_csv(args.out / "voice_all_sites.csv", index=False)
    print(f"\nSaved clean CSVs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
