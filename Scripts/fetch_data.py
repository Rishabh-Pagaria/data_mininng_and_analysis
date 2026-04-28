"""
fetch_data.py
-------------
Downloads the CDC 500 Cities: Local Data for Better Health (2019 Release)
from the Socrata Open Data API and saves it as a raw CSV file.

Usage (from project root):
    python scripts/fetch_data.py
    python scripts/fetch_data.py --out data/raw/cdc_500cities_raw.csv
    python scripts/fetch_data.py --max-rows 100000   # quick dev subset

Output:
    data/raw/cdc_500cities_raw.csv  — full raw dataset
    data/raw/download_meta.json     — shape, timestamp, column list
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_ID   = "6vp6-wxuq"
BASE_URL     = f"https://data.cdc.gov/resource/{DATASET_ID}.csv"
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT  = PROJECT_ROOT / "data/raw/cdc_500cities_raw.csv"
META_OUT     = PROJECT_ROOT / "data/raw/download_meta.json"
PAGE_SIZE    = 50_000

# Columns expected in the dataset — used for schema validation after download.
REQUIRED_COLS = [
    "year", "stateabbr", "cityname", "geographiclevel",
    "measure", "data_value", "data_value_type", "tractfips",
    "cityfips", "geolocation", "populationcount", "category",
]


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_socrata_csv(
    endpoint: str,
    page_size: int = PAGE_SIZE,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """
    Page through a Socrata CSV endpoint using $limit / $offset.

    Args:
        endpoint  : Full Socrata .csv resource URL.
        page_size : Rows per HTTP request (50K is a safe upper bound for CDC).
        max_rows  : Hard cap for quick iteration; None fetches the full table.

    Returns:
        Concatenated DataFrame of all fetched pages.
    """
    chunks: list[pd.DataFrame] = []
    offset = 0
    total  = 0

    while True:
        limit = page_size
        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break
            limit = min(limit, remaining)

        params = {"$limit": limit, "$offset": offset}

        try:
            resp = requests.get(endpoint, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  [ERROR] Request failed at offset {offset}: {exc}", file=sys.stderr)
            raise

        chunk = pd.read_csv(pd.io.common.StringIO(resp.text))

        if chunk.empty:
            break

        chunks.append(chunk)
        got     = len(chunk)
        total  += got
        offset += got
        print(f"  Fetched {got:>6,} rows  (running total: {total:>7,})")

        # Last page is smaller than the requested limit — stop paging.
        if got < limit:
            break

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if any required column is absent from the download."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Downloaded dataset is missing expected columns: {missing}\n"
            "The Socrata schema may have changed — update REQUIRED_COLS."
        )
    print(f"  Schema validation passed  ({len(df.columns)} columns present)")


def validate_size(df: pd.DataFrame) -> None:
    """Warn if the row count is unexpectedly low (likely a partial download)."""
    if df.shape[0] < 500_000:
        print(
            f"  [WARNING] Only {df.shape[0]:,} rows downloaded — expected ~810K.\n"
            "  If you did not pass --max-rows, the download may be incomplete.",
            file=sys.stderr,
        )
    else:
        print(f"  Size check passed  ({df.shape[0]:,} rows, {df.shape[1]} columns)")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def write_metadata(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save a small JSON file next to the CSV recording provenance information.
    This makes it easy to verify the download is fresh and complete.
    """
    meta = {
        "source_url"       : BASE_URL,
        "dataset_id"       : DATASET_ID,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows"             : df.shape[0],
        "columns"          : df.shape[1],
        "column_names"     : list(df.columns),
        "geographic_levels": sorted(df["geographiclevel"].dropna().unique().tolist())
                             if "geographiclevel" in df.columns else [],
        "years"            : sorted(df["year"].dropna().unique().tolist())
                             if "year" in df.columns else [],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  Metadata written → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the CDC 500 Cities dataset from Socrata."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Cap the download at N rows (useful for quick dev runs).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=PAGE_SIZE,
        metavar="N",
        help=f"Rows per Socrata request (default: {PAGE_SIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("CDC 500 Cities — Data Fetch")
    print("=" * 60)
    if args.max_rows:
        print(f"[DEV MODE] Capping download at {args.max_rows:,} rows")
    print(f"Endpoint   : {BASE_URL}")
    print(f"Output     : {args.out}")
    print()

    t0 = time.time()

    print("Downloading...")
    df = fetch_socrata_csv(BASE_URL, page_size=args.page_size, max_rows=args.max_rows)

    print()
    print("Validating...")
    validate_schema(df)
    validate_size(df)

    print()
    print("Saving...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"  Raw CSV written → {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    write_metadata(df, META_OUT)

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.0f}s  |  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()
