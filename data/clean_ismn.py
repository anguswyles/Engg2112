"""
ISMN Data Cleaner
-----------------
Processes all .stm files from an ISMN Header & Values format zip or directory.
Outputs a single combined CSV with quality flag tagging columns added.

Usage:
    python clean_ismn.py --input <zip_or_folder> --output <output.csv>
"""

import os
import re
import sys
import zipfile
import argparse
import pandas as pd
from io import StringIO
from pathlib import Path


# ── Flag classification ────────────────────────────────────────────────────────

# ISMN flags that indicate a dubious/flagged observation
BAD_ISMN_FLAG_PATTERN = re.compile(r'\b(C0[1-3]|D0[1-9]|D10)\b')


def classify_ismn_flag(flag_str: str) -> dict:
    """
    Given the raw ISMN quality flag string (e.g. 'G', 'C03,D10', 'U'),
    return a dict of boolean tag columns.
    """
    flag_str = str(flag_str).strip()
    return {
        "ismn_flag_raw":     flag_str,
        "ismn_flag_good":    flag_str == "G",
        "ismn_flag_unknown": flag_str == "U",
        "ismn_flag_bad":     bool(BAD_ISMN_FLAG_PATTERN.search(flag_str)),
    }


# ── Header parser ──────────────────────────────────────────────────────────────

def parse_header(line: str) -> dict:
    """
    Parse the single header line of an ISMN .stm file.
    Format: CSE  Network  Station  Lat  Lon  Elevation  DepthFrom  DepthTo  [Sensor...]
    Fields are whitespace-separated; sensor name may contain spaces so we
    capture everything after the 8th token.
    """
    parts = line.split()
    if len(parts) < 8:
        raise ValueError(f"Header has fewer than 8 fields: {line!r}")
    return {
        "cse":        parts[0],
        "network":    parts[1],
        "station":    parts[2],
        "latitude":   float(parts[3]),
        "longitude":  float(parts[4]),
        "elevation_m": float(parts[5]),
        "depth_from_m": float(parts[6]),
        "depth_to_m":   float(parts[7]),
        "sensor":     " ".join(parts[8:]) if len(parts) > 8 else "",
    }


# ── Record parser ──────────────────────────────────────────────────────────────

def parse_records(lines: list[str], header_meta: dict, source_file: str) -> pd.DataFrame:
    """
    Parse data record lines into a DataFrame. Each record line:
        yyyy/mm/dd HH:MM  value  ismn_flag  provider_flag
    """
    rows = []
    for lineno, line in enumerate(lines, start=2):  # line 1 is header
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # ── Malformed row check ────────────────────────────────────────────────
        if len(parts) != 5:
            rows.append({
                **header_meta,
                "source_file":      source_file,
                "datetime_utc":     None,
                "sm_value":         None,
                "ismn_flag_raw":    None,
                "ismn_flag_good":   False,
                "ismn_flag_unknown":False,
                "ismn_flag_bad":    False,
                "provider_flag":    None,
                "issue_malformed":  True,
                "issue_bad_value":  False,
                "issue_duplicate":  False,   # filled later
            })
            continue

        date_str, time_str, val_str, ismn_flag, prov_flag = parts

        # ── Timestamp parse ────────────────────────────────────────────────────
        try:
            dt = pd.to_datetime(f"{date_str} {time_str}", format="%Y/%m/%d %H:%M", utc=True)
        except Exception:
            dt = None

        # ── Value parse ───────────────────────────────────────────────────────
        try:
            value = float(val_str)
            bad_value = False
            # -9999 / -999 are common ISMN no-data sentinels
            if value in (-9999.0, -999.0, -9999, -999):
                bad_value = True
                value = None
        except ValueError:
            value = None
            bad_value = True

        flag_info = classify_ismn_flag(ismn_flag)

        rows.append({
            **header_meta,
            "source_file":        source_file,
            "datetime_utc":       dt,
            "sm_value":           value,
            **flag_info,
            "provider_flag":      prov_flag.strip(),
            "issue_malformed":    (dt is None),
            "issue_bad_value":    bad_value,
            "issue_duplicate":    False,   # filled below
        })

    df = pd.DataFrame(rows)

    # ── Duplicate timestamp detection ─────────────────────────────────────────
    if not df.empty and "datetime_utc" in df.columns:
        dup_mask = df["datetime_utc"].notna() & df.duplicated(subset=["datetime_utc"], keep=False)
        df.loc[dup_mask, "issue_duplicate"] = True
        # Keep first occurrence; mark rest
        first_dup = df["datetime_utc"].notna() & df.duplicated(subset=["datetime_utc"], keep="first")
        df = df[~first_dup].copy()

    return df


# ── File reader ────────────────────────────────────────────────────────────────

def read_stm_file(content: str, source_file: str) -> pd.DataFrame:
    lines = content.splitlines()
    if not lines:
        return pd.DataFrame()

    try:
        header_meta = parse_header(lines[0])
    except Exception as e:
        print(f"  WARNING: Could not parse header of {source_file}: {e}", file=sys.stderr)
        return pd.DataFrame()

    return parse_records(lines[1:], header_meta, source_file)


# ── Source loader (zip or directory) ──────────────────────────────────────────

def iter_stm_files(source: str):
    """Yield (name, content_str) for every .stm file in a zip or directory."""
    source = Path(source)

    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as zf:
            stm_names = [n for n in zf.namelist() if n.endswith(".stm")]
            for name in stm_names:
                with zf.open(name) as fh:
                    content = fh.read().decode("utf-8", errors="replace")
                yield name, content

    elif source.is_dir():
        for path in sorted(source.rglob("*.stm")):
            content = path.read_text(encoding="utf-8", errors="replace")
            yield str(path.relative_to(source)), content

    else:
        raise ValueError(f"Source must be a .zip file or a directory, got: {source}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean ISMN Header & Values .stm files")
    parser.add_argument("--input",  required=True, help="Path to .zip or extracted folder")
    parser.add_argument("--output", required=True, help="Path for output CSV")
    args = parser.parse_args()

    all_frames = []
    file_count = 0

    print(f"Reading from: {args.input}")
    for name, content in iter_stm_files(args.input):
        file_count += 1
        print(f"  [{file_count}] {name}")
        df = read_stm_file(content, source_file=name)
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        print("No data found. Exiting.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)

    # ── Sort by station + time ─────────────────────────────────────────────────
    combined.sort_values(["network", "station", "datetime_utc"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    total         = len(combined)
    n_good        = combined["ismn_flag_good"].sum()
    n_unknown     = combined["ismn_flag_unknown"].sum()
    n_bad_flag    = combined["ismn_flag_bad"].sum()
    n_bad_value   = combined["issue_bad_value"].sum()
    n_malformed   = combined["issue_malformed"].sum()
    n_duplicate   = combined["issue_duplicate"].sum()

    print("\n── Cleaning summary ──────────────────────────────────────────")
    print(f"  Files processed     : {file_count}")
    print(f"  Total rows          : {total:,}")
    print(f"  ISMN flag G (good)  : {n_good:,}  ({100*n_good/total:.1f}%)")
    print(f"  ISMN flag U (unkwn) : {n_unknown:,}  ({100*n_unknown/total:.1f}%)")
    print(f"  ISMN flag bad (C/D) : {n_bad_flag:,}  ({100*n_bad_flag/total:.1f}%)")
    print(f"  Bad/sentinel values : {n_bad_value:,}")
    print(f"  Malformed rows      : {n_malformed:,}")
    print(f"  Duplicate timestamps: {n_duplicate:,}  (first kept, rest dropped)")
    print("──────────────────────────────────────────────────────────────")

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nCleaned data written to: {out_path}")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
