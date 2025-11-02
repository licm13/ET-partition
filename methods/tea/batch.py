"""Batch processing workflow for the TEA transpiration partitioning algorithm."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import pandas as pd

from .TEA.TEA import simplePartition

DEFAULT_PATTERN = re.compile(
    r"^(?:AMF|FLX)_.*_FLUXNET(?:2015)?_FULLSET_\d{4}-\d{4}_\d+-\d+$"
)


def _expected_csv_name(folder_name: str) -> str:
    if "_FLUXNET2015_FULLSET_" in folder_name:
        return folder_name.replace(
            "_FLUXNET2015_FULLSET_", "_FLUXNET2015_FULLSET_HH_"
        ) + ".csv"
    if "_FLUXNET_FULLSET_" in folder_name:
        return folder_name.replace("_FLUXNET_FULLSET_", "_FLUXNET_FULLSET_HH_") + ".csv"
    raise ValueError(f"Folder name does not follow expected pattern: {folder_name}")


def process_site_folder(folder_path: Path, output_path: Path) -> None:
    """Execute the TEA workflow for a single Fluxnet-style folder."""

    folder_name = folder_path.name
    print(f"\n[Processing folder]: {folder_name}")

    csv_filename = _expected_csv_name(folder_name)
    csv_filepath = folder_path / csv_filename

    if not csv_filepath.exists():
        print(f"  -> CSV file not found: {csv_filename}, skipping.")
        return

    print(f"  -> Reading file: {csv_filename}")

    try:
        df = pd.read_csv(csv_filepath, on_bad_lines="skip")
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"  -> Failed to read CSV file: {exc}")
        return

    column_mapping = {
        "LE_F_MDS": "ET",
        "GPP_NT_VUT_REF": "GPP",
        "TA_F_MDS": "Tair",
        "RH": "RH",
        "VPD_F_MDS": "VPD",
        "P_ERA": "precip",
        "SW_IN_F": "Rg",
        "WS": "u",
        "SW_IN_POT": "Rg_pot",
    }

    original_columns = list(column_mapping.keys())
    missing_cols = [col for col in original_columns if col not in df.columns]
    if missing_cols:
        print(f"  -> Missing required columns {missing_cols}, skipping.")
        return

    processed_df = df[original_columns].copy()
    processed_df.rename(columns=column_mapping, inplace=True)
    processed_df["ET"] = processed_df["ET"] * 0.0007348

    num_rows = len(processed_df)
    processed_df["timestamp"] = range(0, num_rows * 30, 30)
    processed_df = processed_df[
        ["timestamp"] + [col for col in processed_df.columns if col != "timestamp"]
    ]

    print("  -> Pre-processing finished, running TEA simplePartition...")

    timestamp = processed_df["timestamp"].values
    ET = processed_df["ET"].values
    GPP = processed_df["GPP"].values
    RH = processed_df["RH"].values
    Rg = processed_df["Rg"].values
    Rg_pot = processed_df["Rg_pot"].values
    Tair = processed_df["Tair"].values
    VPD = processed_df["VPD"].values
    precip = processed_df["precip"].values
    u = processed_df["u"].values

    TEA_T, TEA_E, TEA_WUE = simplePartition(
        timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u
    )

    sitename = folder_name.split("_")[1]
    output_filename = f"{sitename}_TEA_results.csv"
    output_filepath = output_path / output_filename

    results_df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "TEA_T": TEA_T,
            "TEA_E": TEA_E,
            "TEA_WUE": TEA_WUE,
        }
    )
    results_df.to_csv(output_filepath, index=False)
    print(f"  -> Saved results to: {output_filepath}")


def iter_site_folders(
    base_path: Path, pattern: re.Pattern[str] = DEFAULT_PATTERN
) -> Iterable[Path]:
    """Yield folders that match the TEA naming convention."""

    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    for entry in sorted(base_path.iterdir()):
        if entry.is_dir() and pattern.match(entry.name):
            yield entry


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Command-line entry point for the TEA batch workflow."""

    repo_root = Path(__file__).resolve().parents[2]
    default_base = repo_root / "data" / "test_site"
    default_output = repo_root / "outputs" / "tea"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=default_base,
        help="Directory containing Fluxnet/AmeriFlux style site folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output,
        help="Directory where TEA results will be stored.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN.pattern,
        help="Regular expression used to match site folder names.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing of sites.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores).",
    )

    args = parser.parse_args(args=list(argv) if argv is not None else None)
    args.output_path.mkdir(parents=True, exist_ok=True)
    folder_pattern = re.compile(args.pattern)

    print("--- TEA batch processing ---")
    print(f"Scanning directory: {args.base_path}")
    print(f"Output directory:  {args.output_path}")

    # Collect all folders
    site_folders = list(iter_site_folders(args.base_path, folder_pattern))
    print(f"Found {len(site_folders)} site folders")

    if args.parallel:
        # Parallel processing
        workers = args.workers if args.workers else multiprocessing.cpu_count()
        print(f"Using parallel processing with {workers} workers")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_folder = {
                executor.submit(process_site_folder, folder, args.output_path): folder
                for folder in site_folders
            }

            # Process completed tasks
            completed = 0
            total = len(site_folders)

            for future in as_completed(future_to_folder):
                folder_path = future_to_folder[future]
                completed += 1

                try:
                    future.result()
                    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
                except Exception as e:
                    print(f"Error processing {folder_path.name}: {e}")

    else:
        # Serial processing
        for folder_path in site_folders:
            process_site_folder(folder_path, args.output_path)

    print("--- Processing complete ---")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
