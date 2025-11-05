"""Batch processing workflow for the TEA transpiration partitioning algorithm."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .TEA.TEA import simplePartition
from ..common_utils import (
    DEFAULT_FLUXNET_PATTERN,
    get_expected_csv_filename,
    extract_site_id,
    iter_site_folders,
    process_sites_parallel,
)


def process_site_folder(folder_path: Path, output_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Execute the TEA workflow for a single Fluxnet-style folder.
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (success, error_message)
    """
    folder_name = folder_path.name
    print(f"\n[Processing folder]: {folder_name}")

    try:
        csv_filename = get_expected_csv_filename(folder_name)
    except ValueError as exc:
        error_msg = f"Invalid folder name: {exc}"
        print(f"  -> {error_msg}")
        return False, error_msg
    
    csv_filepath = folder_path / csv_filename

    if not csv_filepath.exists():
        error_msg = f"CSV file not found: {csv_filename}"
        print(f"  -> {error_msg}, skipping.")
        return False, error_msg

    print(f"  -> Reading file: {csv_filename}")

    try:
        input_data = pd.read_csv(csv_filepath, on_bad_lines="skip")
    except Exception as exc:
        error_msg = f"Failed to read CSV file: {exc}"
        print(f"  -> {error_msg}")
        return False, error_msg

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
    missing_cols = [col for col in original_columns if col not in input_data.columns]
    if missing_cols:
        error_msg = f"Missing required columns {missing_cols}"
        print(f"  -> {error_msg}, skipping.")
        return False, error_msg

    processed_data = input_data[original_columns].copy()
    processed_data.rename(columns=column_mapping, inplace=True)
    processed_data["ET"] = processed_data["ET"] * 0.0007348

    num_rows = len(processed_data)
    processed_data["timestamp"] = range(0, num_rows * 30, 30)
    processed_data = processed_data[
        ["timestamp"] + [col for col in processed_data.columns if col != "timestamp"]
    ]

    print("  -> Pre-processing finished, running TEA simplePartition...")

    timestamp = processed_data["timestamp"].values
    et_values = processed_data["ET"].values
    gpp_values = processed_data["GPP"].values
    rh_values = processed_data["RH"].values
    rg_values = processed_data["Rg"].values
    rg_pot_values = processed_data["Rg_pot"].values
    tair_values = processed_data["Tair"].values
    vpd_values = processed_data["VPD"].values
    precip_values = processed_data["precip"].values
    u_values = processed_data["u"].values

    tea_transpiration, tea_evaporation, tea_wue = simplePartition(
        timestamp, et_values, gpp_values, rh_values, rg_values, rg_pot_values,
        tair_values, vpd_values, precip_values, u_values
    )

    try:
        sitename = extract_site_id(folder_name)
    except IndexError:
        error_msg = f"Could not extract site ID from folder name"
        print(f"  -> {error_msg}")
        return False, error_msg
        
    output_filename = f"{sitename}_TEA_results.csv"
    output_filepath = output_path / output_filename

    results_dataframe = pd.DataFrame(
        {
            "timestamp": timestamp,
            "TEA_T": tea_transpiration,
            "TEA_E": tea_evaporation,
            "TEA_WUE": tea_wue,
        }
    )
    results_dataframe.to_csv(output_filepath, index=False)
    print(f"  -> Saved results to: {output_filepath}")
    
    return True, None


def main(argv: Optional[list[str]] = None) -> None:
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
        default=DEFAULT_FLUXNET_PATTERN.pattern,
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

    print("=" * 60)
    print("TEA batch processing")
    print(f"Scanning directory: {args.base_path}")
    print(f"Output directory:   {args.output_path}")

    # Collect all folders
    site_folders = list(iter_site_folders(args.base_path, folder_pattern))
    print(f"Found {len(site_folders)} site folders")

    if not site_folders:
        print("No site folders found. Exiting.")
        return

    if args.parallel:
        # Parallel processing using common utility
        def process_wrapper(folder_path: Path) -> Tuple[bool, Optional[str]]:
            """Wrapper for parallel execution"""
            return process_site_folder(folder_path, args.output_path)
        
        successful, failed, errors = process_sites_parallel(
            site_folders,
            process_wrapper,
            workers=args.workers,
            description="Processing TEA sites"
        )
        
        if errors:
            print(f"\n{len(errors)} sites had errors during processing")
    else:
        # Serial processing
        successful = 0
        failed = 0
        for folder_path in site_folders:
            success, error_msg = process_site_folder(folder_path, args.output_path)
            if success:
                successful += 1
            else:
                failed += 1

    print("=" * 60)
    print(f"Processing complete: {successful} successful, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
