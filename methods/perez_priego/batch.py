"""Batch processing entry point for the Perez-Priego ET partition method."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import et_partitioning_functions as etp
from ..common_utils import (
    DEFAULT_FLUXNET_PATTERN,
    get_expected_csv_filename,
    extract_site_id,
    iter_site_folders,
    process_sites_parallel,
)

# Perez-Priego uses stricter pattern (FLX only, not AMF)
FOLDER_PATTERN = re.compile(r"^FLX_.*_FLUXNET2015_FULLSET_\d{4}-\d{4}_\d+-\d+$")


def load_site_metadata(site_meta_path: Optional[Path]) -> pd.DataFrame:
    """Return site elevation metadata with altitude in kilometres."""

    if site_meta_path is None:
        return pd.DataFrame(columns=["SITE_ID", "LOCATION_ELEV_KM"])

    if not site_meta_path.exists():
        raise FileNotFoundError(f"Site metadata file not found: {site_meta_path}")

    site_alt_df = pd.read_excel(site_meta_path)
    site_alt_df["LOCATION_ELEV_KM"] = site_alt_df["LOCATION_ELEV"] / 1000.0
    return site_alt_df[["SITE_ID", "LOCATION_ELEV_KM"]].drop_duplicates()


def get_site_altitude(
    site_name: str,
    site_alt_df: pd.DataFrame,
    default_altitude_km: float,
    missing_altitude_sites: List[str],
) -> float:
    """Lookup the site altitude, recording sites without metadata."""

    if not site_alt_df.empty:
        row = site_alt_df[site_alt_df["SITE_ID"] == site_name]
        if not row.empty and pd.notna(row.iloc[0]["LOCATION_ELEV_KM"]):
            return float(row.iloc[0]["LOCATION_ELEV_KM"])

    missing_altitude_sites.append(site_name)
    return default_altitude_km


def process_site_file(
    csv_filepath: Path,
    output_dir: Path,
    site_alt_df: pd.DataFrame,
    default_altitude_km: float,
    missing_altitude_sites: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Run the Perez-Priego partitioning workflow for a single site.
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (success, error_message)
    """
    filename = csv_filepath.name
    try:
        site_name = extract_site_id(filename)
    except IndexError:
        error_msg = f"Could not parse site name from {filename}"
        print(f"   -> {error_msg}, skipping.")
        return False, error_msg

    print(f"\n[Processing Site]: {site_name}")
    print(f" -> Reading data file: {filename}")
    
    try:
        eddy_data = pd.read_csv(csv_filepath, na_values=-9999)
    except Exception as exc:
        error_msg = f"Failed to read CSV: {exc}"
        print(f"   -> {error_msg}")
        return False, error_msg

    # Get altitude
    site_altitude_km = get_site_altitude(
        site_name, site_alt_df, default_altitude_km, missing_altitude_sites
    )
    print(f"   -> Site elevation: {site_altitude_km:.3f} km")

    print(" -> Step 1: Calculating long-term parameters...")
    try:
        chi_optimal = etp.calculate_chi_o(
            eddy_data, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", 
            c_coef=1.189, z=site_altitude_km
        )
        wue_optimal = etp.calculate_WUE_o(
            eddy_data, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", 
            c_coef=1.189, z=site_altitude_km
        )
    except Exception as exc:
        error_msg = f"Failed to calculate parameters: {exc}"
        print(f"   -> {error_msg}")
        return False, error_msg

    print(" -> Step 2: Pre-processing data...")
    processed_data = eddy_data.copy()
    processed_data["rDate"] = pd.to_datetime(
        processed_data["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M"
    )
    processed_data["date"] = processed_data["rDate"].dt.date
    unique_dates = sorted(processed_data["date"].dropna().unique())
    date_map = {date: idx + 1 for idx, date in enumerate(unique_dates)}
    processed_data["loop"] = processed_data["date"].map(date_map)

    print(" -> Step 3: Starting daily ET partitioning loop...")
    unique_days = sorted(processed_data["loop"].dropna().unique())
    results_list = []

    for day_idx in unique_days:
        # Skip edge days (need 5-day window: ±2 days)
        if day_idx < 3 or day_idx > len(unique_days) - 2:
            continue
        
        # Select 5-day window
        window_indices = [day_idx - 2, day_idx - 1, day_idx, day_idx + 1, day_idx + 2]
        window_data = processed_data[processed_data["loop"].isin(window_indices)].copy()
        daytime_data = window_data[window_data["NIGHT"] == 0].copy()
        
        # Check if sufficient data available
        required_cols = ["GPP_NT_VUT_MEAN", "VPD_F", "TA_F"]
        if daytime_data.dropna(subset=required_cols).shape[0] < 50:
            continue
        
        # Optimize parameters for this window
        param_lower = [0, 0, 10, 0]
        param_upper = [400, 0.4, 30, 1]
        
        try:
            optimal_params = etp.optimal_parameters(
                param_lower, param_upper, daytime_data, chi_optimal, wue_optimal
            )
        except Exception:
            continue  # Skip this day if optimization fails
        
        # Calculate transpiration and evaporation
        transpiration = etp.transpiration_model(optimal_params, window_data, chi_optimal)
        
        # Calculate ET from latent heat
        lambda_water = (3147.5 - 2.37 * (window_data["TA_F"].values + 273.15)) * 1000
        et_mmol = window_data["LE_F_MDS"].values / lambda_water * 1e6 / 18
        evaporation = et_mmol - transpiration
        
        # Store results for central day
        window_data["ET"] = et_mmol
        window_data["transpiration_mod"] = transpiration
        window_data["evaporation_mod"] = np.clip(evaporation, a_min=0, a_max=None)
        central_day_result = window_data[window_data["loop"] == day_idx]
        results_list.append(central_day_result)

    print(" -> Step 4: Post-processing and output...")
    if not results_list:
        error_msg = "No valid result for site"
        print(f"   -> {error_msg}, skipping output.")
        return False, error_msg
    
    output_data = pd.concat(results_list)
    output_csv = output_dir / f"{site_name}_pp_output.csv"
    output_data.to_csv(output_csv, index=False)
    print(f" -> [Saved] CSV: {output_csv}")

    # Create diagnostic plot
    output_data["Hour"] = output_data["rDate"].dt.hour
    et_hourly = output_data.groupby("Hour")["ET"].mean()
    transp_hourly = output_data.groupby("Hour")["transpiration_mod"].mean()
    evap_hourly = output_data.groupby("Hour")["evaporation_mod"].mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(et_hourly.index, et_hourly.values, label="ET", color="black")
    ax.plot(
        transp_hourly.index, transp_hourly.values,
        label="Transpiration", linestyle="--", color="green",
    )
    ax.plot(
        evap_hourly.index, evap_hourly.values,
        label="Evaporation", linestyle=":", color="red",
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Flux (mmol/m²/s)")
    ax.set_title(f"Site {site_name} Diurnal Water Fluxes")
    ax.legend()
    plt.tight_layout()
    plot_path = output_dir / f"{site_name}_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f" -> [Saved] Plot: {plot_path}")
    
    return True, None


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line interface for batch processing."""

    repo_root = Path(__file__).resolve().parents[2]
    default_base = repo_root / "data" / "test_site"
    default_output = repo_root / "outputs" / "perez_priego"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=default_base,
        help="Directory containing Fluxnet-style site folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output,
        help="Directory where processed outputs will be written.",
    )
    parser.add_argument(
        "--site-metadata",
        type=Path,
        default=None,
        help="Optional Excel file with SITE_ID and LOCATION_ELEV columns.",
    )
    parser.add_argument(
        "--default-altitude",
        type=float,
        default=0.5,
        help="Fallback site altitude in kilometres when metadata are missing.",
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

    site_alt_df = load_site_metadata(args.site_metadata)
    missing_altitude_sites: List[str] = []

    print("=" * 60)
    print(f"Starting batch process: {args.base_path}")
    print("=" * 60)

    # Collect all site files to process
    site_files = []
    for folder_path in iter_site_folders(args.base_path, FOLDER_PATTERN):
        try:
            csv_filename = get_expected_csv_filename(folder_path.name)
        except ValueError:
            print(f" -> Invalid folder name: {folder_path.name}")
            continue
            
        csv_filepath = folder_path / csv_filename
        if not csv_filepath.exists():
            print(f" -> CSV not found for folder: {folder_path.name}")
            continue
        site_files.append(csv_filepath)

    print(f"Found {len(site_files)} site files to process")

    if not site_files:
        print("No site files found. Exiting.")
        return

    if args.parallel:
        # Parallel processing using common utility
        def process_wrapper(csv_filepath: Path) -> Tuple[bool, Optional[str]]:
            """Wrapper for parallel execution"""
            return process_site_file(
                csv_filepath,
                args.output_path,
                site_alt_df,
                args.default_altitude,
                missing_altitude_sites,
            )
        
        successful, failed, errors = process_sites_parallel(
            site_files,
            process_wrapper,
            workers=args.workers,
            description="Processing Perez-Priego sites"
        )
        
        if errors:
            print(f"\n{len(errors)} files had errors during processing")
    else:
        # Serial processing
        successful = 0
        failed = 0
        for csv_filepath in site_files:
            success, error_msg = process_site_file(
                csv_filepath,
                args.output_path,
                site_alt_df,
                args.default_altitude,
                missing_altitude_sites,
            )
            if success:
                successful += 1
            else:
                failed += 1

    if missing_altitude_sites:
        missing_path = args.output_path / "missing_altitude_sites.csv"
        pd.Series(missing_altitude_sites, name="SITE_ID").drop_duplicates().to_csv(
            missing_path, index=False
        )
        print(f"\n[Saved missing altitude sites] -> {missing_path}")

    print("=" * 60)
    print(f"Processing complete: {successful} successful, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
