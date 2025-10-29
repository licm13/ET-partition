"""Batch processing entry point for the Perez-Priego ET partition method."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import et_partitioning_functions as etp

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
) -> None:
    """Run the Perez-Priego partitioning workflow for a single site."""

    filename = csv_filepath.name
    try:
        site_name = filename.split("_")[1]
    except IndexError:
        print(f"   -> Could not parse site name from {filename}, skipping.")
        return

    print(f"\n[Processing Site]: {site_name}")
    print(f" -> Reading data file: {filename}")
    eddy_sample = pd.read_csv(csv_filepath, na_values=-9999)

    # Get altitude
    site_altitude_km = get_site_altitude(
        site_name, site_alt_df, default_altitude_km, missing_altitude_sites
    )
    print(f"   -> Site elevation: {site_altitude_km:.3f} km")

    print(" -> Step 1: Calculating long-term parameters...")
    Chi_o = etp.calculate_chi_o(
        eddy_sample, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", c_coef=1.189, z=site_altitude_km
    )
    WUE_o = etp.calculate_WUE_o(
        eddy_sample, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", c_coef=1.189, z=site_altitude_km
    )

    print(" -> Step 2: Pre-processing data...")
    ds = eddy_sample.copy()
    ds["rDate"] = pd.to_datetime(ds["TIMESTAMP_END"].astype(str), format="%Y%m%d%H%M")
    ds["date"] = ds["rDate"].dt.date
    unique_dates = sorted(ds["date"].dropna().unique())
    date_map = {date: i + 1 for i, date in enumerate(unique_dates)}
    ds["loop"] = ds["date"].map(date_map)

    print(" -> Step 3: Starting daily ET partitioning loop...")
    unique_days = sorted(ds["loop"].dropna().unique())
    list_of_results = []

    for i in unique_days:
        if i < 3 or i > len(unique_days) - 2:
            continue
        window_indices = [i - 2, i - 1, i, i + 1, i + 2]
        tmp = ds[ds["loop"].isin(window_indices)].copy()
        tmpp = tmp[tmp["NIGHT"] == 0].copy()
        if tmpp.dropna(subset=["GPP_NT_VUT_MEAN", "VPD_F", "TA_F"]).shape[0] < 50:
            continue
        par_lower = [0, 0, 10, 0]
        par_upper = [400, 0.4, 30, 1]
        optimal_par = etp.optimal_parameters(par_lower, par_upper, tmpp, Chi_o, WUE_o)
        transpiration_mod = etp.transpiration_model(optimal_par, tmp, Chi_o)
        landa = (3147.5 - 2.37 * (tmp["TA_F"].values + 273.15)) * 1000
        ET_mmol = tmp["LE_F_MDS"].values / landa * 1e6 / 18
        evaporation_mod = ET_mmol - transpiration_mod
        tmp["ET"] = ET_mmol
        tmp["transpiration_mod"] = transpiration_mod
        tmp["evaporation_mod"] = np.clip(evaporation_mod, a_min=0, a_max=None)
        central_day_result = tmp[tmp["loop"] == i]
        list_of_results.append(central_day_result)

    print(" -> Step 4: Post-processing and output...")
    if not list_of_results:
        print(f"   -> No valid result for {site_name}, skipping output.")
        return
    out = pd.concat(list_of_results)
    output_csv = output_dir / f"{site_name}_pp_output.csv"
    out.to_csv(output_csv, index=False)
    print(f" -> [Saved] CSV: {output_csv}")

    out["Hour"] = out["rDate"].dt.hour
    ET_agg = out.groupby("Hour")["ET"].mean()
    transp_agg = out.groupby("Hour")["transpiration_mod"].mean()
    evap_agg = out.groupby("Hour")["evaporation_mod"].mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(ET_agg.index, ET_agg.values, label="ET", color="black")
    ax.plot(
        transp_agg.index,
        transp_agg.values,
        label="Transpiration",
        linestyle="--",
        color="green",
    )
    ax.plot(
        evap_agg.index,
        evap_agg.values,
        label="Evaporation",
        linestyle=":",
        color="red",
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


def iter_site_folders(base_path: Path, pattern: re.Pattern[str]) -> Iterable[Path]:
    """Yield subdirectories that match the Fluxnet naming convention."""

    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    for folder in sorted(base_path.iterdir()):
        if folder.is_dir() and pattern.match(folder.name):
            yield folder


def main(argv: Optional[Iterable[str]] = None) -> None:
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

    args = parser.parse_args(args=list(argv) if argv is not None else None)

    args.output_path.mkdir(parents=True, exist_ok=True)

    site_alt_df = load_site_metadata(args.site_metadata)
    missing_altitude_sites: List[str] = []

    print("=" * 60)
    print(f"Starting batch process: {args.base_path}")
    print("=" * 60)

    for folder_path in iter_site_folders(args.base_path, FOLDER_PATTERN):
        csv_filename = (
            folder_path.name.replace("_FLUXNET2015_FULLSET_", "_FLUXNET2015_FULLSET_HH_")
            + ".csv"
        )
        csv_filepath = folder_path / csv_filename
        if not csv_filepath.exists():
            print(f" -> CSV not found for folder: {folder_path.name}")
            continue
        try:
            process_site_file(
                csv_filepath,
                args.output_path,
                site_alt_df,
                args.default_altitude,
                missing_altitude_sites,
            )
        except Exception as exc:  # pragma: no cover - rich CLI feedback
            print(f"[Error processing {folder_path.name}]: {exc}")

    if missing_altitude_sites:
        missing_path = args.output_path / "missing_altitude_sites.csv"
        pd.Series(missing_altitude_sites, name="SITE_ID").drop_duplicates().to_csv(
            missing_path, index=False
        )
        print(f"\n[Saved missing altitude sites] -> {missing_path}")

    print("=" * 60)
    print("Batch processing complete.")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
