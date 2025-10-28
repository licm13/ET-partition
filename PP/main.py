# -*- coding: utf-8 -*-
# @Author: Gemini AI
# @Date: 2025-07-17
# @Description: ET partitioning model batch processing master script (revised)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import et_partitioning_functions as etp
import os
import re

# ==================================================================
# === 1. Global Configuration ===
# ==================================================================
# BASE_PATH = r'Z:\Observation\FLUXNET.4.0\FLUXNET2015-Tier2'
BASE_PATH = r'Z:\LCM\ET_T_Partition\PP'

SITE_META_PATH = r'Z:\Observation\FLUXNET.4.0\Site_List.xlsx'
OUTPUT_PATH = "Output"
FOLDER_PATTERN = re.compile(r'^FLX_.*_FLUXNET2015_FULLSET_\d{4}-\d{4}_\d+-\d+$')
DEFAULT_ALTITUDE_KM = 0.5  # 新默认值

# === Load site elevation metadata ===
site_alt_df = pd.read_excel(SITE_META_PATH)
site_alt_df['LOCATION_ELEV_KM'] = site_alt_df['LOCATION_ELEV'] / 1000.0
site_alt_df = site_alt_df[['SITE_ID', 'LOCATION_ELEV_KM']].drop_duplicates()

# 用于记录哪些站点找不到 elevation
missing_altitude_sites = []

def get_site_altitude(site_name):
    """根据站点名查找 elevation，单位 km"""
    row = site_alt_df[site_alt_df['SITE_ID'] == site_name]
    if not row.empty and pd.notna(row.iloc[0]['LOCATION_ELEV_KM']):
        return float(row.iloc[0]['LOCATION_ELEV_KM'])
    else:
        missing_altitude_sites.append(site_name)
        return DEFAULT_ALTITUDE_KM

# ==================================================================
# === 2. Single Site Processing Function ===
# ==================================================================
def process_site_file(csv_filepath, output_dir):
    filename = os.path.basename(csv_filepath)
    try:
        site_name = filename.split('_')[1]
    except IndexError:
        print(f"   -> Could not parse site name from {filename}, skipping.")
        return

    print(f"\n[Processing Site]: {site_name}")
    print(f" -> Reading data file: {filename}")
    eddy_sample = pd.read_csv(csv_filepath, na_values=-9999)

    # Get altitude
    site_altitude_km = get_site_altitude(site_name)
    print(f"   -> Site elevation: {site_altitude_km:.3f} km")

    print(" -> Step 1: Calculating long-term parameters...")
    Chi_o = etp.calculate_chi_o(eddy_sample, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", c_coef=1.189, z=site_altitude_km)
    WUE_o = etp.calculate_WUE_o(eddy_sample, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", c_coef=1.189, z=site_altitude_km)

    print(" -> Step 2: Pre-processing data...")
    ds = eddy_sample.copy()
    ds['rDate'] = pd.to_datetime(ds['TIMESTAMP_END'].astype(str), format='%Y%m%d%H%M')
    ds['date'] = ds['rDate'].dt.date
    unique_dates = sorted(ds['date'].dropna().unique())
    date_map = {date: i + 1 for i, date in enumerate(unique_dates)}
    ds['loop'] = ds['date'].map(date_map)

    print(" -> Step 3: Starting daily ET partitioning loop...")
    unique_days = sorted(ds['loop'].dropna().unique())
    list_of_results = []

    for i in unique_days:
        if i < 3 or i > len(unique_days) - 2:
            continue
        window_indices = [i - 2, i - 1, i, i + 1, i + 2]
        tmp = ds[ds['loop'].isin(window_indices)].copy()
        tmpp = tmp[tmp['NIGHT'] == 0].copy()
        if tmpp.dropna(subset=['GPP_NT_VUT_MEAN', 'VPD_F', 'TA_F']).shape[0] < 50:
            continue
        par_lower = [0, 0, 10, 0]
        par_upper = [400, 0.4, 30, 1]
        optimal_par = etp.optimal_parameters(par_lower, par_upper, tmpp, Chi_o, WUE_o)
        transpiration_mod = etp.transpiration_model(optimal_par, tmp, Chi_o)
        landa = (3147.5 - 2.37 * (tmp['TA_F'].values + 273.15)) * 1000
        ET_mmol = tmp['LE_F_MDS'].values / landa * 1e6 / 18
        evaporation_mod = ET_mmol - transpiration_mod
        tmp['ET'] = ET_mmol
        tmp['transpiration_mod'] = transpiration_mod
        tmp['evaporation_mod'] = np.clip(evaporation_mod, a_min=0, a_max=None)
        central_day_result = tmp[tmp['loop'] == i]
        list_of_results.append(central_day_result)

    print(" -> Step 4: Post-processing and output...")
    if not list_of_results:
        print(f"   -> No valid result for {site_name}, skipping output.")
        return
    out = pd.concat(list_of_results)
    output_csv = os.path.join(output_dir, f"{site_name}_pp_output.csv")
    out.to_csv(output_csv, index=False)
    print(f" -> [Saved] CSV: {output_csv}")

    out['Hour'] = out['rDate'].dt.hour
    ET_agg = out.groupby('Hour')['ET'].mean()
    transp_agg = out.groupby('Hour')['transpiration_mod'].mean()
    evap_agg = out.groupby('Hour')['evaporation_mod'].mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(ET_agg.index, ET_agg.values, label='ET', color='black')
    ax.plot(transp_agg.index, transp_agg.values, label='Transpiration', linestyle='--', color='green')
    ax.plot(evap_agg.index, evap_agg.values, label='Evaporation', linestyle=':', color='red')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Flux (mmol/m²/s)")
    ax.set_title(f"Site {site_name} Diurnal Water Fluxes")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{site_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f" -> [Saved] Plot: {plot_path}")

# ==================================================================
# === 3. Main Execution Logic ===
# ==================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print("=" * 60)
    print(f"Starting batch process: {BASE_PATH}")
    print("=" * 60)

    for folder_name in os.listdir(BASE_PATH):
        folder_path = os.path.join(BASE_PATH, folder_name)
        if os.path.isdir(folder_path) and FOLDER_PATTERN.match(folder_name):
            csv_filename = folder_name.replace('_FLUXNET2015_FULLSET_', '_FLUXNET2015_FULLSET_HH_') + '.csv'
            csv_filepath = os.path.join(folder_path, csv_filename)
            if not os.path.exists(csv_filepath):
                print(f" -> CSV not found for folder: {folder_name}")
                continue
            try:
                process_site_file(csv_filepath, OUTPUT_PATH)
            except Exception as e:
                print(f"[Error processing {folder_name}]: {e}")

    # 保存 elevation 缺失站点
    if missing_altitude_sites:
        missing_path = os.path.join(OUTPUT_PATH, 'missing_altitude_sites.csv')
        pd.Series(missing_altitude_sites, name='SITE_ID').drop_duplicates().to_csv(missing_path, index=False)
        print(f"\n[Saved missing altitude sites] -> {missing_path}")

    print("=" * 60)
    print("Batch processing complete.")
    print("=" * 60)
