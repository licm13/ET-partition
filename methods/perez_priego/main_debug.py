# -*- coding: utf-8 -*-
# @Author: Gemini AI
# @Date: 2025-07-23
# @Description: 调试版 ET 分配主脚本（仅处理单站，带日志）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import et_partitioning_functions as etp
import os

# 设置路径
CSV_FILE = r"./FLX_FI-Hyy_FLUXNET2015_FULLSET_HH_2008-2010_1-3.csv"
OUTPUT_PATH = "./Output"
SITE_NAME = "FI-Hyy"
DEFAULT_ALTITUDE_KM = 0.5

def get_site_altitude(site_name):
    return DEFAULT_ALTITUDE_KM  # 若无站点元数据，统一返回默认值

def process_site_file(csv_filepath, output_dir):
    filename = os.path.basename(csv_filepath)
    print(f"\n[Processing Site]: {SITE_NAME}")
    print(f" -> Reading data file: {filename}")
    eddy_sample = pd.read_csv(csv_filepath, na_values=-9999)

    # Get altitude
    site_altitude_km = get_site_altitude(SITE_NAME)
    print(f"   -> Site elevation: {site_altitude_km:.3f} km")

    print(" -> Step 1: Calculating long-term parameters...")
    Chi_o = etp.calculate_chi_o(eddy_sample, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", c_coef=1.189, z=site_altitude_km)
    WUE_o = etp.calculate_WUE_o(eddy_sample, "GPP_NT_VUT_MEAN", "VPD_F", "TA_F", c_coef=1.189, z=site_altitude_km)

    print(" -> Step 2: Pre-processing data...")
    ds = eddy_sample.copy()
    ds['rDate'] = pd.to_datetime(ds['TIMESTAMP_END'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    ds = ds.dropna(subset=['rDate'])
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
        valid_points = tmpp.dropna(subset=['GPP_NT_VUT_MEAN', 'VPD_F', 'TA_F']).shape[0]
        print(f"   -> Day {i}/{len(unique_days)}, valid points = {valid_points}")
        if valid_points < 30:
            print("      [Skipped] Not enough valid data.")
            continue
        try:
            print("      -> Optimizing parameters...")
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
            print("      -> Success.")
        except Exception as e:
            print(f"      [Error] Skipping day {i}: {e}")
            continue

    print(" -> Step 4: Post-processing and output...")
    if not list_of_results:
        print(f"   -> No valid result for {SITE_NAME}, skipping output.")
        return
    out = pd.concat(list_of_results)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{SITE_NAME}_pp_output.csv")
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
    ax.set_title(f"Site {SITE_NAME} Diurnal Water Fluxes")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{SITE_NAME}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f" -> [Saved] Plot: {plot_path}")

if __name__ == "__main__":
    print("=" * 60)
    print(f"Starting single site test for: {CSV_FILE}")
    print("=" * 60)
    process_site_file(CSV_FILE, OUTPUT_PATH)
    print("=" * 60)
    print("Processing complete.")
    print("=" * 60)
