#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ET Partition - Python API Usage Examples
=========================================

This script demonstrates how to use the core ET partitioning algorithms directly
without going through the batch processing layer.

This is useful when you want to:
- Integrate ET partitioning into your own data pipeline
- Process custom data formats
- Have fine-grained control over preprocessing steps

Requirements:
    - Install ET-partition: pip install -e .
    - Have test data in data/test_site/

Usage:
    python examples/python_api_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime


def example_uwue_api():
    """
    示例：直接使用 uWUE 核心 API
    Example: Direct usage of uWUE core API
    """
    print("\n" + "=" * 80)
    print("示例1：uWUE 核心 API 使用 / Example 1: uWUE Core API Usage")
    print("=" * 80)

    # Import core functions
    from methods.uwue import zhou_part, zhouFlags, bigleaf, build_dataset_modified

    # 1. Load data using pandas
    data_path = project_root / "data" / "test_site" / \
                "FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3" / \
                "FLX_FI-Hyy_FLUXNET2015_FULLSET_HH_2008-2010_1-3.csv"

    print(f"\n📂 加载数据 / Loading data from: {data_path.name}")

    # Use the preprocessing function to build dataset (handles FLUXNET format)
    ec = build_dataset_modified(str(data_path))
    print(f"✅ 数据加载完成 / Data loaded: {len(ec.time)} time steps")

    # 2. Prepare data for uWUE calculation
    nStepsPerDay = 48  # Half-hourly data
    hourlyMask = xr.DataArray(
        np.ones(ec.time.shape).astype(bool),
        coords=[ec.time],
        dims=['time']
    )

    # Calculate ET from latent heat flux
    ec['ET'] = bigleaf.LE_to_ET(ec.LE, ec.TA) * 60 * 60 * (24 / nStepsPerDay)
    ec['ET'] = ec['ET'].assign_attrs(long_name='evapotranspiration', units='mm per timestep')

    # Fill missing net radiation
    missing_netrad = np.isnan(ec['NETRAD'])
    ec['NETRAD'][missing_netrad] = ec['LE'][missing_netrad] + ec['H'][missing_netrad] + ec['G'][missing_netrad]

    # Calculate potential ET
    PET, _ = bigleaf.PET(ec.TA, ec.PA, ec.NETRAD, G=ec.G, S=None,
                         alpha=1.26, missing_G_as_NA=False, missing_S_as_NA=False)
    ec['PET'] = PET * 60 * 60 * (24 / nStepsPerDay)

    # 3. Calculate masks for Zhou partitioning
    print("\n🔍 计算数据质量掩码 / Calculating quality masks...")
    uWUEa_Mask, uWUEp_Mask = zhouFlags(ec, nStepsPerDay, hourlyMask, GPPvariant='GPP_NT')
    print(f"   uWUEa mask: {uWUEa_Mask.sum()} valid points")
    print(f"   uWUEp mask: {uWUEp_Mask.sum()} valid points")

    # 4. Call the core zhou_part function
    print("\n🧮 执行 uWUE 拆分计算 / Running uWUE partitioning...")

    # Calculate GPP * sqrt(VPD)
    GxV = ec['GPP_NT'].values * np.sqrt(ec['VPD'].values)

    # Run Zhou partitioning
    uWUEp, zhou_T, zhou_T_8day = zhou_part(
        ET=ec['ET'].values,
        GxV=GxV,
        uWUEa_Mask=uWUEa_Mask,
        uWUEp_Mask=uWUEp_Mask,
        nStepsPerDay=nStepsPerDay,
        hourlyMask=hourlyMask,
        rho=0.95
    )

    print(f"✅ uWUE 拆分完成 / uWUE partitioning complete")
    print(f"   uWUEp (potential WUE): {uWUEp:.4f}")

    # 5. Create results DataFrame
    # Aggregate to daily for easier viewing
    ET_daily = ec['ET'].values.reshape(-1, nStepsPerDay).sum(axis=1)
    E_daily = ET_daily - zhou_T

    results_df = pd.DataFrame({
        'date': pd.to_datetime(ec.time.values[::48].astype(str)),
        'ET': ET_daily,
        'T': zhou_T,
        'E': E_daily,
        'T_ET_ratio': zhou_T / (ET_daily + 1e-6)
    })

    # 6. Display results
    print("\n📊 结果预览 / Results preview (first 10 days):")
    print(results_df.head(10).to_string(index=False))

    print(f"\n📈 统计摘要 / Summary statistics:")
    print(f"   平均蒸腾 / Mean T: {results_df['T'].mean():.2f} mm/day")
    print(f"   平均蒸发 / Mean E: {results_df['E'].mean():.2f} mm/day")
    print(f"   平均 T/ET 比率 / Mean T/ET ratio: {results_df['T_ET_ratio'].mean():.2f}")

    return results_df


def example_tea_api():
    """
    示例：直接使用 TEA 核心 API
    Example: Direct usage of TEA core API
    """
    print("\n" + "=" * 80)
    print("示例2：TEA 核心 API 使用 / Example 2: TEA Core API Usage")
    print("=" * 80)

    # Import core functions
    from methods.tea import simplePartition

    # 1. Load data using pandas
    data_path = project_root / "data" / "test_site" / \
                "FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3" / \
                "FLX_FI-Hyy_FLUXNET2015_FULLSET_HH_2008-2010_1-3.csv"

    print(f"\n📂 加载数据 / Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)

    # 2. Prepare required variables for TEA
    # TEA requires: timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u
    print("\n🔄 准备 TEA 所需变量 / Preparing variables for TEA...")

    # Convert timestamp
    timestamp = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')

    # Extract required variables with FLUXNET naming
    ET = df['LE_F_MDS'].values * 0.035 / 1000  # Convert W/m2 to mm/hh (approximate)
    GPP = df['GPP_NT_VUT_REF'].values  # umol/m2/s
    Tair = df['TA_F'].values  # deg C
    RH = df['RH'].values  # %
    VPD = df['VPD_F'].values  # hPa
    precip = df['P_F'].values  # mm/hh
    Rg = df['SW_IN_F'].values  # W/m2
    Rg_pot = df['SW_IN_POT'].values  # W/m2
    u = df['WS_F'].values  # m/s

    # Handle missing values
    print(f"   数据点数 / Data points: {len(timestamp)}")
    print(f"   时间范围 / Time range: {timestamp.min()} to {timestamp.max()}")

    # 3. Call the core simplePartition function
    print("\n🧮 执行 TEA 拆分计算 / Running TEA partitioning...")
    print("   (这可能需要几分钟 / This may take a few minutes...)")

    try:
        TEA_T, TEA_E, TEA_WUE = simplePartition(
            timestamp=timestamp.values,
            ET=ET,
            GPP=GPP,
            RH=RH,
            Rg=Rg,
            Rg_pot=Rg_pot,
            Tair=Tair,
            VPD=VPD,
            precip=precip,
            u=u,
            qualityFlag=None  # Will use internal quality flags
        )

        print(f"✅ TEA 拆分完成 / TEA partitioning complete")

        # 4. Create results DataFrame
        results_df = pd.DataFrame({
            'datetime': timestamp,
            'TEA_T': TEA_T,  # mm/hh
            'TEA_E': TEA_E,  # mm/hh
            'TEA_WUE': TEA_WUE,  # g C / kg H2O
            'ET': ET
        })

        # Convert to daily for display
        results_df['date'] = results_df['datetime'].dt.date
        daily_results = results_df.groupby('date').agg({
            'TEA_T': lambda x: x.sum() * 48,  # Convert to mm/day
            'TEA_E': lambda x: x.sum() * 48,
            'TEA_WUE': 'mean',
            'ET': lambda x: x.sum() * 48
        }).reset_index()
        daily_results['date'] = pd.to_datetime(daily_results['date'])

        # 5. Display results
        print("\n📊 结果预览 / Results preview (first 10 days):")
        print(daily_results.head(10).to_string(index=False))

        print(f"\n📈 统计摘要 / Summary statistics:")
        print(f"   平均蒸腾 / Mean T: {daily_results['TEA_T'].mean():.2f} mm/day")
        print(f"   平均蒸发 / Mean E: {daily_results['TEA_E'].mean():.2f} mm/day")
        print(f"   平均 WUE / Mean WUE: {daily_results['TEA_WUE'].mean():.2f} g C/kg H2O")

        return daily_results

    except Exception as e:
        print(f"❌ TEA 计算失败 / TEA calculation failed: {e}")
        print("   这可能是由于数据质量或缺失值导致的")
        print("   This may be due to data quality or missing values")
        import traceback
        traceback.print_exc()
        return None


def example_perez_priego_api():
    """
    示例：直接使用 Perez-Priego 核心 API
    Example: Direct usage of Perez-Priego core API
    """
    print("\n" + "=" * 80)
    print("示例3：Perez-Priego 核心 API 使用 / Example 3: Perez-Priego Core API Usage")
    print("=" * 80)

    # Import core functions
    from methods.perez_priego import (
        calculate_chi_o,
        calculate_WUE_o,
        optimal_parameters,
        transpiration_model,
        photos_model
    )

    # 1. Load data using pandas
    data_path = project_root / "data" / "test_site" / \
                "FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3" / \
                "FLX_FI-Hyy_FLUXNET2015_FULLSET_HH_2008-2010_1-3.csv"

    print(f"\n📂 加载数据 / Loading data from: {data_path.name}")
    df = pd.read_csv(data_path)

    # 2. Prepare required variables
    print("\n🔄 准备 Perez-Priego 所需变量 / Preparing variables for Perez-Priego...")

    # Filter daytime data with valid values
    df_filtered = df[
        (df['SW_IN_F'] > 50) &  # Daytime
        (df['GPP_NT_VUT_MEAN'].notna()) &
        (df['VPD_F'].notna()) &
        (df['TA_F'].notna()) &
        (df['H_F_MDS'].notna())
    ].copy()

    print(f"   过滤后数据点数 / Filtered data points: {len(df_filtered)}")

    if len(df_filtered) < 100:
        print("❌ 数据点不足，无法运行 Perez-Priego 方法")
        print("   Not enough data points to run Perez-Priego method")
        return None

    # 3. Calculate chi_o and WUE_o
    print("\n🧮 计算 chi_o 和 WUE_o / Calculating chi_o and WUE_o...")

    # Site parameters for FI-Hyy
    z = 0.181  # Elevation in km (181m)
    c_coef = 0.41  # C coefficient for ENF (Evergreen Needleleaf Forest)

    try:
        chi_o = calculate_chi_o(
            df_filtered,
            col_photos='GPP_NT_VUT_MEAN',
            col_vpd='VPD_F',
            col_tair='TA_F',
            c_coef=c_coef,
            z=z
        )

        WUE_o = calculate_WUE_o(
            df_filtered,
            col_photos='GPP_NT_VUT_MEAN',
            col_vpd='VPD_F',
            col_tair='TA_F',
            c_coef=c_coef,
            z=z
        )

        print(f"✅ chi_o = {chi_o:.4f}")
        print(f"✅ WUE_o = {WUE_o:.4f} g C/kg H2O")

        # 4. Optimize parameters (this uses MCMC and may take time)
        print("\n🔬 优化模型参数 / Optimizing model parameters...")
        print("   (这可能需要30秒 / This may take ~30 seconds...)")

        # Parameter bounds
        par_lower = [10, 0.01, 10, 0.1]  # [a1, D0, Topt, beta]
        par_upper = [200, 0.5, 35, 2.0]

        optimal_par = optimal_parameters(
            par_lower=par_lower,
            par_upper=par_upper,
            data=df_filtered,
            Chi_o=chi_o,
            WUE_o=WUE_o
        )

        print(f"✅ 最优参数 / Optimal parameters:")
        print(f"   a1 (light response): {optimal_par[0]:.2f}")
        print(f"   D0 (VPD sensitivity): {optimal_par[1]:.4f}")
        print(f"   Topt (optimal T): {optimal_par[2]:.2f} °C")
        print(f"   beta (stress param): {optimal_par[3]:.4f}")

        # 5. Calculate transpiration
        print("\n🌿 计算蒸腾量 / Calculating transpiration...")

        T_modeled = transpiration_model(optimal_par, df_filtered, chi_o)

        # Create results
        results_df = pd.DataFrame({
            'TIMESTAMP_START': pd.to_datetime(df_filtered['TIMESTAMP_START'], format='%Y%m%d%H%M'),
            'transpiration': T_modeled,  # mm/hh
            'GPP': df_filtered['GPP_NT_VUT_MEAN'].values,
            'VPD': df_filtered['VPD_F'].values,
            'Tair': df_filtered['TA_F'].values
        })

        # Add evaporation (E = ET - T)
        # Note: Would need ET from the dataset

        # Convert to daily
        results_df['date'] = results_df['TIMESTAMP_START'].dt.date
        daily_results = results_df.groupby('date').agg({
            'transpiration': lambda x: x.sum() * 48,  # Convert to mm/day
            'GPP': 'mean',
            'VPD': 'mean',
            'Tair': 'mean'
        }).reset_index()
        daily_results['date'] = pd.to_datetime(daily_results['date'])

        # 6. Display results
        print("\n📊 结果预览 / Results preview (first 10 days):")
        print(daily_results.head(10).to_string(index=False))

        print(f"\n📈 统计摘要 / Summary statistics:")
        print(f"   平均蒸腾 / Mean T: {daily_results['transpiration'].mean():.2f} mm/day")

        return daily_results

    except Exception as e:
        print(f"❌ Perez-Priego 计算失败 / Perez-Priego calculation failed: {e}")
        print("   这可能是由于 MCMC 优化超时或数据问题")
        print("   This may be due to MCMC timeout or data issues")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数：运行所有 API 示例
    Main function: Run all API examples
    """
    print("=" * 80)
    print("ET 蒸散发拆分 - Python API 使用示例")
    print("ET Partition - Python API Usage Examples")
    print("=" * 80)
    print("\n这些示例展示如何直接调用核心算法，而不使用批处理层。")
    print("These examples show how to call core algorithms directly without batch processing.")

    results = {}

    # Run examples
    try:
        results['uwue'] = example_uwue_api()
    except Exception as e:
        print(f"\n❌ uWUE 示例失败 / uWUE example failed: {e}")

    try:
        results['tea'] = example_tea_api()
    except Exception as e:
        print(f"\n❌ TEA 示例失败 / TEA example failed: {e}")

    try:
        results['perez_priego'] = example_perez_priego_api()
    except Exception as e:
        print(f"\n❌ Perez-Priego 示例失败 / Perez-Priego example failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("所有 API 示例运行完成！ / All API examples completed!")
    print("=" * 80)
    print("\n💡 使用提示 / Usage Tips:")
    print("   - 这些核心函数可以集成到你自己的数据流水线中")
    print("   - These core functions can be integrated into your own data pipeline")
    print("   - 你可以自定义预处理步骤和参数")
    print("   - You can customize preprocessing steps and parameters")
    print("   - 对于批量处理，使用 batch.py 模块更方便")
    print("   - For batch processing, use the batch.py modules instead")

    return 0


if __name__ == "__main__":
    sys.exit(main())
