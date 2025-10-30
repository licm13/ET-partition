#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ET Partition - Basic Usage Examples
====================================

This script demonstrates basic usage of all three ET partitioning methods.

Requirements:
    - Install ET-partition: pip install -e .
    - Have test data in data/test_site/

Usage:
    python examples/basic_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis import PFTScenario, PartitionComparison


def example_uwue_method():
    """
    示例1：使用uWUE方法进行ET拆分

    Example 1: ET partitioning using uWUE method
    """
    print("\n" + "="*80)
    print("示例1：uWUE方法 / Example 1: uWUE Method")
    print("="*80)

    from methods.uwue.batch import uWUEBatchProcessor

    # 设置路径 / Setup paths
    base_path = project_root / "data" / "test_site"
    output_path = project_root / "outputs" / "example_uwue"

    # 初始化处理器 / Initialize processor
    print("初始化uWUE批处理器... / Initializing uWUE batch processor...")
    processor = uWUEBatchProcessor(
        base_path=str(base_path),
        output_path=str(output_path),
        create_plots=True
    )

    # 运行处理 / Run processing
    print("运行uWUE处理... / Running uWUE processing...")
    processor.run()

    # 读取结果 / Read results
    output_files = list(output_path.glob("*_uWUE_output.csv"))
    if output_files:
        result = pd.read_csv(output_files[0])
        print(f"\n结果预览 / Results preview ({output_files[0].name}):")
        print(result.head())
        print(f"\n蒸腾平均值 / Mean transpiration: {result['T'].mean():.2f} mm/day")
        print(f"蒸发平均值 / Mean evaporation: {result['E'].mean():.2f} mm/day")
        print(f"T/ET比率 / T/ET ratio: {result['T_ET_ratio'].mean():.2f}")

    print(f"\n输出文件保存至 / Outputs saved to: {output_path}")


def example_tea_method():
    """
    示例2：使用TEA方法进行ET拆分

    Example 2: ET partitioning using TEA method
    """
    print("\n" + "="*80)
    print("示例2：TEA方法 / Example 2: TEA Method")
    print("="*80)

    from methods.tea.batch import main as tea_main

    # 设置路径 / Setup paths
    base_path = project_root / "data" / "test_site"
    output_path = project_root / "outputs" / "example_tea"

    # 准备参数 / Prepare arguments
    class Args:
        base_path = str(base_path)
        output_path = str(output_path)
        pattern = r"FLX_.*_FLUXNET.*"

    # 运行处理 / Run processing
    print("运行TEA处理... / Running TEA processing...")
    tea_main(Args())

    # 读取结果 / Read results
    output_files = list(output_path.glob("*_TEA_results.csv"))
    if output_files:
        result = pd.read_csv(output_files[0])
        print(f"\n结果预览 / Results preview ({output_files[0].name}):")
        print(result.head())

        # 转换为日尺度 / Convert to daily
        result['T_mm_day'] = result['TEA_T'] * 48  # 48 half-hours per day
        result['E_mm_day'] = result['TEA_E'] * 48

        print(f"\n蒸腾平均值 / Mean transpiration: {result['T_mm_day'].mean():.2f} mm/day")
        print(f"蒸发平均值 / Mean evaporation: {result['E_mm_day'].mean():.2f} mm/day")
        print(f"WUE平均值 / Mean WUE: {result['TEA_WUE'].mean():.2f} g C/kg H2O")

    print(f"\n输出文件保存至 / Outputs saved to: {output_path}")


def example_perez_priego_method():
    """
    示例3：使用Perez-Priego方法进行ET拆分

    Example 3: ET partitioning using Perez-Priego method
    """
    print("\n" + "="*80)
    print("示例3：Perez-Priego方法 / Example 3: Perez-Priego Method")
    print("="*80)

    from methods.perez_priego.batch import main as pp_main

    # 设置路径 / Setup paths
    base_path = project_root / "data" / "test_site"
    output_path = project_root / "outputs" / "example_pp"

    # 准备参数 / Prepare arguments
    class Args:
        base_path = str(base_path)
        output_path = str(output_path)
        site_metadata = None
        default_altitude = 0.181  # FI-Hyy elevation: 181m = 0.181km

    # 运行处理 / Run processing
    print("运行Perez-Priego处理... / Running Perez-Priego processing...")
    pp_main(Args())

    # 读取结果 / Read results
    output_files = list(output_path.glob("*_pp_output.csv"))
    if output_files:
        result = pd.read_csv(output_files[0])
        print(f"\n结果预览 / Results preview ({output_files[0].name}):")
        print(result[['TIMESTAMP_START', 'transpiration', 'evaporation', 'T_ET_ratio']].head())

        # 转换为日尺度 / Convert to daily
        result['T_mm_day'] = result['transpiration'] * 48
        result['E_mm_day'] = result['evaporation'] * 48

        print(f"\n蒸腾平均值 / Mean transpiration: {result['T_mm_day'].mean():.2f} mm/day")
        print(f"蒸发平均值 / Mean evaporation: {result['E_mm_day'].mean():.2f} mm/day")
        print(f"T/ET比率 / T/ET ratio: {result['T_ET_ratio'].mean():.2f}")

    print(f"\n输出文件保存至 / Outputs saved to: {output_path}")


def compare_methods():
    """
    示例4：比较三种方法的结果

    Example 4: Compare results from all three methods
    """
    print("\n" + "="*80)
    print("示例4：方法比较 / Example 4: Method Comparison")
    print("="*80)

    # 定义输出路径 / Define output paths
    uwue_path = project_root / "outputs" / "example_uwue"
    tea_path = project_root / "outputs" / "example_tea"
    pp_path = project_root / "outputs" / "example_pp"

    # 检查文件是否存在 / Check if files exist
    uwue_files = list(uwue_path.glob("*_uWUE_output.csv"))
    tea_files = list(tea_path.glob("*_TEA_results.csv"))
    pp_files = list(pp_path.glob("*_pp_output.csv"))

    if not (uwue_files and tea_files and pp_files):
        print("请先运行前面的示例以生成结果文件")
        print("Please run previous examples first to generate result files")
        return

    # 读取uWUE结果 (日尺度) / Read uWUE results (daily)
    uwue_df = pd.read_csv(uwue_files[0])
    uwue_df['date'] = pd.to_datetime(uwue_df['date'])

    # 读取TEA结果并聚合到日尺度 / Read TEA results and aggregate to daily
    tea_df = pd.read_csv(tea_files[0])
    tea_df['datetime'] = pd.to_datetime(tea_df['datetime'])
    tea_df['date'] = tea_df['datetime'].dt.date
    tea_daily = tea_df.groupby('date').agg({
        'TEA_T': lambda x: x.sum() * 48,  # Convert to mm/day
        'TEA_E': lambda x: x.sum() * 48
    }).reset_index()
    tea_daily['date'] = pd.to_datetime(tea_daily['date'])

    # 读取Perez-Priego结果并聚合到日尺度 / Read PP results and aggregate to daily
    pp_df = pd.read_csv(pp_files[0])
    pp_df['TIMESTAMP_START'] = pd.to_datetime(pp_df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    pp_df['date'] = pp_df['TIMESTAMP_START'].dt.date
    pp_daily = pp_df.groupby('date').agg({
        'transpiration': lambda x: x.sum() * 48,
        'evaporation': lambda x: x.sum() * 48
    }).reset_index()
    pp_daily['date'] = pd.to_datetime(pp_daily['date'])

    # 合并数据 / Merge data
    merged = uwue_df[['date', 'T', 'E']].rename(columns={'T': 'uWUE_T', 'E': 'uWUE_E'})
    merged = merged.merge(tea_daily[['date', 'TEA_T', 'TEA_E']], on='date', how='outer')
    merged = merged.merge(pp_daily[['date', 'transpiration', 'evaporation']],
                         on='date', how='outer')
    merged = merged.rename(columns={'transpiration': 'PP_T', 'evaporation': 'PP_E'})

    # 计算统计信息 / Calculate statistics
    print("\n日均蒸腾量 (mm/day) / Daily Mean Transpiration (mm/day):")
    print(f"  uWUE:        {merged['uWUE_T'].mean():.2f}")
    print(f"  TEA:         {merged['TEA_T'].mean():.2f}")
    print(f"  Perez-Priego: {merged['PP_T'].mean():.2f}")

    print("\n日均蒸发量 (mm/day) / Daily Mean Evaporation (mm/day):")
    print(f"  uWUE:        {merged['uWUE_E'].mean():.2f}")
    print(f"  TEA:         {merged['TEA_E'].mean():.2f}")
    print(f"  Perez-Priego: {merged['PP_E'].mean():.2f}")

    # 绘制比较图 / Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 蒸腾比较 / Transpiration comparison
    axes[0].plot(merged['date'], merged['uWUE_T'], label='uWUE', alpha=0.7)
    axes[0].plot(merged['date'], merged['TEA_T'], label='TEA', alpha=0.7)
    axes[0].plot(merged['date'], merged['PP_T'], label='Perez-Priego', alpha=0.7)
    axes[0].set_ylabel('Transpiration (mm/day)')
    axes[0].set_title('蒸腾量比较 / Transpiration Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 蒸发比较 / Evaporation comparison
    axes[1].plot(merged['date'], merged['uWUE_E'], label='uWUE', alpha=0.7)
    axes[1].plot(merged['date'], merged['TEA_E'], label='TEA', alpha=0.7)
    axes[1].plot(merged['date'], merged['PP_E'], label='Perez-Priego', alpha=0.7)
    axes[1].set_ylabel('Evaporation (mm/day)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('蒸发量比较 / Evaporation Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表 / Save figure
    output_dir = project_root / "outputs" / "example_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "method_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n比较图保存至 / Comparison plot saved to: {fig_path}")

    # 保存合并数据 / Save merged data
    csv_path = output_dir / "method_comparison.csv"
    merged.to_csv(csv_path, index=False)
    print(f"比较数据保存至 / Comparison data saved to: {csv_path}")


def advanced_pft_comparison():
    """Advanced comparison including synthetic PFT-based stress tests."""

    print("\n" + "=" * 80)
    print("高级分析：PFT情景对比 / Advanced Analysis: PFT Scenario Comparison")
    print("=" * 80)

    scenarios = [
        PFTScenario(
            name="ENF",
            canopy_conductance=0.9,
            vpd_sensitivity=0.6,
            soil_evap_fraction=0.25,
            photosynthesis_efficiency=1.2,
            interception_ratio=0.35,
            noise_std=0.05,
            transpiration_bias=1.05,
        ),
        PFTScenario(
            name="DBF",
            canopy_conductance=0.75,
            vpd_sensitivity=0.45,
            soil_evap_fraction=0.35,
            photosynthesis_efficiency=1.0,
            interception_ratio=0.25,
            noise_std=0.07,
            transpiration_bias=0.95,
        ),
        PFTScenario(
            name="CSH",
            canopy_conductance=0.55,
            vpd_sensitivity=0.3,
            soil_evap_fraction=0.45,
            photosynthesis_efficiency=0.8,
            interception_ratio=0.18,
            noise_std=0.09,
            transpiration_bias=0.85,
        ),
    ]

    comparison = PartitionComparison(scenarios, n_days=120)
    results = comparison.run()
    result_df = comparison.results_to_dataframe(results)
    summary_df = comparison.aggregate_metrics(result_df)

    output_dir = project_root / "outputs" / "advanced_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = output_dir / "pft_method_diagnostics.csv"
    summary_path = output_dir / "pft_method_summary.csv"

    result_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("详细的情景/方法指标保存至 / Detailed metrics saved to:", detailed_path)
    print("跨情景汇总保存至 / Scenario summary saved to:", summary_path)

    print("\n跨情景平均RMSE (T, E) / Cross-scenario mean RMSE (T, E):")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['method']}: RMSE_T={row['rmse_T_mean']:.3f}, "
            f"RMSE_E={row['rmse_E_mean']:.3f}, "
            f"T/ET_RMSE={row['t_et_ratio_rmse']:.3f}"
        )

def main():
    """
    主函数：运行所有示例

    Main function: Run all examples
    """
    print("="*80)
    print("ET蒸散发拆分 - 基本使用示例")
    print("ET Partition - Basic Usage Examples")
    print("="*80)

    try:
        # 运行示例 / Run examples
        example_uwue_method()
        example_tea_method()
        example_perez_priego_method()
        compare_methods()
        advanced_pft_comparison()

        print("\n" + "="*80)
        print("所有示例运行完成！ / All examples completed!")
        print("="*80)

    except Exception as e:
        print(f"\n错误 / Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
