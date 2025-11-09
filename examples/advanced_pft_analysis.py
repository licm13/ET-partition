#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ET Partition - Advanced PFT Analysis Examples
==============================================

This script demonstrates advanced Plant Functional Type (PFT) scenario analysis
including synthetic PFT-based stress tests with comprehensive metrics and visualizations.

Requirements:
    - Install ET-partition: pip install -e .
    - analysis package with PFT scenarios

Usage:
    python examples/advanced_pft_analysis.py
    python examples/advanced_pft_analysis.py --examples advanced
    python examples/advanced_pft_analysis.py --examples comprehensive
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    PFT_ENF, PFT_DBF, PFT_GRA, PFT_CSH,
    PFTScenario, PartitionComparison,
    get_pft_scenario, list_pft_scenarios,
    PREDEFINED_PFT_SCENARIOS,
    visualization
)
from utils.plotting import save_figure, print_performance_summary


def advanced_pft_comparison():
    """
    示例5：高级PFT对比分析

    Advanced comparison including synthetic PFT-based stress tests with
    comprehensive metrics and visualizations.
    """
    print("\n" + "=" * 80)
    print("示例5：高级PFT情景对比 / Example 5: Advanced PFT Scenario Comparison")
    print("=" * 80)

    print("\n可用PFT场景 / Available PFT scenarios:")
    print(", ".join(list_pft_scenarios()))

    # Select scenarios for comparison
    scenarios = [PFT_ENF, PFT_DBF, PFT_GRA, PFT_CSH]

    print(f"\n运行{len(scenarios)}个PFT场景的对比分析...")
    print(f"Running comparison across {len(scenarios)} PFT scenarios...")

    # Run comprehensive comparison
    comparison = PartitionComparison(
        scenarios,
        n_days=180,
        seed=42,
        include_seasonal_analysis=True,
        include_stress_analysis=True
    )
    results = comparison.run()
    aggregate_results_df = comparison.results_to_dataframe(results)
    performance_summary_df = comparison.aggregate_metrics(aggregate_results_df)

    # Create output directory
    output_dir = project_root / "outputs" / "advanced_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_path = output_dir / "pft_method_diagnostics.csv"
    summary_path = output_dir / "pft_method_summary.csv"
    aggregate_results_df.to_csv(detailed_path, index=False)
    performance_summary_df.to_csv(summary_path, index=False)

    print("\n详细的情景/方法指标保存至 / Detailed metrics saved to:", detailed_path)
    print("跨情景汇总保存至 / Scenario summary saved to:", summary_path)

    # Display summary statistics
    print("\n跨情景平均性能指标 / Cross-scenario mean performance metrics:")
    print("=" * 80)
    print_performance_summary(performance_summary_df)

    # Generate visualizations
    print("\n生成可视化图表 / Generating visualizations...")

    try:
        # Heatmap of RMSE_T
        heatmap_fig = visualization.plot_performance_heatmap(
            aggregate_results_df, metric="rmse_T", title="Transpiration RMSE across PFTs"
        )
        save_figure(heatmap_fig, output_dir / "heatmap_rmse_T.png")

        # Bar plots of multiple metrics
        method_bars_fig = visualization.plot_method_comparison_bars(performance_summary_df)
        save_figure(method_bars_fig, output_dir / "method_comparison_bars.png")

        # Time series for one scenario
        from analysis import run_method_emulators as run_emulators
        scenario_name = "ENF"
        synthetic_data = comparison.get_synthetic_data(scenario_name)
        if synthetic_data is not None:
            emulator_estimates = run_emulators(synthetic_data)
            timeseries_fig = visualization.plot_time_series_comparison(
                synthetic_data, emulator_estimates, scenario_name, n_days=30
            )
            save_figure(timeseries_fig, output_dir / f"timeseries_{scenario_name}.png")

            # Stress response analysis
            stress_response_fig = visualization.plot_stress_response(
                synthetic_data, emulator_estimates, scenario_name
            )
            save_figure(stress_response_fig, output_dir / f"stress_response_{scenario_name}.png")

    except OSError as e:
        print(f"  可视化过程中出现I/O错误 / I/O error during visualization: {e}")
    except Exception as e:
        print(f"  可视化过程中出现警告 / Warning during visualization: {e}")
        import traceback
        traceback.print_exc()

    # Performance ranking
    print("\n方法性能排名 (按RMSE_T) / Method ranking by RMSE_T:")
    ranking_df = comparison.performance_ranking(aggregate_results_df, metric="rmse_T")
    for i, row in enumerate(ranking_df.itertuples(index=False), start=1):
        print(f"  {i}. {row.method}: {row.rmse_T:.3f}")

    print(f"\n所有输出文件保存至 / All outputs saved to: {output_dir}")


def comprehensive_pft_analysis():
    """
    示例6：全面的多PFT场景分析

    Comprehensive analysis across all predefined PFT scenarios with
    detailed diagnostics and visualizations.
    """
    print("\n" + "=" * 80)
    print("示例6：全面PFT分析 / Example 6: Comprehensive Multi-PFT Analysis")
    print("=" * 80)

    # Use subset of PFTs for faster execution
    scenarios = [s for s in PREDEFINED_PFT_SCENARIOS if s.name in ['ENF', 'DBF', 'GRA', 'CSH', 'CRO']]

    print(f"\n分析{len(scenarios)}个PFT场景: {[s.name for s in scenarios]}")
    print(f"Analyzing {len(scenarios)} PFT scenarios: {[s.name for s in scenarios]}")

    # Run extended comparison
    comparison = PartitionComparison(
        scenarios,
        n_days=365,  # Full year
        seed=2024,
        include_seasonal_analysis=True,
        include_stress_analysis=True
    )

    print("\n运行模拟（这可能需要一些时间）...")
    print("Running simulations (this may take a while)...")

    results = comparison.run()
    aggregate_results_df = comparison.results_to_dataframe(results)
    performance_summary_df = comparison.aggregate_metrics(aggregate_results_df)

    # Create output directory
    output_dir = project_root / "outputs" / "comprehensive_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all results
    aggregate_results_df.to_csv(output_dir / "full_results.csv", index=False)
    performance_summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Analyze seasonal performance if available
    seasonal_cols = [col for col in aggregate_results_df.columns if 'rmse_T_' in col and col != 'rmse_T']
    if seasonal_cols:
        print("\n季节性性能分析 / Seasonal performance analysis:")
        for season in ['spring', 'summer', 'fall', 'winter']:
            col_name = f'rmse_T_{season}'
            if col_name in aggregate_results_df.columns:
                season_mean = aggregate_results_df.groupby('method')[col_name].mean()
                print(f"\n{season.capitalize()}:")
                for method, value in season_mean.items():
                    print(f"  {method}: {value:.3f}")

        # Plot seasonal heatmap
        try:
            fig = visualization.plot_seasonal_performance(aggregate_results_df)
            save_figure(fig, output_dir / "seasonal_performance.png")
            print(f"\n季节性能图已保存 / Seasonal performance plot saved")
        except OSError as e:
            print(f"无法生成季节性能图 (I/O error) / Cannot generate seasonal plot: {e}")
        except Exception as e:
            print(f"无法生成季节性能图 / Cannot generate seasonal plot: {e}")
            import traceback
            traceback.print_exc()

    # Generate comprehensive visualizations
    print("\n生成综合可视化图表...")
    print("Generating comprehensive visualizations...")

    # Multiple metric heatmaps
    metrics_to_plot = ['rmse_T', 'rmse_E', 'correlation_T', 'nse_T', 'kge_T']
    for metric in metrics_to_plot:
        if metric in aggregate_results_df.columns:
            try:
                fig = visualization.plot_performance_heatmap(aggregate_results_df, metric=metric)
                save_figure(fig, output_dir / f"heatmap_{metric}.png")
            except Exception:
                pass

    print(f"\n综合分析完成！所有输出保存至 / Comprehensive analysis complete! All outputs saved to:")
    print(f"  {output_dir}")

    # Print best method for each PFT
    print("\n各PFT最佳方法 (按RMSE_T) / Best method for each PFT (by RMSE_T):")
    for scenario_name in aggregate_results_df['scenario'].unique():
        scenario_data = aggregate_results_df[aggregate_results_df['scenario'] == scenario_name]
        best_method = scenario_data.loc[scenario_data['rmse_T'].idxmin(), 'method']
        best_rmse = scenario_data['rmse_T'].min()
        print(f"  {scenario_name}: {best_method} (RMSE = {best_rmse:.3f})")


def main():
    """
    主函数：运行高级PFT分析示例
    Main function: Run advanced PFT analysis examples
    """
    print("=" * 80)
    print("ET蒸散发拆分 - 高级PFT分析示例")
    print("ET Partition - Advanced PFT Analysis Examples")
    print("=" * 80)

    # Check command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run advanced PFT analysis examples")
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=["advanced", "comprehensive", "all"],
        default=["all"],
        help="Which examples to run (default: all)"
    )

    # Parse args, handling both direct execution and module execution
    try:
        args = parser.parse_args()
    except:
        # If parsing fails (e.g., when imported), use defaults
        class Args:
            examples = ["all"]
        args = Args()

    try:
        run_all = "all" in args.examples

        if run_all or "advanced" in args.examples:
            advanced_pft_comparison()

        if run_all or "comprehensive" in args.examples:
            comprehensive_pft_analysis()

        print("\n" + "=" * 80)
        print("所有高级分析示例运行完成！ / All advanced analysis examples completed!")
        print("=" * 80)
        print("\n使用提示 / Usage tips:")
        print("  运行特定示例 / Run specific examples:")
        print("    python examples/advanced_pft_analysis.py --examples advanced")
        print("    python examples/advanced_pft_analysis.py --examples comprehensive")

    except Exception as e:
        print(f"\n错误 / Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
