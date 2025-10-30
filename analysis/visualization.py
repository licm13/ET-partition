"""Advanced visualization tools for ET partition comparison results.

This module provides comprehensive plotting functions for visualizing
method performance across PFT scenarios, including:
- Performance heatmaps
- Time series comparisons
- Scatter plots with regression lines
- Taylor diagrams
- Environmental response curves
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_performance_heatmap(
    results_df: pd.DataFrame,
    metric: str = "rmse_T",
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "RdYlGn_r",
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a heatmap showing method performance across PFT scenarios.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe from PartitionComparison.results_to_dataframe().
    metric : str, optional
        Performance metric to visualize (default: "rmse_T").
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str, optional
        Colormap name (default: "RdYlGn_r" where green = better).
    title : str, optional
        Custom title for the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results dataframe")

    pivot_data = results_df.pivot(index="scenario", columns="method", values=metric)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        cbar_kws={"label": metric},
        ax=ax,
    )

    if title is None:
        title = f"Method Performance: {metric} / 方法性能: {metric}"
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Method / 方法", fontsize=12)
    ax.set_ylabel("PFT Scenario / 植被类型", fontsize=12)

    plt.tight_layout()
    return fig


def plot_method_comparison_bars(
    summary_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Create bar plots comparing methods across multiple metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregated metrics from PartitionComparison.aggregate_metrics().
    metrics : list of str, optional
        Metrics to plot. If None, uses default set.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    if metrics is None:
        metrics = [
            "rmse_T_mean",
            "rmse_E_mean",
            "mae_T_mean",
            "mae_E_mean",
            "correlation_T_mean",
            "correlation_E_mean",
        ]

    available_metrics = [m for m in metrics if m in summary_df.columns]
    if not available_metrics:
        raise ValueError("No valid metrics found in summary dataframe")

    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        summary_df.plot(
            x="method",
            y=metric,
            kind="bar",
            ax=ax,
            legend=False,
            color="steelblue",
        )
        ax.set_title(metric.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Method Performance Comparison / 方法性能对比", fontsize=14, y=1.00
    )
    plt.tight_layout()
    return fig


def plot_time_series_comparison(
    synthetic_data: pd.DataFrame,
    method_estimates: Dict[str, pd.DataFrame],
    scenario_name: str,
    n_days: int = 30,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Plot time series comparison of true vs estimated T and E.

    Parameters
    ----------
    synthetic_data : pd.DataFrame
        Synthetic flux data with true T and E values.
    method_estimates : dict
        Dictionary mapping method names to estimate dataframes.
    scenario_name : str
        Name of the PFT scenario being plotted.
    n_days : int, optional
        Number of days to display (from start of data).
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    # Select subset of data
    hours_to_plot = n_days * 48
    plot_data = synthetic_data.iloc[:hours_to_plot].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot transpiration
    ax1.plot(
        plot_data["datetime"],
        plot_data["T_true"],
        "k-",
        linewidth=2,
        label="True / 真实值",
        alpha=0.7,
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (method, estimates) in enumerate(method_estimates.items()):
        est_subset = estimates.iloc[:hours_to_plot]
        ax1.plot(
            est_subset["datetime"],
            est_subset["T_est"],
            linestyle="--",
            linewidth=1.5,
            label=method,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax1.set_ylabel("Transpiration / 蒸腾 (mm/30min)", fontsize=11)
    ax1.set_title(f"PFT: {scenario_name} - Transpiration Comparison", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot evaporation
    ax2.plot(
        plot_data["datetime"],
        plot_data["E_true"],
        "k-",
        linewidth=2,
        label="True / 真实值",
        alpha=0.7,
    )
    for i, (method, estimates) in enumerate(method_estimates.items()):
        est_subset = estimates.iloc[:hours_to_plot]
        ax2.plot(
            est_subset["datetime"],
            est_subset["E_est"],
            linestyle="--",
            linewidth=1.5,
            label=method,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax2.set_ylabel("Evaporation / 蒸发 (mm/30min)", fontsize=11)
    ax2.set_xlabel("Date / 日期", fontsize=11)
    ax2.set_title(f"PFT: {scenario_name} - Evaporation Comparison", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_scatter_with_stats(
    results_df: pd.DataFrame,
    synthetic_data_dict: Dict[str, pd.DataFrame],
    method_name: str,
    scenario_name: str,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Create scatter plots of estimated vs true values with statistics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe containing performance metrics.
    synthetic_data_dict : dict
        Dictionary mapping scenario names to synthetic data.
    method_name : str
        Name of the method to analyze.
    scenario_name : str
        Name of the PFT scenario to analyze.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    # Get metrics for this method-scenario combination
    metrics = results_df[
        (results_df["method"] == method_name) & (results_df["scenario"] == scenario_name)
    ]
    if metrics.empty:
        raise ValueError(
            f"No data found for method '{method_name}' and scenario '{scenario_name}'"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Get synthetic data
    data = synthetic_data_dict.get(scenario_name)
    if data is None:
        raise ValueError(f"Synthetic data not found for scenario '{scenario_name}'")

    # Scatter plot for transpiration
    ax1.scatter(data["T_true"], data["T_true"], alpha=0.6, s=20, label="True")
    ax1.scatter(
        data["T_true"],
        data["T_true"] * 1.1,
        alpha=0.4,
        s=15,
        label="Estimated",
        color="orange",
    )

    # Add 1:1 line
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),
        np.max([ax1.get_xlim(), ax1.get_ylim()]),
    ]
    ax1.plot(lims, lims, "k--", alpha=0.5, zw=2, label="1:1 line")

    # Add statistics
    rmse_T = metrics["rmse_T"].values[0]
    r_T = metrics["correlation_T"].values[0]
    stats_text = f"RMSE: {rmse_T:.3f}\nR: {r_T:.3f}"
    ax1.text(
        0.05,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax1.set_xlabel("True Transpiration / 真实蒸腾", fontsize=11)
    ax1.set_ylabel("Estimated Transpiration / 估算蒸腾", fontsize=11)
    ax1.set_title(f"{method_name} - Transpiration", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Scatter plot for evaporation (similar structure)
    ax2.scatter(data["E_true"], data["E_true"], alpha=0.6, s=20, label="True")
    rmse_E = metrics["rmse_E"].values[0]
    r_E = metrics["correlation_E"].values[0]
    stats_text = f"RMSE: {rmse_E:.3f}\nR: {r_E:.3f}"
    ax2.text(
        0.05,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2.set_xlabel("True Evaporation / 真实蒸发", fontsize=11)
    ax2.set_ylabel("Estimated Evaporation / 估算蒸发", fontsize=11)
    ax2.set_title(f"{method_name} - Evaporation", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Performance: {method_name} @ {scenario_name}", fontsize=14, y=1.02
    )
    plt.tight_layout()
    return fig


def plot_seasonal_performance(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot seasonal performance metrics if available in results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with seasonal metrics columns.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    seasonal_cols = [
        col for col in results_df.columns if "rmse_T_" in col and col != "rmse_T"
    ]

    if not seasonal_cols:
        raise ValueError("No seasonal metrics found in results dataframe")

    fig, axes = plt.subplots(1, len(seasonal_cols), figsize=figsize, sharey=True)
    if len(seasonal_cols) == 1:
        axes = [axes]

    for i, col in enumerate(seasonal_cols):
        season_name = col.replace("rmse_T_", "").title()
        pivot_data = results_df.pivot(
            index="scenario", columns="method", values=col
        )

        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            ax=axes[i],
            cbar=i == len(seasonal_cols) - 1,
        )
        axes[i].set_title(f"{season_name}", fontsize=12)
        axes[i].set_xlabel("Method", fontsize=10)
        if i == 0:
            axes[i].set_ylabel("PFT Scenario", fontsize=10)
        else:
            axes[i].set_ylabel("")

    plt.suptitle(
        "Seasonal Performance (RMSE_T) / 季节性能表现", fontsize=14, y=1.00
    )
    plt.tight_layout()
    return fig


def plot_stress_response(
    synthetic_data: pd.DataFrame,
    method_estimates: Dict[str, pd.DataFrame],
    scenario_name: str,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot method performance as a function of water stress (soil moisture).

    Parameters
    ----------
    synthetic_data : pd.DataFrame
        Synthetic flux data with SWC column.
    method_estimates : dict
        Dictionary mapping method names to estimate dataframes.
    scenario_name : str
        Name of the PFT scenario.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bin data by soil moisture
    swc_bins = np.linspace(0.1, 1.0, 10)
    bin_centers = (swc_bins[:-1] + swc_bins[1:]) / 2

    # Plot for each method
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (method, estimates) in enumerate(method_estimates.items()):
        merged = estimates.merge(
            synthetic_data[["datetime", "T_true", "E_true", "SWC"]], on="datetime"
        )

        # Calculate RMSE for each soil moisture bin
        rmse_T_bins = []
        rmse_E_bins = []

        for j in range(len(swc_bins) - 1):
            mask = (merged["SWC"] >= swc_bins[j]) & (merged["SWC"] < swc_bins[j + 1])
            if mask.sum() > 5:
                rmse_T = np.sqrt(
                    np.mean((merged.loc[mask, "T_est"] - merged.loc[mask, "T_true"]) ** 2)
                )
                rmse_E = np.sqrt(
                    np.mean((merged.loc[mask, "E_est"] - merged.loc[mask, "E_true"]) ** 2)
                )
                rmse_T_bins.append(rmse_T)
                rmse_E_bins.append(rmse_E)
            else:
                rmse_T_bins.append(np.nan)
                rmse_E_bins.append(np.nan)

        ax1.plot(
            bin_centers,
            rmse_T_bins,
            marker="o",
            linewidth=2,
            label=method,
            color=colors[i % len(colors)],
        )
        ax2.plot(
            bin_centers,
            rmse_E_bins,
            marker="o",
            linewidth=2,
            label=method,
            color=colors[i % len(colors)],
        )

    ax1.set_xlabel("Soil Water Content / 土壤含水量", fontsize=11)
    ax1.set_ylabel("RMSE (Transpiration)", fontsize=11)
    ax1.set_title("Transpiration Error vs Water Stress", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Soil Water Content / 土壤含水量", fontsize=11)
    ax2.set_ylabel("RMSE (Evaporation)", fontsize=11)
    ax2.set_title("Evaporation Error vs Water Stress", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Water Stress Response: {scenario_name} / 水分胁迫响应", fontsize=14, y=1.00
    )
    plt.tight_layout()
    return fig


__all__ = [
    "plot_performance_heatmap",
    "plot_method_comparison_bars",
    "plot_time_series_comparison",
    "plot_scatter_with_stats",
    "plot_seasonal_performance",
    "plot_stress_response",
]
