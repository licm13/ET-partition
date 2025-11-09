"""
Plotting utilities for ET partition analysis.

This module provides helper functions for saving figures and printing
performance summaries, reducing code duplication in analysis scripts.
"""

from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd


def save_figure(
    fig,
    output_path: Union[Path, str],
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save a matplotlib figure to a file with consistent settings.
    
    Creates parent directories if needed, saves the figure, prints a
    confirmation message, and closes the figure to free memory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_path : Path or str
        Path where the figure should be saved.
    dpi : int, default=300
        Resolution in dots per inch.
    bbox_inches : str, default='tight'
        Bounding box setting for saved figure.
    
    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"  已保存 / Saved: {output_path.name}")
    plt.close(fig)


def print_performance_summary(performance_summary_df: pd.DataFrame) -> None:
    """
    Print performance summary statistics from aggregated metrics DataFrame.
    
    Iterates over methods and prints RMSE, NSE, KGE, and correlation metrics
    with mean and standard deviation across scenarios.
    
    Parameters
    ----------
    performance_summary_df : pd.DataFrame
        DataFrame with columns: method, rmse_T_mean, rmse_T_std, rmse_E_mean,
        rmse_E_std, nse_T_mean, nse_T_std, kge_T_mean, kge_T_std,
        correlation_T_mean, correlation_T_std.
    
    Returns
    -------
    None
    """
    for row in performance_summary_df.itertuples(index=False):
        print(f"\n{row.method}:")
        print(f"  RMSE_T: {row.rmse_T_mean:.3f} ± {row.rmse_T_std:.3f}")
        print(f"  RMSE_E: {row.rmse_E_mean:.3f} ± {row.rmse_E_std:.3f}")
        print(f"  NSE_T:  {row.nse_T_mean:.3f} ± {row.nse_T_std:.3f}")
        print(f"  KGE_T:  {row.kge_T_mean:.3f} ± {row.kge_T_std:.3f}")
        print(f"  Corr_T: {row.correlation_T_mean:.3f} ± {row.correlation_T_std:.3f}")
