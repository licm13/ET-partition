"""High level comparison utilities for ET partitioning methods.

This module provides advanced tools for comparing different ET partitioning methods
across multiple Plant Functional Type (PFT) scenarios. It includes comprehensive
metrics for evaluating method performance under varying environmental conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .simulation import PFTScenario, generate_synthetic_flux_data, run_method_emulators


@dataclass
class ComparisonResult:
    """Summary statistics for a single method within one scenario.

    Attributes
    ----------
    scenario : str
        Name of the PFT scenario being evaluated.
    method : str
        Name of the ET partitioning method.
    bias_T : float
        Mean bias for transpiration (estimated - true).
    bias_E : float
        Mean bias for evaporation (estimated - true).
    mae_T : float
        Mean absolute error for transpiration.
    mae_E : float
        Mean absolute error for evaporation.
    rmse_T : float
        Root mean squared error for transpiration.
    rmse_E : float
        Root mean squared error for evaporation.
    relative_error_T : float
        Mean relative error for transpiration (RMSE / mean_true).
    relative_error_E : float
        Mean relative error for evaporation (RMSE / mean_true).
    correlation_T : float
        Pearson correlation coefficient between estimated and true transpiration.
    correlation_E : float
        Pearson correlation coefficient between estimated and true evaporation.
    t_et_ratio_error : float
        RMSE of T/ET ratio.
    nse_T : float
        Nash-Sutcliffe Efficiency for transpiration.
    nse_E : float
        Nash-Sutcliffe Efficiency for evaporation.
    kge_T : float
        Kling-Gupta Efficiency for transpiration.
    kge_E : float
        Kling-Gupta Efficiency for evaporation.
    seasonal_metrics : Dict[str, float]
        Additional metrics by season (optional).
    """

    scenario: str
    method: str
    bias_T: float
    bias_E: float
    mae_T: float
    mae_E: float
    rmse_T: float
    rmse_E: float
    relative_error_T: float
    relative_error_E: float
    correlation_T: float
    correlation_E: float
    t_et_ratio_error: float
    nse_T: float
    nse_E: float
    kge_T: float
    kge_E: float
    seasonal_metrics: Dict[str, float] = field(default_factory=dict)


class PartitionComparison:
    """Run synthetic experiments to compare ET partitioning approaches.

    This class orchestrates comprehensive benchmarking experiments to evaluate
    the performance of different ET partitioning methods across various PFT
    scenarios and environmental conditions.

    Parameters
    ----------
    scenarios : Iterable[PFTScenario]
        Collection of PFT scenarios to evaluate.
    n_days : int, optional
        Number of days to simulate for each scenario (default: 180).
    seed : int or None, optional
        Random seed for reproducibility (default: 42).
    include_seasonal_analysis : bool, optional
        Whether to compute seasonal breakdown metrics (default: True).
    include_stress_analysis : bool, optional
        Whether to analyze performance under water stress conditions (default: True).
    """

    def __init__(
        self,
        scenarios: Iterable[PFTScenario],
        n_days: int = 180,
        seed: int | None = 42,
        include_seasonal_analysis: bool = True,
        include_stress_analysis: bool = True,
    ) -> None:
        self.scenarios = list(scenarios)
        if not self.scenarios:
            raise ValueError("At least one PFT scenario is required")
        self.n_days = n_days
        self.seed = seed
        self.include_seasonal_analysis = include_seasonal_analysis
        self.include_stress_analysis = include_stress_analysis
        self._current_scenario: PFTScenario | None = None
        self._synthetic_data: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency.

        NSE = 1 - Σ(obs - sim)² / Σ(obs - mean(obs))²
        """
        numerator = np.sum((observed - simulated) ** 2)
        denominator = np.sum((observed - np.mean(observed)) ** 2)
        if denominator == 0:
            return np.nan
        return 1 - (numerator / denominator)

    @staticmethod
    def _calculate_kge(observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency.

        KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
        where r = correlation, α = std_sim/std_obs, β = mean_sim/mean_obs
        """
        r = np.corrcoef(observed, simulated)[0, 1]
        alpha = np.std(simulated) / (np.std(observed) + 1e-10)
        beta = np.mean(simulated) / (np.mean(observed) + 1e-10)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return kge

    def _score_method(
        self, df: pd.DataFrame, estimate: pd.DataFrame, method: str
    ) -> ComparisonResult:
        """Calculate comprehensive performance metrics for a single method.

        Parameters
        ----------
        df : pd.DataFrame
            Synthetic flux data with true T and E values.
        estimate : pd.DataFrame
            Estimated T and E values from the partitioning method.
        method : str
            Name of the method being evaluated.

        Returns
        -------
        ComparisonResult
            Comprehensive metrics for this method-scenario combination.
        """
        merged = estimate.merge(
            df[["datetime", "T_true", "E_true", "VPD", "SWC"]], on="datetime"
        )

        # Basic metrics
        bias_T = (merged["T_est"] - merged["T_true"]).mean()
        bias_E = (merged["E_est"] - merged["E_true"]).mean()

        mae_T = np.mean(np.abs(merged["T_est"] - merged["T_true"]))
        mae_E = np.mean(np.abs(merged["E_est"] - merged["E_true"]))

        rmse_T = np.sqrt(np.mean((merged["T_est"] - merged["T_true"]) ** 2))
        rmse_E = np.sqrt(np.mean((merged["E_est"] - merged["E_true"]) ** 2))

        mean_T_true = merged["T_true"].mean()
        mean_E_true = merged["E_true"].mean()
        relative_error_T = rmse_T / (mean_T_true + 1e-6)
        relative_error_E = rmse_E / (mean_E_true + 1e-6)

        corr_T = merged[["T_est", "T_true"]].corr().iloc[0, 1]
        corr_E = merged[["E_est", "E_true"]].corr().iloc[0, 1]

        # Ratio metrics
        t_et_true = merged["T_true"] / np.maximum(
            merged["T_true"] + merged["E_true"], 1e-6
        )
        t_et_est = merged["T_est"] / np.maximum(
            merged["T_est"] + merged["E_est"], 1e-6
        )
        t_et_error = np.sqrt(np.mean((t_et_est - t_et_true) ** 2))

        # Advanced efficiency metrics
        nse_T = self._calculate_nse(merged["T_true"].values, merged["T_est"].values)
        nse_E = self._calculate_nse(merged["E_true"].values, merged["E_est"].values)
        kge_T = self._calculate_kge(merged["T_true"].values, merged["T_est"].values)
        kge_E = self._calculate_kge(merged["E_true"].values, merged["E_est"].values)

        # Seasonal metrics (if enabled)
        seasonal_metrics = {}
        if self.include_seasonal_analysis:
            merged["season"] = (merged["datetime"].dt.month % 12 // 3 + 1).map(
                {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
            )
            for season in ["spring", "summer", "fall", "winter"]:
                season_data = merged[merged["season"] == season]
                if len(season_data) > 10:
                    season_rmse_T = np.sqrt(
                        np.mean((season_data["T_est"] - season_data["T_true"]) ** 2)
                    )
                    seasonal_metrics[f"rmse_T_{season}"] = season_rmse_T

        # Water stress analysis (if enabled)
        if self.include_stress_analysis:
            dry_mask = merged["SWC"] < 0.3
            wet_mask = merged["SWC"] > 0.7
            # Provide finite fallback values when there are too few samples
            if dry_mask.sum() > 10:
                dry_rmse_T = np.sqrt(
                    np.mean(
                        (merged.loc[dry_mask, "T_est"] - merged.loc[dry_mask, "T_true"])
                        ** 2
                    )
                )
            else:
                # fallback to overall rmse_T to ensure a finite metric is present
                dry_rmse_T = rmse_T
            seasonal_metrics["rmse_T_dry"] = dry_rmse_T

            if wet_mask.sum() > 10:
                wet_rmse_T = np.sqrt(
                    np.mean(
                        (merged.loc[wet_mask, "T_est"] - merged.loc[wet_mask, "T_true"])
                        ** 2
                    )
                )
            else:
                # fallback to overall rmse_T to ensure a finite metric is present
                wet_rmse_T = rmse_T
            seasonal_metrics["rmse_T_wet"] = wet_rmse_T

        scenario_name = self._current_scenario.name if self._current_scenario else "unknown"

        return ComparisonResult(
            scenario=scenario_name,
            method=method,
            bias_T=bias_T,
            bias_E=bias_E,
            mae_T=mae_T,
            mae_E=mae_E,
            rmse_T=rmse_T,
            rmse_E=rmse_E,
            relative_error_T=relative_error_T,
            relative_error_E=relative_error_E,
            correlation_T=corr_T,
            correlation_E=corr_E,
            t_et_ratio_error=t_et_error,
            nse_T=nse_T,
            nse_E=nse_E,
            kge_T=kge_T,
            kge_E=kge_E,
            seasonal_metrics=seasonal_metrics,
        )

    def run(self) -> List[ComparisonResult]:
        """Execute the synthetic experiment for all scenarios and methods.

        Returns
        -------
        List[ComparisonResult]
            Comprehensive performance metrics for each method-scenario combination.
        """
        results: List[ComparisonResult] = []
        for i, scenario in enumerate(self.scenarios):
            seed = None if self.seed is None else self.seed + i
            df = generate_synthetic_flux_data(scenario, n_days=self.n_days, seed=seed)
            self._synthetic_data[scenario.name] = df  # Store for later analysis
            method_outputs = run_method_emulators(df)
            self._current_scenario = scenario
            for method, estimate in method_outputs.items():
                results.append(self._score_method(df, estimate, method))
        return results

    def get_synthetic_data(self, scenario_name: str) -> Optional[pd.DataFrame]:
        """Retrieve stored synthetic data for a specific scenario.

        Parameters
        ----------
        scenario_name : str
            Name of the scenario to retrieve.

        Returns
        -------
        pd.DataFrame or None
            Synthetic flux data if available, None otherwise.
        """
        return self._synthetic_data.get(scenario_name)

    def results_to_dataframe(self, results: Iterable[ComparisonResult]) -> pd.DataFrame:
        """Convert comparison results to a tidy dataframe.

        Parameters
        ----------
        results : Iterable[ComparisonResult]
            Collection of comparison results to convert.

        Returns
        -------
        pd.DataFrame
            Tidy dataframe with one row per method-scenario combination.
        """
        records = []
        for r in results:
            base_record = {
                "scenario": r.scenario,
                "method": r.method,
                "bias_T": r.bias_T,
                "bias_E": r.bias_E,
                "mae_T": r.mae_T,
                "mae_E": r.mae_E,
                "rmse_T": r.rmse_T,
                "rmse_E": r.rmse_E,
                "relative_error_T": r.relative_error_T,
                "relative_error_E": r.relative_error_E,
                "correlation_T": r.correlation_T,
                "correlation_E": r.correlation_E,
                "t_et_ratio_error": r.t_et_ratio_error,
                "nse_T": r.nse_T,
                "nse_E": r.nse_E,
                "kge_T": r.kge_T,
                "kge_E": r.kge_E,
            }
            # Add seasonal metrics if present
            base_record.update(r.seasonal_metrics)
            records.append(base_record)
        return pd.DataFrame(records)

    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated statistics across scenarios for each method.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe as produced by results_to_dataframe().

        Returns
        -------
        pd.DataFrame
            Aggregated metrics with one row per method.
        """
        agg_dict = {
            "bias_T": ["mean", "std"],
            "bias_E": ["mean", "std"],
            "mae_T": ["mean", "std"],
            "mae_E": ["mean", "std"],
            "rmse_T": ["mean", "std"],
            "rmse_E": ["mean", "std"],
            "relative_error_T": ["mean", "std"],
            "relative_error_E": ["mean", "std"],
            "correlation_T": ["mean", "std"],
            "correlation_E": ["mean", "std"],
            "t_et_ratio_error": ["mean", "std"],
            "nse_T": ["mean", "std"],
            "nse_E": ["mean", "std"],
            "kge_T": ["mean", "std"],
            "kge_E": ["mean", "std"],
        }

        summary = df.groupby("method").agg(agg_dict)
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        return summary.reset_index()

    def performance_ranking(self, df: pd.DataFrame, metric: str = "rmse_T") -> pd.DataFrame:
        """Rank methods by a specific performance metric.

        Parameters
        ----------
        df : pd.DataFrame
            Results dataframe as produced by results_to_dataframe().
        metric : str, optional
            Metric to use for ranking (default: "rmse_T").

        Returns
        -------
        pd.DataFrame
            Methods ranked by the specified metric.
        """
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in results dataframe")

        ranking = df.groupby("method")[metric].mean().sort_values()
        return pd.DataFrame({"method": ranking.index, metric: ranking.values})


__all__ = ["PartitionComparison", "ComparisonResult"]
