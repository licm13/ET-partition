"""High level comparison utilities for ET partitioning methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .simulation import PFTScenario, generate_synthetic_flux_data, run_method_emulators


@dataclass
class ComparisonResult:
    """Summary statistics for a single method within one scenario."""

    scenario: str
    method: str
    bias_T: float
    bias_E: float
    rmse_T: float
    rmse_E: float
    correlation_T: float
    correlation_E: float
    t_et_ratio_error: float


class PartitionComparison:
    """Run synthetic experiments to compare ET partitioning approaches."""

    def __init__(
        self,
        scenarios: Iterable[PFTScenario],
        n_days: int = 180,
        seed: int | None = 42,
    ) -> None:
        self.scenarios = list(scenarios)
        if not self.scenarios:
            raise ValueError("At least one PFT scenario is required")
        self.n_days = n_days
        self.seed = seed
        self._current_scenario: PFTScenario | None = None

    def _score_method(self, df: pd.DataFrame, estimate: pd.DataFrame, method: str) -> ComparisonResult:
        merged = estimate.merge(df[["datetime", "T_true", "E_true"]], on="datetime")

        bias_T = (merged["T_est"] - merged["T_true"]).mean()
        bias_E = (merged["E_est"] - merged["E_true"]).mean()
        rmse_T = np.sqrt(np.mean((merged["T_est"] - merged["T_true"]) ** 2))
        rmse_E = np.sqrt(np.mean((merged["E_est"] - merged["E_true"]) ** 2))
        corr_T = merged[["T_est", "T_true"]].corr().iloc[0, 1]
        corr_E = merged[["E_est", "E_true"]].corr().iloc[0, 1]

        t_et_true = merged["T_true"] / np.maximum(merged["T_true"] + merged["E_true"], 1e-6)
        t_et_est = merged["T_est"] / np.maximum(merged["T_est"] + merged["E_est"], 1e-6)
        t_et_error = np.sqrt(np.mean((t_et_est - t_et_true) ** 2))

        scenario_name = self._current_scenario.name if self._current_scenario else "unknown"

        return ComparisonResult(
            scenario=scenario_name,
            method=method,
            bias_T=bias_T,
            bias_E=bias_E,
            rmse_T=rmse_T,
            rmse_E=rmse_E,
            correlation_T=corr_T,
            correlation_E=corr_E,
            t_et_ratio_error=t_et_error,
        )

    def run(self) -> List[ComparisonResult]:
        """Execute the synthetic experiment for all scenarios and methods."""

        results: List[ComparisonResult] = []
        for i, scenario in enumerate(self.scenarios):
            seed = None if self.seed is None else self.seed + i
            df = generate_synthetic_flux_data(scenario, n_days=self.n_days, seed=seed)
            method_outputs = run_method_emulators(df)
            self._current_scenario = scenario
            for method, estimate in method_outputs.items():
                results.append(self._score_method(df, estimate, method))
        return results

    def results_to_dataframe(self, results: Iterable[ComparisonResult]) -> pd.DataFrame:
        """Convert comparison results to a tidy dataframe."""

        records = [
            {
                "scenario": r.scenario,
                "method": r.method,
                "bias_T": r.bias_T,
                "bias_E": r.bias_E,
                "rmse_T": r.rmse_T,
                "rmse_E": r.rmse_E,
                "correlation_T": r.correlation_T,
                "correlation_E": r.correlation_E,
                "t_et_ratio_error": r.t_et_ratio_error,
            }
            for r in results
        ]
        return pd.DataFrame(records)

    def aggregate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated statistics across scenarios for each method."""

        summary = df.groupby("method").agg(
            bias_T_mean=("bias_T", "mean"),
            bias_E_mean=("bias_E", "mean"),
            rmse_T_mean=("rmse_T", "mean"),
            rmse_E_mean=("rmse_E", "mean"),
            corr_T_mean=("correlation_T", "mean"),
            corr_E_mean=("correlation_E", "mean"),
            t_et_ratio_rmse=("t_et_ratio_error", "mean"),
        )
        return summary.reset_index()


__all__ = ["PartitionComparison", "ComparisonResult"]
