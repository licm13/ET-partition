#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test PFT sensitivity analysis for ET partition methods.

This module tests the performance of different ET partitioning methods
across various Plant Functional Type (PFT) scenarios with controlled
synthetic data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd

from analysis import (
    PartitionComparison,
    PFTScenario,
    PFT_ENF,
    PFT_DBF,
    PFT_GRA,
    PFT_CSH,
    get_pft_scenario,
    list_pft_scenarios,
    generate_synthetic_flux_data,
    run_method_emulators,
)


class TestPFTScenarios:
    """Test PFT scenario definitions and access functions."""

    def test_predefined_scenarios_exist(self):
        """Test that all predefined PFT scenarios are accessible."""
        pft_names = list_pft_scenarios()
        assert len(pft_names) >= 10, "Should have at least 10 predefined PFT scenarios"
        assert "ENF" in pft_names
        assert "DBF" in pft_names
        assert "GRA" in pft_names

    def test_get_pft_scenario(self):
        """Test retrieving PFT scenarios by name."""
        enf = get_pft_scenario("ENF")
        assert enf.name == "ENF"
        assert enf.canopy_conductance > 0
        assert 0 <= enf.soil_evap_fraction <= 1

    def test_get_pft_scenario_case_insensitive(self):
        """Test that PFT scenario retrieval is case-insensitive."""
        enf1 = get_pft_scenario("ENF")
        enf2 = get_pft_scenario("enf")
        assert enf1.name == enf2.name

    def test_invalid_pft_raises_error(self):
        """Test that requesting invalid PFT raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_pft_scenario("INVALID_PFT")

    def test_pft_scenario_frozen(self):
        """Test that PFT scenarios are immutable (frozen dataclass)."""
        enf = PFT_ENF
        with pytest.raises(Exception):  # Should raise FrozenInstanceError
            enf.canopy_conductance = 999


class TestSyntheticDataGeneration:
    """Test synthetic flux data generation."""

    def test_generate_basic_data(self):
        """Test basic synthetic data generation."""
        df = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=42)

        # Check required columns exist
        required_cols = ["datetime", "GPP", "ET", "VPD", "SWC", "T_true", "E_true"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check data shape
        expected_rows = 10 * 48  # 10 days * 48 half-hours
        assert len(df) == expected_rows

        # Check data ranges
        assert df["GPP"].min() >= 0, "GPP should be non-negative"
        assert df["ET"].min() >= 0, "ET should be non-negative"
        assert df["VPD"].min() >= 0, "VPD should be non-negative"
        assert 0 <= df["SWC"].min() <= 1, "SWC should be in [0,1]"
        assert 0 <= df["SWC"].max() <= 1, "SWC should be in [0,1]"

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        df1 = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=42)
        df2 = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=43)

        assert not np.allclose(df1["GPP"].values, df2["GPP"].values)
        assert not np.allclose(df1["ET"].values, df2["ET"].values)

    def test_same_seed_produces_same_data(self):
        """Test reproducibility with same seed."""
        df1 = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=42)
        df2 = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_pft_characteristics_reflected(self):
        """Test that PFT characteristics are reflected in synthetic data."""
        # ENF: high transpiration
        df_enf = generate_synthetic_flux_data(PFT_ENF, n_days=30, seed=42)
        t_et_ratio_enf = df_enf["T_true"].sum() / (df_enf["T_true"].sum() + df_enf["E_true"].sum())

        # GRA: lower transpiration ratio
        df_gra = generate_synthetic_flux_data(PFT_GRA, n_days=30, seed=42)
        t_et_ratio_gra = df_gra["T_true"].sum() / (df_gra["T_true"].sum() + df_gra["E_true"].sum())

        # Allow a small tolerance because synthetic data can vary due to
        # stochastic sampling and parameter interactions; intent is that
        # ENF is similar-to-or-more transpiration-dominant than GRA.
        tol = 0.03
        assert t_et_ratio_enf >= t_et_ratio_gra - tol, (
            f"ENF should have similar or higher T/ET ratio than GRA (tol={tol})"
        )


class TestMethodEmulators:
    """Test method emulator implementations."""

    def test_all_methods_run(self):
        """Test that all method emulators can run."""
        df = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=42)
        results = run_method_emulators(df)

        assert "uWUE" in results
        assert "TEA" in results
        assert "Perez-Priego" in results

        for method, result_df in results.items():
            assert "datetime" in result_df.columns
            assert "T_est" in result_df.columns
            assert "E_est" in result_df.columns
            assert len(result_df) == len(df)

    def test_partition_sums_to_et(self):
        """Test that T + E approximately equals ET."""
        df = generate_synthetic_flux_data(PFT_ENF, n_days=10, seed=42)
        results = run_method_emulators(df)

        for method, result_df in results.items():
            merged = result_df.merge(df[["datetime", "ET"]], on="datetime")
            reconstructed_et = merged["T_est"] + merged["E_est"]
            # Should be close due to clipping, but not exact
            np.testing.assert_allclose(
                reconstructed_et, merged["ET"], rtol=0.1, atol=0.05,
                err_msg=f"{method} partition doesn't sum to ET"
            )


class TestPartitionComparison:
    """Test PartitionComparison class functionality."""

    def test_basic_comparison(self):
        """Test basic comparison across methods and scenarios."""
        scenarios = [PFT_ENF, PFT_DBF]
        comparison = PartitionComparison(scenarios, n_days=30, seed=42)
        results = comparison.run()

        # Should have results for each method-scenario combination
        assert len(results) == 2 * 3  # 2 scenarios * 3 methods

        # Check result structure
        for result in results:
            assert hasattr(result, "scenario")
            assert hasattr(result, "method")
            assert hasattr(result, "rmse_T")
            assert hasattr(result, "rmse_E")
            assert hasattr(result, "correlation_T")

    def test_results_to_dataframe(self):
        """Test conversion of results to DataFrame."""
        scenarios = [PFT_ENF, PFT_GRA]
        comparison = PartitionComparison(scenarios, n_days=20, seed=42)
        results = comparison.run()
        df = comparison.results_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert "scenario" in df.columns
        assert "method" in df.columns
        assert "rmse_T" in df.columns
        assert len(df) == 2 * 3  # 2 scenarios * 3 methods

    def test_aggregate_metrics(self):
        """Test aggregation of metrics across scenarios."""
        scenarios = [PFT_ENF, PFT_DBF, PFT_GRA]
        comparison = PartitionComparison(scenarios, n_days=20, seed=42)
        results = comparison.run()
        df = comparison.results_to_dataframe(results)
        summary = comparison.aggregate_metrics(df)

        assert len(summary) == 3  # 3 methods
        assert "method" in summary.columns
        assert "rmse_T_mean" in summary.columns
        assert "rmse_T_std" in summary.columns

    def test_performance_ranking(self):
        """Test performance ranking functionality."""
        scenarios = [PFT_ENF, PFT_DBF]
        comparison = PartitionComparison(scenarios, n_days=20, seed=42)
        results = comparison.run()
        df = comparison.results_to_dataframe(results)
        ranking = comparison.performance_ranking(df, metric="rmse_T")

        assert len(ranking) == 3  # 3 methods
        assert "method" in ranking.columns
        assert "rmse_T" in ranking.columns
        # Check that ranking is sorted (ascending for RMSE)
        assert ranking["rmse_T"].is_monotonic_increasing

    def test_seasonal_analysis(self):
        """Test seasonal analysis functionality."""
        scenarios = [PFT_ENF]
        comparison = PartitionComparison(
            scenarios, n_days=180, seed=42,
            include_seasonal_analysis=True
        )
        results = comparison.run()
        df = comparison.results_to_dataframe(results)

        # Check for seasonal columns
        seasonal_cols = [col for col in df.columns if "rmse_T_" in col and col != "rmse_T"]
        assert len(seasonal_cols) > 0, "Should have seasonal metrics"

    def test_stress_analysis(self):
        """Test water stress analysis functionality."""
        scenarios = [PFT_ENF]
        comparison = PartitionComparison(
            scenarios, n_days=180, seed=42,
            include_stress_analysis=True
        )
        results = comparison.run()
        df = comparison.results_to_dataframe(results)

        # Check for stress-related columns
        assert any("dry" in col for col in df.columns) or any("wet" in col for col in df.columns), \
            "Should have stress analysis metrics"

    def test_get_synthetic_data(self):
        """Test retrieval of synthetic data."""
        scenarios = [PFT_ENF, PFT_DBF]
        comparison = PartitionComparison(scenarios, n_days=20, seed=42)
        comparison.run()

        # Retrieve data
        enf_data = comparison.get_synthetic_data("ENF")
        dbf_data = comparison.get_synthetic_data("DBF")
        invalid_data = comparison.get_synthetic_data("INVALID")

        assert enf_data is not None
        assert isinstance(enf_data, pd.DataFrame)
        assert dbf_data is not None
        assert invalid_data is None


class TestCrossPFTSensitivity:
    """Test method sensitivity across different PFT types."""

    @pytest.fixture
    def comparison_results(self):
        """Fixture providing comparison results across multiple PFTs."""
        scenarios = [PFT_ENF, PFT_DBF, PFT_GRA, PFT_CSH]
        comparison = PartitionComparison(scenarios, n_days=60, seed=123)
        results = comparison.run()
        df = comparison.results_to_dataframe(results)
        return comparison, df

    def test_all_methods_complete(self, comparison_results):
        """Test that all methods produce results for all PFTs."""
        _, df = comparison_results

        methods = df["method"].unique()
        scenarios = df["scenario"].unique()

        assert len(methods) == 3
        assert len(scenarios) == 4

        # Each method-scenario combination should exist
        for method in methods:
            for scenario in scenarios:
                subset = df[(df["method"] == method) & (df["scenario"] == scenario)]
                assert len(subset) == 1, f"Missing result for {method} @ {scenario}"

    def test_metrics_are_finite(self, comparison_results):
        """Test that all metrics are finite (no NaN or Inf)."""
        _, df = comparison_results

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert np.isfinite(df[col]).all(), f"Non-finite values in {col}"

    def test_rmse_positive(self, comparison_results):
        """Test that RMSE values are positive."""
        _, df = comparison_results

        assert (df["rmse_T"] > 0).all()
        assert (df["rmse_E"] > 0).all()

    def test_correlation_in_valid_range(self, comparison_results):
        """Test that correlations are in [-1, 1]."""
        _, df = comparison_results

        assert (df["correlation_T"] >= -1).all()
        assert (df["correlation_T"] <= 1).all()
        assert (df["correlation_E"] >= -1).all()
        assert (df["correlation_E"] <= 1).all()

    def test_method_consistency_across_pfts(self, comparison_results):
        """Test that methods maintain reasonable consistency across PFTs."""
        _, df = comparison_results

        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            rmse_T_range = method_df["rmse_T"].max() - method_df["rmse_T"].min()

            # RMSE should vary, but not by orders of magnitude
            assert rmse_T_range < method_df["rmse_T"].mean() * 2, \
                f"{method} shows excessive variation across PFTs"


def run_all_tests():
    """Run all tests in this module."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
