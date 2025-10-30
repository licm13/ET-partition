import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import PartitionComparison, PFTScenario, generate_synthetic_flux_data


def test_generate_synthetic_flux_data_shape():
    scenario = PFTScenario(
        name="Test",
        canopy_conductance=0.8,
        vpd_sensitivity=0.5,
        soil_evap_fraction=0.3,
        photosynthesis_efficiency=1.1,
        interception_ratio=0.2,
        noise_std=0.05,
        transpiration_bias=1.0,
    )
    df = generate_synthetic_flux_data(scenario, n_days=10, seed=0)
    assert len(df) == 10 * 48
    required_cols = {"datetime", "GPP", "ET", "T_true", "E_true"}
    assert required_cols.issubset(df.columns)
    assert pd.api.types.is_datetime64_ns_dtype(df["datetime"])  # type: ignore[arg-type]


def test_partition_comparison_runs_multiple_scenarios():
    scenarios = [
        PFTScenario(
            name="S1",
            canopy_conductance=0.9,
            vpd_sensitivity=0.4,
            soil_evap_fraction=0.3,
            photosynthesis_efficiency=1.0,
            interception_ratio=0.25,
            noise_std=0.04,
            transpiration_bias=1.0,
        ),
        PFTScenario(
            name="S2",
            canopy_conductance=0.7,
            vpd_sensitivity=0.6,
            soil_evap_fraction=0.4,
            photosynthesis_efficiency=0.9,
            interception_ratio=0.3,
            noise_std=0.06,
            transpiration_bias=0.9,
        ),
    ]
    comparison = PartitionComparison(scenarios, n_days=5, seed=1)
    results = comparison.run()
    assert len(results) == 3 * len(scenarios)
    df = comparison.results_to_dataframe(results)
    summary = comparison.aggregate_metrics(df)
    assert set(summary["method"]) == {"uWUE", "TEA", "Perez-Priego"}
