"""Analysis utilities for ET partitioning methods.

This package provides comprehensive tools for comparing ET partitioning methods:
- simulation: Synthetic flux data generation with PFT scenarios
- comparison: Performance metrics and benchmarking
- visualization: Advanced plotting functions
"""

from .comparison import PartitionComparison, ComparisonResult
from .simulation import (
    PFTScenario,
    generate_synthetic_flux_data,
    run_method_emulators,
    get_pft_scenario,
    list_pft_scenarios,
    PREDEFINED_PFT_SCENARIOS,
    PFT_ENF,
    PFT_DBF,
    PFT_EBF,
    PFT_MF,
    PFT_CSH,
    PFT_OSH,
    PFT_WSA,
    PFT_GRA,
    PFT_CRO,
    PFT_WET,
)
from . import visualization

__all__ = [
    "PartitionComparison",
    "ComparisonResult",
    "PFTScenario",
    "generate_synthetic_flux_data",
    "run_method_emulators",
    "get_pft_scenario",
    "list_pft_scenarios",
    "PREDEFINED_PFT_SCENARIOS",
    "PFT_ENF",
    "PFT_DBF",
    "PFT_EBF",
    "PFT_MF",
    "PFT_CSH",
    "PFT_OSH",
    "PFT_WSA",
    "PFT_GRA",
    "PFT_CRO",
    "PFT_WET",
    "visualization",
]
