"""Synthetic data generation and method emulation for ET partition comparison.

This module provides comprehensive tools for generating realistic synthetic flux data
across different Plant Functional Types (PFTs) and evaluating ET partitioning methods
under controlled conditions.

Key Components
--------------
- PFTScenario: Configuration class defining PFT characteristics
- Predefined PFT scenarios for common vegetation types
- Synthetic flux data generation with realistic seasonal and diurnal patterns
- Method emulators for uWUE, TEA, and Perez-Priego approaches
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PFTScenario:
    """Configuration describing a synthetic Plant Functional Type (PFT) scenario.

    The parameters are chosen to emulate distinct canopy structures and water-use
    strategies. Values are loosely inspired by literature benchmarks and provide
    enough variability to stress the partitioning algorithms in simulation.

    Attributes
    ----------
    name:
        Unique identifier of the PFT scenario.
    canopy_conductance:
        Proxy for aerodynamic/canopy conductance (mol m-2 s-1).
    vpd_sensitivity:
        Sensitivity of stomatal conductance to VPD (kPa-1).
    soil_evap_fraction:
        Baseline fraction of ET that comes from soil evaporation under wet soil.
    photosynthesis_efficiency:
        Scaling factor controlling GPP response to radiation.
    interception_ratio:
        Fraction of precipitation that is intercepted and evaporated directly.
    noise_std:
        Standard deviation of observational noise applied to ET (mm day-1).
    transpiration_bias:
        Systematic multiplicative bias in transpiration (to mimic structural
        uncertainty of the canopy model). Values >1 increase transpiration.
    """

    name: str
    canopy_conductance: float
    vpd_sensitivity: float
    soil_evap_fraction: float
    photosynthesis_efficiency: float
    interception_ratio: float
    noise_std: float
    transpiration_bias: float


def _daily_radiation(day_of_year: np.ndarray) -> np.ndarray:
    """Return a smooth annual cycle of incoming radiation (MJ m-2 day-1)."""

    return 12 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)


def _daily_vpd(day_of_year: np.ndarray, base: float) -> np.ndarray:
    """Generate VPD series following a seasonal cycle with synoptic variability."""

    seasonal = base + 0.6 * np.sin(2 * np.pi * (day_of_year + 30) / 365)
    noise = np.random.normal(0, 0.2, size=day_of_year.shape)
    return np.clip(seasonal + noise, 0.1, None)


def _soil_moisture_series(n_days: int, wet_fraction: float) -> np.ndarray:
    """Generate soil moisture between 0 and 1 with random wetting events."""

    soil = np.full(n_days, 0.5)
    rain_days = np.random.choice(n_days, size=max(1, int(0.15 * n_days)), replace=False)
    soil[rain_days] = 0.9
    drydown_rate = 0.01 + (1 - wet_fraction) * 0.02
    for i in range(1, n_days):
        soil[i] = max(0.1, soil[i - 1] - drydown_rate)
        if i in rain_days:
            soil[i] = 0.9
    noise = np.random.normal(0, 0.03, size=n_days)
    return np.clip(soil + noise, 0.1, 1.0)


def generate_synthetic_flux_data(
    scenario: PFTScenario,
    n_days: int = 180,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic half-hourly dataset for a PFT scenario.

    Parameters
    ----------
    scenario:
        Description of the canopy/soil configuration to simulate.
    n_days:
        Number of days to simulate.
    seed:
        Seed for the random generator to ensure reproducibility.

    Returns
    -------
    pandas.DataFrame
        Half-hourly synthetic observations with columns commonly required by
        partitioning algorithms: ``datetime``, ``GPP``, ``ET`` and auxiliary
        meteorology.
    """

    if seed is not None:
        np.random.seed(seed)

    n_steps = n_days * 48  # half-hourly resolution
    day_of_year = np.tile(np.arange(1, n_days + 1), 48)
    doy_unique = np.arange(1, n_days + 1)

    radiation_daily = _daily_radiation(doy_unique)
    diurnal_cycle = np.clip(np.sin(np.linspace(0, np.pi, 48)), 0, None)
    radiation = np.repeat(radiation_daily, 48) * np.tile(diurnal_cycle, n_days)

    vpd_daily = _daily_vpd(doy_unique, base=1.0)
    vpd = np.repeat(vpd_daily, 48)

    soil_moisture_daily = _soil_moisture_series(n_days, scenario.soil_evap_fraction)
    soil_moisture = np.repeat(soil_moisture_daily, 48)

    photosynthesis = (
        scenario.photosynthesis_efficiency
        * radiation
        * np.exp(-scenario.vpd_sensitivity * vpd)
        * soil_moisture
    )
    photosynthesis += np.random.normal(0, 0.05 * photosynthesis.max(), size=n_steps)
    photosynthesis = np.clip(photosynthesis, 0, None)

    transpiration_potential = (
        scenario.canopy_conductance
        * (radiation / radiation.max())
        * np.exp(-scenario.vpd_sensitivity * vpd)
        * soil_moisture
    )

    transpiration = scenario.transpiration_bias * transpiration_potential
    evaporation = (
        scenario.soil_evap_fraction * (1 - soil_moisture) * 0.5
        + scenario.interception_ratio * (radiation / radiation.max())
    )

    # enforce energy balance and add observation noise
    et = transpiration + evaporation
    et_obs = et + np.random.normal(0, scenario.noise_std, size=n_steps)
    et_obs = np.clip(et_obs, 0, None)

    # Precipitation indicator to mimic wet/dry transitions
    precipitation = np.random.gamma(shape=0.6, scale=3, size=n_steps)
    precipitation[np.repeat(soil_moisture_daily < 0.3, 48)] *= 0.3

    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2000-01-01", periods=n_steps, freq="30min"),
            "GPP": photosynthesis,
            "ET": et_obs,
            "VPD": vpd,
            "SWC": soil_moisture,
            "radiation": radiation,
            "precipitation": precipitation,
            "T_true": transpiration,
            "E_true": evaporation,
        }
    )
    return df


def _uwue_emulator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Emulate the uWUE method using water-use efficiency relationships."""

    uwue = df["GPP"] / np.sqrt(np.maximum(df["VPD"], 0.1))
    transpiration = uwue / (uwue.max() + 1e-6) * df["ET"]
    evaporation = df["ET"] - transpiration
    return transpiration.clip(lower=0), evaporation.clip(lower=0)


def _tea_emulator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Emulate the TEA method based on carbon-water coupling."""

    beta = np.clip(df["SWC"], 0.2, 1.0)
    transpiration = beta * df["GPP"] / (df["GPP"].max() + 1e-6) * df["ET"]
    evaporation = df["ET"] - transpiration
    return transpiration.clip(lower=0), evaporation.clip(lower=0)


def _perez_priego_emulator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Emulate the Perez-Priego method leveraging radiation partitioning."""

    f_rad = df["radiation"] / (df["radiation"].max() + 1e-6)
    transpiration = (0.4 + 0.5 * f_rad) * df["ET"] * (1 - 0.3 * (1 - df["SWC"]))
    evaporation = df["ET"] - transpiration
    return transpiration.clip(lower=0), evaporation.clip(lower=0)


_METHOD_EMULATORS = {
    "uWUE": _uwue_emulator,
    "TEA": _tea_emulator,
    "Perez-Priego": _perez_priego_emulator,
}


def run_method_emulators(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run surrogate versions of each partitioning method on a dataset.

    Parameters
    ----------
    df:
        Synthetic dataset as produced by :func:`generate_synthetic_flux_data`.

    Returns
    -------
    dict
        Mapping from method name to a dataframe with estimated ``T`` and ``E``.
    """

    results: Dict[str, pd.DataFrame] = {}
    for method, emulator in _METHOD_EMULATORS.items():
        t, e = emulator(df)
        results[method] = pd.DataFrame(
            {
                "datetime": df["datetime"],
                "T_est": t,
                "E_est": e,
            }
        )
    return results


def list_available_methods() -> List[str]:
    """Return the names of available surrogate methods."""

    return list(_METHOD_EMULATORS.keys())


# ==============================================================================
# Predefined PFT Scenarios / 预定义植物功能型场景
# ==============================================================================

# Evergreen Needleleaf Forest (ENF) / 常绿针叶林
# Characteristics: High canopy conductance, moderate VPD sensitivity,
# low soil evaporation due to dense canopy cover
PFT_ENF = PFTScenario(
    name="ENF",
    canopy_conductance=0.9,
    vpd_sensitivity=0.6,
    soil_evap_fraction=0.25,
    photosynthesis_efficiency=1.2,
    interception_ratio=0.35,
    noise_std=0.05,
    transpiration_bias=1.05,
)

# Deciduous Broadleaf Forest (DBF) / 落叶阔叶林
# Characteristics: Moderate canopy conductance, seasonal variation,
# moderate soil evaporation
PFT_DBF = PFTScenario(
    name="DBF",
    canopy_conductance=0.75,
    vpd_sensitivity=0.45,
    soil_evap_fraction=0.35,
    photosynthesis_efficiency=1.0,
    interception_ratio=0.25,
    noise_std=0.07,
    transpiration_bias=0.95,
)

# Evergreen Broadleaf Forest (EBF) / 常绿阔叶林
# Characteristics: High transpiration, low VPD sensitivity (tropical),
# low soil evaporation
PFT_EBF = PFTScenario(
    name="EBF",
    canopy_conductance=0.95,
    vpd_sensitivity=0.35,
    soil_evap_fraction=0.20,
    photosynthesis_efficiency=1.3,
    interception_ratio=0.40,
    noise_std=0.06,
    transpiration_bias=1.10,
)

# Mixed Forest (MF) / 混交林
# Characteristics: Intermediate properties between ENF and DBF
PFT_MF = PFTScenario(
    name="MF",
    canopy_conductance=0.80,
    vpd_sensitivity=0.50,
    soil_evap_fraction=0.30,
    photosynthesis_efficiency=1.1,
    interception_ratio=0.30,
    noise_std=0.06,
    transpiration_bias=1.00,
)

# Closed Shrubland (CSH) / 密闭灌丛
# Characteristics: Lower canopy conductance, higher soil evaporation,
# lower photosynthesis efficiency
PFT_CSH = PFTScenario(
    name="CSH",
    canopy_conductance=0.55,
    vpd_sensitivity=0.30,
    soil_evap_fraction=0.45,
    photosynthesis_efficiency=0.8,
    interception_ratio=0.18,
    noise_std=0.09,
    transpiration_bias=0.85,
)

# Open Shrubland (OSH) / 开放灌丛
# Characteristics: Low canopy conductance, high soil evaporation,
# adapted to dry conditions
PFT_OSH = PFTScenario(
    name="OSH",
    canopy_conductance=0.45,
    vpd_sensitivity=0.25,
    soil_evap_fraction=0.55,
    photosynthesis_efficiency=0.7,
    interception_ratio=0.12,
    noise_std=0.10,
    transpiration_bias=0.80,
)

# Woody Savannas (WSA) / 木本稀树草原
# Characteristics: Mixed tree-grass system, high soil evaporation,
# variable photosynthesis
PFT_WSA = PFTScenario(
    name="WSA",
    canopy_conductance=0.65,
    vpd_sensitivity=0.40,
    soil_evap_fraction=0.50,
    photosynthesis_efficiency=0.9,
    interception_ratio=0.20,
    noise_std=0.08,
    transpiration_bias=0.90,
)

# Grassland (GRA) / 草地
# Characteristics: Low canopy height, high soil evaporation,
# rapid response to precipitation
PFT_GRA = PFTScenario(
    name="GRA",
    canopy_conductance=0.50,
    vpd_sensitivity=0.35,
    soil_evap_fraction=0.60,
    photosynthesis_efficiency=0.75,
    interception_ratio=0.10,
    noise_std=0.11,
    transpiration_bias=0.75,
)

# Cropland (CRO) / 农田
# Characteristics: Seasonal management, moderate characteristics,
# variable depending on crop type
PFT_CRO = PFTScenario(
    name="CRO",
    canopy_conductance=0.70,
    vpd_sensitivity=0.42,
    soil_evap_fraction=0.40,
    photosynthesis_efficiency=1.05,
    interception_ratio=0.22,
    noise_std=0.08,
    transpiration_bias=0.92,
)

# Wetland (WET) / 湿地
# Characteristics: High water availability, high evaporation,
# unique water cycling
PFT_WET = PFTScenario(
    name="WET",
    canopy_conductance=0.85,
    vpd_sensitivity=0.20,
    soil_evap_fraction=0.65,
    photosynthesis_efficiency=0.95,
    interception_ratio=0.15,
    noise_std=0.07,
    transpiration_bias=0.88,
)


# Collection of all predefined scenarios / 所有预定义场景集合
PREDEFINED_PFT_SCENARIOS = [
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
]


def get_pft_scenario(name: str) -> PFTScenario:
    """Retrieve a predefined PFT scenario by name.

    Parameters
    ----------
    name : str
        Name of the PFT scenario (e.g., 'ENF', 'DBF', 'GRA').

    Returns
    -------
    PFTScenario
        The requested PFT scenario configuration.

    Raises
    ------
    ValueError
        If the specified PFT name is not found.

    Examples
    --------
    >>> scenario = get_pft_scenario('ENF')
    >>> print(scenario.name)
    ENF
    """
    for scenario in PREDEFINED_PFT_SCENARIOS:
        if scenario.name.upper() == name.upper():
            return scenario
    available = [s.name for s in PREDEFINED_PFT_SCENARIOS]
    raise ValueError(
        f"PFT scenario '{name}' not found. Available: {', '.join(available)}"
    )


def list_pft_scenarios() -> List[str]:
    """List all available predefined PFT scenarios.

    Returns
    -------
    List[str]
        Names of all predefined PFT scenarios.
    """
    return [s.name for s in PREDEFINED_PFT_SCENARIOS]


__all__ = [
    "PFTScenario",
    "generate_synthetic_flux_data",
    "run_method_emulators",
    "list_available_methods",
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
]
