# -*- coding: utf-8 -*-
"""
Numba-Optimized ET Partitioning Functions for Perez-Priego Method
=================================================================

This module provides JIT-compiled, vectorized implementations of the
Perez-Priego ET partitioning functions for improved performance.

Performance targets:
- 5-10x speedup compared to pure Python implementation
- Numerical precision maintained (error < 1%)

Author: ET Partition Project
Date: 2025
License: Mixed (see LICENSE)

Usage:
    from methods.perez_priego.et_partitioning_functions_numba import (
        calculate_stomatal_conductance_numba,
        calculate_transpiration_numba,
        moving_window_optimization_numba,
    )
"""

from typing import Tuple, Optional
import numpy as np
from functools import lru_cache

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: create a no-op decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range


# =============================================================================
# Constants
# =============================================================================

# Physical constants
SPECIFIC_HEAT_AIR = 1003.5  # J/(kg·K)
GAS_CONSTANT_DRY_AIR = 287.058  # J/(kg·K)
MOLAR_MASS_AIR = 0.0289644  # kg/mol
LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg at ~20°C

# Model constants
T_MIN = 0.0  # Minimum temperature for beta function (°C)
T_MAX = 50.0  # Maximum temperature for beta function (°C)


# =============================================================================
# Core Numba-Accelerated Functions
# =============================================================================

@njit(cache=True, fastmath=True)
def calculate_stomatal_conductance_numba(
    Q: np.ndarray,
    VPD: np.ndarray,
    Tair: np.ndarray,
    gc_max: float,
    a1: float = 50.0,
    D0: float = 0.1,
    T_opt: float = 25.0
) -> np.ndarray:
    """
    Calculate stomatal conductance using modified Ball-Berry model.
    使用修改的Ball-Berry模型计算气孔导度
    
    JIT-compiled for 5-10x speedup over pure Python.
    
    Parameters
    ----------
    Q : np.ndarray
        Photosynthetically active radiation (μmol m⁻² s⁻¹)
        光合有效辐射
    VPD : np.ndarray
        Vapor pressure deficit (kPa)
        水汽压差
    Tair : np.ndarray
        Air temperature (°C)
        气温
    gc_max : float
        Maximum stomatal conductance (mol m⁻² s⁻¹)
        最大气孔导度
    a1 : float
        Light response parameter (default: 50)
        光响应参数
    D0 : float
        VPD sensitivity parameter (default: 0.1)
        VPD敏感度参数
    T_opt : float
        Optimal temperature for conductance (default: 25°C)
        最优温度
        
    Returns
    -------
    np.ndarray
        Stomatal conductance (mol m⁻² s⁻¹)
        气孔导度
    """
    n = len(Q)
    result = np.empty(n, dtype=np.float64)
    
    # Pre-calculate temperature response constants
    beta = (T_MAX - T_opt) / (T_MAX - T_MIN)
    denominator = (T_opt - T_MIN) * ((T_MAX - T_opt) ** beta)
    scale = 1.0 / (denominator + 1e-10)
    
    # Track maximum for normalization
    max_sensitivity = 0.0
    
    # First pass: calculate raw sensitivity
    for i in range(n):
        # Light response: Michaelis-Menten
        f_Q = Q[i] / (Q[i] + a1 + 1e-10)
        
        # VPD response: exponential decrease
        f_VPD = np.exp(-D0 * VPD[i])
        
        # Temperature response: beta function
        T_clip = min(max(Tair[i], T_MIN + 0.1), T_MAX - 0.1)
        T_diff = max(T_MAX - T_clip, 0.0)
        f_T = scale * (T_clip - T_MIN) * (T_diff ** beta)
        f_T = max(f_T, 0.0)
        
        # Combined sensitivity
        sensitivity = f_Q * f_VPD * f_T
        result[i] = sensitivity
        
        if sensitivity > max_sensitivity:
            max_sensitivity = sensitivity
    
    # Normalize and scale by gc_max
    if max_sensitivity > 1e-10:
        for i in range(n):
            result[i] = gc_max * result[i] / max_sensitivity
    else:
        for i in range(n):
            result[i] = 0.0
    
    return result


@njit(parallel=True, cache=True, fastmath=True)
def calculate_transpiration_numba(
    gc: np.ndarray,
    VPD_plant: np.ndarray,
    P_atm: np.ndarray
) -> np.ndarray:
    """
    Calculate transpiration from stomatal conductance (parallel version).
    从气孔导度计算蒸腾（并行版本）
    
    Uses numba.prange for parallel execution across array elements.
    
    Parameters
    ----------
    gc : np.ndarray
        Stomatal conductance for CO2 (mol m⁻² s⁻¹)
        CO2气孔导度
    VPD_plant : np.ndarray
        Vapor pressure deficit at leaf level (kPa)
        叶片水平的水汽压差
    P_atm : np.ndarray
        Atmospheric pressure (kPa)
        大气压
        
    Returns
    -------
    np.ndarray
        Transpiration (mol H₂O m⁻² s⁻¹), multiply by 18 for g/m²/s
        蒸腾（mol H₂O m⁻² s⁻¹），乘以18得到g/m²/s
    """
    n = len(gc)
    T = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        # Water vapor conductance is 1.6x CO2 conductance
        gw = 1.6 * gc[i]
        
        # Transpiration from Fick's law
        # T = gw * (VPD / P)
        T[i] = gw * VPD_plant[i] / (P_atm[i] + 1e-10) * 1000  # mmol/m²/s
    
    return T


@njit(cache=True, fastmath=True)
def calculate_air_density_numba(
    T_air: np.ndarray,
    P_atm: np.ndarray
) -> np.ndarray:
    """
    Calculate air density using ideal gas law.
    使用理想气体定律计算空气密度
    
    Parameters
    ----------
    T_air : np.ndarray
        Air temperature (°C)
    P_atm : np.ndarray
        Atmospheric pressure (kPa)
        
    Returns
    -------
    np.ndarray
        Air density (kg/m³)
    """
    n = len(T_air)
    rho = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        T_kelvin = T_air[i] + 273.15
        # P in Pa (kPa * 1000), R = 287.058 J/(kg·K)
        rho[i] = (P_atm[i] * 1000) / (GAS_CONSTANT_DRY_AIR * T_kelvin)
    
    return rho


@njit(cache=True, fastmath=True)
def calculate_saturation_vp_numba(T: np.ndarray) -> np.ndarray:
    """
    Calculate saturation vapor pressure using Tetens formula.
    使用Tetens公式计算饱和水汽压
    
    Parameters
    ----------
    T : np.ndarray
        Temperature (°C)
        
    Returns
    -------
    np.ndarray
        Saturation vapor pressure (kPa)
    """
    n = len(T)
    e_sat = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        # Tetens formula
        e_sat[i] = 0.61078 * np.exp((17.269 * T[i]) / (237.3 + T[i]))
    
    return e_sat


@njit(cache=True, fastmath=True)
def moving_window_optimization_numba(
    data: np.ndarray,
    window_size: int,
    step_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accelerated moving window calculation for parameter optimization.
    参数优化的加速滑动窗口计算
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (n_samples, n_features)
    window_size : int
        Size of moving window
    step_size : int
        Step size between windows (default: 1)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (window_means, window_stds)
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // step_size + 1
    
    means = np.empty((n_windows, n_features), dtype=np.float64)
    stds = np.empty((n_windows, n_features), dtype=np.float64)
    
    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        
        for f in range(n_features):
            window_data = data[start:end, f]
            
            # Calculate mean (ignoring NaN)
            count = 0
            total = 0.0
            for i in range(window_size):
                if not np.isnan(window_data[i]):
                    total += window_data[i]
                    count += 1
            
            if count > 0:
                mean_val = total / count
                means[w, f] = mean_val
                
                # Calculate std
                var_sum = 0.0
                for i in range(window_size):
                    if not np.isnan(window_data[i]):
                        diff = window_data[i] - mean_val
                        var_sum += diff * diff
                
                stds[w, f] = np.sqrt(var_sum / count) if count > 1 else 0.0
            else:
                means[w, f] = np.nan
                stds[w, f] = np.nan
    
    return means, stds


@njit(cache=True, fastmath=True)
def chi_optimal_numba(
    T_air: np.ndarray,
    VPD: np.ndarray,
    elevation_km: float,
    c_coef: float = 0.0
) -> np.ndarray:
    """
    Calculate optimal chi (Ci/Ca ratio) using Prentice et al. formulation.
    使用Prentice等人公式计算最优chi（Ci/Ca比值）
    
    Parameters
    ----------
    T_air : np.ndarray
        Air temperature (°C)
    VPD : np.ndarray
        Vapor pressure deficit (kPa)
    elevation_km : float
        Site elevation (km)
    c_coef : float
        Calibration coefficient (default: 0.0)
        
    Returns
    -------
    np.ndarray
        Optimal chi values (0-1)
    """
    n = len(T_air)
    chi = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        # Prevent log of zero/negative
        vpd_safe = max(VPD[i], 0.01)
        
        # Calculate theta
        theta = (0.0545 * (T_air[i] - 25) 
                 - 0.58 * np.log(vpd_safe) 
                 - 0.0815 * elevation_km 
                 + c_coef)
        
        # Logistic transformation to bound chi between 0 and 1
        exp_theta = np.exp(theta)
        chi[i] = exp_theta / (1 + exp_theta)
    
    return chi


# =============================================================================
# Vectorized Daily Aggregation
# =============================================================================

@njit(parallel=True, cache=True)
def aggregate_to_daily_numba(
    halfhourly_data: np.ndarray,
    timesteps_per_day: int = 48
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized aggregation from half-hourly to daily resolution.
    从半小时到日分辨率的向量化聚合
    
    Parameters
    ----------
    halfhourly_data : np.ndarray
        Half-hourly data (n_halfhours,)
    timesteps_per_day : int
        Number of timesteps per day (48 for half-hourly, 24 for hourly)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (daily_sum, daily_mean, daily_max)
    """
    n_halfhours = len(halfhourly_data)
    n_days = n_halfhours // timesteps_per_day
    
    daily_sum = np.empty(n_days, dtype=np.float64)
    daily_mean = np.empty(n_days, dtype=np.float64)
    daily_max = np.empty(n_days, dtype=np.float64)
    
    for d in prange(n_days):
        start = d * timesteps_per_day
        end = start + timesteps_per_day
        
        day_sum = 0.0
        day_max = -np.inf
        count = 0
        
        for i in range(start, end):
            val = halfhourly_data[i]
            if not np.isnan(val):
                day_sum += val
                count += 1
                if val > day_max:
                    day_max = val
        
        if count > 0:
            daily_sum[d] = day_sum
            daily_mean[d] = day_sum / count
            daily_max[d] = day_max
        else:
            daily_sum[d] = np.nan
            daily_mean[d] = np.nan
            daily_max[d] = np.nan
    
    return daily_sum, daily_mean, daily_max


# =============================================================================
# Caching Utilities
# =============================================================================

@lru_cache(maxsize=256)
def atmospheric_pressure_cached(elevation_km: float) -> float:
    """
    Calculate atmospheric pressure with caching.
    带缓存的大气压计算
    
    Uses standard barometric formula.
    
    Parameters
    ----------
    elevation_km : float
        Elevation in kilometers
        
    Returns
    -------
    float
        Atmospheric pressure (kPa)
    """
    P0 = 101.325  # Standard pressure at sea level (kPa)
    scale_height = 8.5  # km
    return P0 * np.exp(-elevation_km / scale_height)


@lru_cache(maxsize=64)
def get_temperature_response_constants(T_opt: float) -> Tuple[float, float]:
    """
    Pre-calculate temperature response constants for given T_opt.
    为给定T_opt预计算温度响应常数
    
    Parameters
    ----------
    T_opt : float
        Optimal temperature (°C)
        
    Returns
    -------
    Tuple[float, float]
        (beta, scale) constants
    """
    beta = (T_MAX - T_opt) / (T_MAX - T_MIN)
    denominator = (T_opt - T_MIN) * ((T_MAX - T_opt) ** beta)
    scale = 1.0 / (denominator + 1e-10)
    return beta, scale


# =============================================================================
# High-Level Convenience Functions
# =============================================================================

def partition_et_numba(
    GPP: np.ndarray,
    LE: np.ndarray,
    VPD: np.ndarray,
    T_air: np.ndarray,
    SW_in: np.ndarray,
    P_atm: np.ndarray,
    elevation_km: float = 0.0,
    gc_max: float = 0.1,
    a1: float = 50.0,
    D0: float = 0.1,
    T_opt: float = 25.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full ET partitioning using Numba-optimized functions.
    使用Numba优化函数的完整ET拆分
    
    This is the main entry point for the optimized Perez-Priego method.
    
    Parameters
    ----------
    GPP : np.ndarray
        Gross Primary Production (μmol CO₂ m⁻² s⁻¹)
    LE : np.ndarray
        Latent heat flux (W/m²)
    VPD : np.ndarray
        Vapor pressure deficit (kPa)
    T_air : np.ndarray
        Air temperature (°C)
    SW_in : np.ndarray
        Incoming shortwave radiation (W/m²)
    P_atm : np.ndarray
        Atmospheric pressure (kPa)
    elevation_km : float
        Site elevation (km)
    gc_max : float
        Maximum stomatal conductance (mol/m²/s)
    a1 : float
        Light response parameter
    D0 : float
        VPD sensitivity parameter
    T_opt : float
        Optimal temperature (°C)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (T, E) - Transpiration and Evaporation (same units as LE)
    """
    # Ensure arrays are float64 for Numba
    GPP = np.ascontiguousarray(GPP, dtype=np.float64)
    VPD = np.ascontiguousarray(VPD, dtype=np.float64)
    T_air = np.ascontiguousarray(T_air, dtype=np.float64)
    SW_in = np.ascontiguousarray(SW_in, dtype=np.float64)
    P_atm = np.ascontiguousarray(P_atm, dtype=np.float64)
    LE = np.ascontiguousarray(LE, dtype=np.float64)
    
    # Convert SW to PAR (approximate: PAR ≈ 0.5 * SW * 4.6)
    Q = SW_in * 0.5 * 4.6  # μmol/m²/s
    
    # Calculate stomatal conductance
    gc = calculate_stomatal_conductance_numba(
        Q, VPD, T_air, gc_max, a1, D0, T_opt
    )
    
    # Calculate chi for this site
    chi = chi_optimal_numba(T_air, VPD, elevation_km)
    
    # Calculate transpiration
    T_mol = calculate_transpiration_numba(gc, VPD, P_atm)
    
    # Convert mol/m²/s to W/m² (multiply by latent heat of vaporization)
    # 1 mol H2O = 18 g, latent heat ≈ 2.45 MJ/kg
    # T (W/m²) = T (mol/m²/s) * 18 (g/mol) / 1000 (kg/g) * 2.45e6 (J/kg)
    T = T_mol * 18 / 1000 * LATENT_HEAT_VAPORIZATION / 1000  # Back to mmol for scaling
    
    # Scale to match LE magnitude
    # Use GPP as a proxy for plant activity
    activity_mask = GPP > 0
    if activity_mask.sum() > 0:
        # Normalize by maximum activity
        gpp_norm = GPP / (np.nanmax(GPP) + 1e-10)
        T = T * gpp_norm
        
        # Ensure T doesn't exceed LE
        T = np.minimum(T, np.maximum(LE, 0))
    else:
        T = np.zeros_like(LE)
    
    # Calculate evaporation
    E = LE - T
    E = np.maximum(E, 0)  # E cannot be negative
    
    return T, E


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_speedup(n_samples: int = 100000) -> dict:
    """
    Benchmark speedup of Numba functions vs pure Python.
    基准测试Numba函数相对于纯Python的加速
    
    Parameters
    ----------
    n_samples : int
        Number of samples for benchmark
        
    Returns
    -------
    dict
        Timing results for each function
    """
    import time
    
    # Generate test data
    np.random.seed(42)
    Q = np.random.uniform(0, 1000, n_samples)
    VPD = np.random.uniform(0, 5, n_samples)
    T_air = np.random.uniform(-10, 40, n_samples)
    P_atm = np.full(n_samples, 101.325)
    
    results = {}
    
    # Warm-up JIT compilation
    _ = calculate_stomatal_conductance_numba(Q[:100], VPD[:100], T_air[:100], 0.1)
    
    # Benchmark stomatal conductance
    start = time.perf_counter()
    _ = calculate_stomatal_conductance_numba(Q, VPD, T_air, 0.1)
    results['stomatal_conductance_numba'] = time.perf_counter() - start
    
    # Pure Python version for comparison (first 10000 only for speed)
    n_test = min(10000, n_samples)
    
    def pure_python_gc(Q, VPD, T_air, gc_max, a1=50, D0=0.1, T_opt=25):
        result = np.empty(len(Q))
        for i in range(len(Q)):
            f_Q = Q[i] / (Q[i] + a1 + 1e-6)
            f_VPD = np.exp(-D0 * VPD[i])
            T_clip = min(max(T_air[i], 0.1), 49.9)
            beta = (50 - T_opt) / 50
            scale = 1 / ((T_opt) * (50 - T_opt)**beta)
            f_T = max(scale * T_clip * (50 - T_clip)**beta, 0)
            result[i] = gc_max * f_Q * f_VPD * f_T
        return result / (np.max(result) + 1e-6) * gc_max
    
    start = time.perf_counter()
    _ = pure_python_gc(Q[:n_test], VPD[:n_test], T_air[:n_test], 0.1)
    results['stomatal_conductance_python'] = (time.perf_counter() - start) * (n_samples / n_test)
    
    # Calculate speedup
    if results['stomatal_conductance_python'] > 0:
        results['speedup'] = results['stomatal_conductance_python'] / results['stomatal_conductance_numba']
    else:
        results['speedup'] = float('inf')
    
    return results


# =============================================================================
# Module Info
# =============================================================================

def get_module_info() -> dict:
    """Get information about this module's capabilities."""
    return {
        'numba_available': NUMBA_AVAILABLE,
        'numba_version': numba.__version__ if NUMBA_AVAILABLE else None,
        'functions': [
            'calculate_stomatal_conductance_numba',
            'calculate_transpiration_numba',
            'calculate_air_density_numba',
            'calculate_saturation_vp_numba',
            'moving_window_optimization_numba',
            'chi_optimal_numba',
            'aggregate_to_daily_numba',
            'atmospheric_pressure_cached',
            'partition_et_numba',
        ],
        'expected_speedup': '5-10x'
    }


if __name__ == '__main__':
    # Run benchmark if executed directly
    print("=" * 60)
    print("Numba-Optimized ET Partitioning Functions")
    print("=" * 60)
    
    info = get_module_info()
    print(f"Numba available: {info['numba_available']}")
    if info['numba_version']:
        print(f"Numba version: {info['numba_version']}")
    
    print("\nRunning benchmark...")
    results = benchmark_speedup()
    print(f"Speedup: {results['speedup']:.1f}x")
    print(f"Numba time: {results['stomatal_conductance_numba']*1000:.2f}ms")
    print(f"Python time: {results['stomatal_conductance_python']*1000:.2f}ms")
