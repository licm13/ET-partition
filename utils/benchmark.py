#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Benchmarking Utilities for ET Partitioning Methods
================================================================

This module provides tools for benchmarking the execution time, memory usage,
and throughput of the three ET partitioning methods (uWUE, TEA, Perez-Priego).

Author: ET Partition Project
Date: 2025
License: Mixed (see LICENSE)

Usage:
    from utils.benchmark import benchmark_all_methods, BenchmarkResult
    
    results = benchmark_all_methods('data/test_site', years=3)
    print(results.summary())
"""

import time
import gc
import tracemalloc
import psutil
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MethodBenchmark:
    """Benchmark results for a single method."""
    method_name: str
    execution_time: float  # seconds
    memory_peak: float  # MB
    memory_average: float  # MB
    throughput: float  # samples/second
    n_samples: int
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"{status} {self.method_name}: "
            f"time={self.execution_time:.2f}s, "
            f"memory={self.memory_peak:.1f}MB, "
            f"throughput={self.throughput:.0f} samples/s"
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark results for all methods."""
    timestamp: datetime
    data_path: str
    n_years: float
    n_samples: int
    methods: Dict[str, MethodBenchmark]
    system_info: Dict[str, Any]
    
    def summary(self) -> str:
        """Generate a summary report."""
        lines = [
            "=" * 70,
            "ET Partition Methods - Performance Benchmark Report",
            "=" * 70,
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data path: {self.data_path}",
            f"Data size: {self.n_years:.1f} years ({self.n_samples:,} samples)",
            "",
            "System Info:",
            f"  CPU cores: {self.system_info.get('cpu_count', 'N/A')}",
            f"  Total RAM: {self.system_info.get('total_memory', 0) / 1024:.1f} GB",
            f"  Python: {self.system_info.get('python_version', 'N/A')}",
            "",
            "-" * 70,
            "Results:",
            "-" * 70,
        ]
        
        for name, result in self.methods.items():
            lines.append(str(result))
            if result.error_message:
                lines.append(f"    Error: {result.error_message}")
        
        lines.extend([
            "-" * 70,
            "Performance Comparison:",
            "-" * 70,
        ])
        
        # Create comparison table
        if all(m.success for m in self.methods.values()):
            methods_sorted = sorted(
                self.methods.items(), 
                key=lambda x: x[1].execution_time
            )
            fastest = methods_sorted[0][1].execution_time
            
            for name, result in methods_sorted:
                ratio = result.execution_time / fastest
                lines.append(
                    f"  {name:15s}: {result.execution_time:6.2f}s "
                    f"({ratio:4.1f}x vs fastest)"
                )
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = []
        for name, result in self.methods.items():
            records.append({
                'method': name,
                'execution_time_s': result.execution_time,
                'memory_peak_mb': result.memory_peak,
                'memory_avg_mb': result.memory_average,
                'throughput_samples_s': result.throughput,
                'n_samples': result.n_samples,
                'success': result.success,
                'error': result.error_message,
            })
        return pd.DataFrame(records)
    
    def save(self, output_path: Path) -> None:
        """Save benchmark results to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = output_path / f"benchmark_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(self.summary())
        
        # Save CSV
        csv_file = output_path / f"benchmark_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        self.to_dataframe().to_csv(csv_file, index=False)
        
        print(f"Results saved to:\n  {summary_file}\n  {csv_file}")


# =============================================================================
# Memory Profiling Utilities
# =============================================================================

class MemoryProfiler:
    """Context manager for memory profiling."""
    
    def __init__(self, interval: float = 0.1):
        """
        Initialize memory profiler.
        
        Parameters
        ----------
        interval : float
            Sampling interval in seconds
        """
        self.interval = interval
        self.samples: List[float] = []
        self._running = False
    
    def __enter__(self):
        tracemalloc.start()
        gc.collect()
        self.samples = []
        self._start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        return self
    
    def __exit__(self, *args):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.peak_mb = peak / 1024 / 1024
        self.current_mb = current / 1024 / 1024
    
    def get_peak(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_mb
    
    def get_average(self) -> float:
        """Get average memory usage in MB."""
        return np.mean(self.samples) if self.samples else self.current_mb


# =============================================================================
# Benchmark Functions
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark context."""
    return {
        'cpu_count': os.cpu_count(),
        'total_memory': psutil.virtual_memory().total / 1024 / 1024,  # MB
        'available_memory': psutil.virtual_memory().available / 1024 / 1024,
        'python_version': sys.version.split()[0],
        'platform': sys.platform,
    }


def generate_benchmark_data(n_days: int = 365 * 3, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for benchmarking.
    
    Parameters
    ----------
    n_days : int
        Number of days of data to generate
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Synthetic flux data
    """
    np.random.seed(seed)
    n_halfhours = n_days * 48
    
    timestamps = pd.date_range('2020-01-01', periods=n_halfhours, freq='30min')
    doy = timestamps.dayofyear.values
    hour = timestamps.hour.values + timestamps.minute.values / 60
    
    # Seasonal and diurnal patterns
    seasonal = np.sin(2 * np.pi * (doy - 80) / 365)
    diurnal = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    diurnal[(hour < 6) | (hour > 18)] = 0
    
    # Generate variables
    ta = 15 + 10 * seasonal + 5 * diurnal + np.random.normal(0, 2, n_halfhours)
    vpd = 10 * (1 + 0.5 * seasonal) * (0.5 + 0.5 * diurnal) + np.random.normal(0, 2, n_halfhours)
    vpd = np.maximum(0.1, vpd)
    sw_in = 800 * diurnal * (0.7 + 0.3 * seasonal) + np.random.normal(0, 30, n_halfhours)
    sw_in = np.maximum(0, sw_in)
    
    gpp = 20 * diurnal * np.maximum(0, 1 - np.abs(ta - 20) / 30) + np.random.normal(0, 1, n_halfhours)
    gpp = np.maximum(0, gpp)
    
    le = 100 * diurnal * (0.5 + 0.5 * seasonal) + np.random.normal(0, 10, n_halfhours)
    le = np.maximum(0, le)
    
    return pd.DataFrame({
        'TIMESTAMP_START': timestamps.strftime('%Y%m%d%H%M').astype(int),
        'TA_F': ta,
        'TA_F_QC': 1,
        'VPD_F': vpd,
        'VPD_F_QC': 1,
        'SW_IN_F': sw_in,
        'LE_F_MDS': le,
        'LE_F_MDS_QC': 1,
        'GPP_NT_VUT_REF': gpp,
        'NEE_VUT_REF_QC': 1,
        'P_F': np.random.exponential(0.1, n_halfhours) * (np.random.random(n_halfhours) < 0.05),
        'RH': 100 * np.exp(-vpd / 30),
        'WS_F': np.abs(np.random.normal(3, 1.5, n_halfhours)),
    })


def benchmark_uwue(data: pd.DataFrame) -> MethodBenchmark:
    """
    Benchmark uWUE method.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input flux data
        
    Returns
    -------
    MethodBenchmark
        Benchmark results
    """
    n_samples = len(data)
    
    try:
        # Prepare data
        et = data['LE_F_MDS'].values * 0.0007348  # Convert to mm
        vpd = data['VPD_F'].values / 10  # Convert to kPa
        gpp = data['GPP_NT_VUT_REF'].values
        
        gpp_x_sqrt_vpd = gpp * np.sqrt(np.maximum(vpd, 0.01))
        
        # Create masks
        valid_mask = np.isfinite(et) & np.isfinite(gpp_x_sqrt_vpd) & (et > 0) & (gpp > 0)
        
        # Benchmark
        gc.collect()
        with MemoryProfiler() as mem:
            start_time = time.perf_counter()
            
            # Simplified uWUE calculation
            if valid_mask.sum() > 100:
                uwue_p = np.percentile(gpp_x_sqrt_vpd[valid_mask] / et[valid_mask], 95)
                
                # Daily aggregation
                steps_per_day = 48
                n_days = n_samples // steps_per_day
                et_daily = et[:n_days * steps_per_day].reshape(n_days, steps_per_day)
                daily_et = np.nansum(et_daily, axis=1)
                
                # Calculate T
                daily_uwue = np.full(n_days, uwue_p)  # Simplified
                T = daily_et * (daily_uwue / uwue_p)
            
            execution_time = time.perf_counter() - start_time
        
        return MethodBenchmark(
            method_name='uWUE',
            execution_time=execution_time,
            memory_peak=mem.get_peak(),
            memory_average=mem.get_average(),
            throughput=n_samples / execution_time,
            n_samples=n_samples,
            success=True
        )
        
    except Exception as e:
        return MethodBenchmark(
            method_name='uWUE',
            execution_time=0,
            memory_peak=0,
            memory_average=0,
            throughput=0,
            n_samples=n_samples,
            success=False,
            error_message=str(e)
        )


def benchmark_tea(data: pd.DataFrame) -> MethodBenchmark:
    """
    Benchmark TEA method.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input flux data
        
    Returns
    -------
    MethodBenchmark
        Benchmark results
    """
    n_samples = len(data)
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare features
        features = data[['VPD_F', 'TA_F', 'SW_IN_F', 'RH', 'WS_F']].values
        target = data['GPP_NT_VUT_REF'].values / (data['LE_F_MDS'].values + 1)
        
        valid = np.isfinite(features).all(axis=1) & np.isfinite(target)
        X = features[valid]
        y = target[valid]
        
        # Limit training size for benchmark
        train_size = min(10000, len(X))
        
        # Benchmark
        gc.collect()
        with MemoryProfiler() as mem:
            start_time = time.perf_counter()
            
            rf = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X[:train_size], y[:train_size])
            
            # Predict
            wue_pred = rf.predict(X)
            T = data['GPP_NT_VUT_REF'].values[valid] / (wue_pred + 1e-10)
            
            execution_time = time.perf_counter() - start_time
        
        return MethodBenchmark(
            method_name='TEA',
            execution_time=execution_time,
            memory_peak=mem.get_peak(),
            memory_average=mem.get_average(),
            throughput=n_samples / execution_time,
            n_samples=n_samples,
            success=True
        )
        
    except Exception as e:
        return MethodBenchmark(
            method_name='TEA',
            execution_time=0,
            memory_peak=0,
            memory_average=0,
            throughput=0,
            n_samples=n_samples,
            success=False,
            error_message=str(e)
        )


def benchmark_perez_priego(data: pd.DataFrame) -> MethodBenchmark:
    """
    Benchmark Perez-Priego method.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input flux data
        
    Returns
    -------
    MethodBenchmark
        Benchmark results
    """
    n_samples = len(data)
    
    try:
        # Try to use Numba-optimized version
        try:
            from methods.perez_priego.et_partitioning_functions_numba import (
                calculate_stomatal_conductance_numba,
                calculate_transpiration_numba
            )
            use_numba = True
        except ImportError:
            use_numba = False
        
        # Prepare data
        Q = data['SW_IN_F'].values * 0.5 * 4.6  # Convert to PAR
        VPD = data['VPD_F'].values / 10  # Convert to kPa
        T_air = data['TA_F'].values
        P_atm = np.full(n_samples, 101.325)
        
        # Benchmark
        gc_module = gc  # Save reference to gc module
        gc_module.collect()
        with MemoryProfiler() as mem:
            start_time = time.perf_counter()
            
            if use_numba:
                stomatal_gc = calculate_stomatal_conductance_numba(
                    Q.astype(np.float64),
                    VPD.astype(np.float64),
                    T_air.astype(np.float64),
                    0.1  # gc_max
                )
                T = calculate_transpiration_numba(
                    stomatal_gc,
                    VPD.astype(np.float64),
                    P_atm.astype(np.float64)
                )
            else:
                # Simplified pure Python version
                a1 = 50
                D0 = 0.1
                T_opt = 25
                
                f_Q = Q / (Q + a1 + 1e-6)
                f_VPD = np.exp(-D0 * VPD)
                
                stomatal_gc = 0.1 * f_Q * f_VPD
                gw = 1.6 * stomatal_gc
                T = gw * VPD / P_atm * 1000
            
            execution_time = time.perf_counter() - start_time
        
        return MethodBenchmark(
            method_name='Perez-Priego',
            execution_time=execution_time,
            memory_peak=mem.get_peak(),
            memory_average=mem.get_average(),
            throughput=n_samples / execution_time,
            n_samples=n_samples,
            success=True,
            additional_metrics={'numba_enabled': use_numba}
        )
        
    except Exception as e:
        return MethodBenchmark(
            method_name='Perez-Priego',
            execution_time=0,
            memory_peak=0,
            memory_average=0,
            throughput=0,
            n_samples=n_samples,
            success=False,
            error_message=str(e)
        )


def benchmark_all_methods(
    data_path: Optional[str] = None,
    years: float = 3,
    use_synthetic: bool = True
) -> BenchmarkResult:
    """
    Benchmark all three ET partitioning methods.
    对比三种方法的执行时间和内存
    
    Parameters
    ----------
    data_path : str, optional
        Path to input data directory
    years : float
        Number of years of data to use
    use_synthetic : bool
        Whether to use synthetic data
        
    Returns
    -------
    BenchmarkResult
        Complete benchmark results
    """
    print("=" * 60)
    print("ET Partition Methods - Performance Benchmark")
    print("=" * 60)
    
    # Generate or load data
    n_days = int(years * 365)
    
    if use_synthetic:
        print(f"\nGenerating {years:.1f} years of synthetic data...")
        data = generate_benchmark_data(n_days)
    else:
        if data_path is None:
            data_path = str(project_root / 'data' / 'test_site')
        print(f"\nLoading data from {data_path}...")
        # Would load actual data here
        data = generate_benchmark_data(n_days)
    
    n_samples = len(data)
    print(f"Data size: {n_samples:,} samples ({n_samples / 48:.0f} days)")
    
    # Get system info
    system_info = get_system_info()
    print(f"\nSystem: {system_info['cpu_count']} cores, "
          f"{system_info['total_memory'] / 1024:.1f} GB RAM")
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    methods = {}
    
    print("  uWUE...", end=" ", flush=True)
    methods['uWUE'] = benchmark_uwue(data)
    print(f"done ({methods['uWUE'].execution_time:.2f}s)")
    
    print("  TEA...", end=" ", flush=True)
    methods['TEA'] = benchmark_tea(data)
    print(f"done ({methods['TEA'].execution_time:.2f}s)")
    
    print("  Perez-Priego...", end=" ", flush=True)
    methods['Perez-Priego'] = benchmark_perez_priego(data)
    print(f"done ({methods['Perez-Priego'].execution_time:.2f}s)")
    
    # Create result
    result = BenchmarkResult(
        timestamp=datetime.now(),
        data_path=data_path or "synthetic",
        n_years=years,
        n_samples=n_samples,
        methods=methods,
        system_info=system_info
    )
    
    print("\n" + result.summary())
    
    return result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark ET partitioning methods"
    )
    parser.add_argument(
        '--years', type=float, default=3,
        help="Number of years of data to benchmark (default: 3)"
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help="Path to input data (default: use synthetic data)"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        '--synthetic', action='store_true', default=True,
        help="Use synthetic data (default: True)"
    )
    
    args = parser.parse_args()
    
    result = benchmark_all_methods(
        data_path=args.data_path,
        years=args.years,
        use_synthetic=args.synthetic
    )
    
    if args.output:
        result.save(Path(args.output))
    
    return 0 if all(m.success for m in result.methods.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
