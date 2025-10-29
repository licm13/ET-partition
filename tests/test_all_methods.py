#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ET Partition Methods - Comprehensive Test Suite
=================================================

This script tests all three ET partitioning methods (uWUE, TEA, Perez-Priego)
using the FI-Hyy test site data located in data/test_site/.

The test suite validates:
1. Data loading and preprocessing
2. Method execution without errors
3. Output file generation
4. Result quality checks (no all-NaN outputs, reasonable value ranges)
5. Performance metrics

Author: ET Partition Project
Date: 2025
License: Mixed (see individual method directories)

Usage:
    python tests/test_all_methods.py

    or

    python -m tests.test_all_methods
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_results.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class TestResult:
    """
    测试结果类，用于记录单个方法的测试结果

    Class to track test results for each method
    """
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.start_time = None
        self.end_time = None
        self.success = False
        self.error_message = None
        self.output_files = []
        self.warnings = []
        self.stats = {}

    def start(self):
        """记录测试开始时间 / Record test start time"""
        self.start_time = time.time()
        logger.info(f"=" * 80)
        logger.info(f"开始测试方法 / Starting test for method: {self.method_name}")
        logger.info(f"=" * 80)

    def finish(self, success: bool = True, error: str = None):
        """
        记录测试完成状态

        Record test completion status

        Args:
            success: 测试是否成功 / Whether test succeeded
            error: 错误信息（如果失败）/ Error message if failed
        """
        self.end_time = time.time()
        self.success = success
        self.error_message = error

        duration = self.end_time - self.start_time
        logger.info(f"测试完成 / Test completed: {self.method_name}")
        logger.info(f"用时 / Duration: {duration:.2f} seconds")
        logger.info(f"状态 / Status: {'成功 / SUCCESS' if success else '失败 / FAILED'}")
        if error:
            logger.error(f"错误信息 / Error: {error}")
        logger.info(f"-" * 80)

    def add_output(self, filepath: Path):
        """添加输出文件 / Add output file"""
        if filepath.exists():
            self.output_files.append(filepath)
            logger.info(f"找到输出文件 / Found output file: {filepath}")

    def add_warning(self, warning: str):
        """添加警告信息 / Add warning message"""
        self.warnings.append(warning)
        logger.warning(f"警告 / Warning: {warning}")

    def add_stat(self, key: str, value):
        """添加统计信息 / Add statistic"""
        self.stats[key] = value


def check_test_data_exists(base_path: Path) -> bool:
    """
    检查测试数据是否存在

    Check if test data exists

    Args:
        base_path: 基础数据路径 / Base data path

    Returns:
        bool: 数据是否存在 / Whether data exists
    """
    logger.info(f"检查测试数据 / Checking test data at: {base_path}")

    if not base_path.exists():
        logger.error(f"测试数据路径不存在 / Test data path does not exist: {base_path}")
        return False

    # Look for FLUXNET site folders
    site_folders = list(base_path.glob("FLX_*_FLUXNET2015_*"))

    if not site_folders:
        logger.error(f"未找到FLUXNET站点文件夹 / No FLUXNET site folders found in: {base_path}")
        return False

    logger.info(f"找到 {len(site_folders)} 个站点文件夹 / Found {len(site_folders)} site folder(s)")
    for folder in site_folders:
        logger.info(f"  - {folder.name}")
        # Check for CSV files
        csv_files = list(folder.glob("*.csv"))
        if csv_files:
            logger.info(f"    包含 {len(csv_files)} 个CSV文件 / Contains {len(csv_files)} CSV file(s)")
        else:
            logger.warning(f"    未找到CSV文件 / No CSV files found")

    return True


def validate_output(result: TestResult, output_dir: Path, expected_patterns: list):
    """
    验证输出文件

    Validate output files

    Args:
        result: 测试结果对象 / TestResult object
        output_dir: 输出目录 / Output directory
        expected_patterns: 期望的文件模式列表 / List of expected file patterns
    """
    logger.info(f"验证输出文件 / Validating outputs in: {output_dir}")

    if not output_dir.exists():
        result.add_warning(f"输出目录不存在 / Output directory does not exist: {output_dir}")
        return

    # Check for expected output files
    found_files = []
    for pattern in expected_patterns:
        matching_files = list(output_dir.rglob(pattern))
        found_files.extend(matching_files)

    if not found_files:
        result.add_warning(f"未找到预期的输出文件 / No expected output files found")
        return

    logger.info(f"找到 {len(found_files)} 个输出文件 / Found {len(found_files)} output file(s)")

    # Validate each output file
    for filepath in found_files:
        result.add_output(filepath)

        # Check file size
        file_size = filepath.stat().st_size
        result.add_stat(f"file_size_{filepath.name}", file_size)

        if file_size == 0:
            result.add_warning(f"输出文件为空 / Output file is empty: {filepath.name}")
            continue

        # Try to read and validate CSV files
        if filepath.suffix == '.csv':
            try:
                df = pd.read_csv(filepath)
                logger.info(f"  - {filepath.name}: {len(df)} 行, {len(df.columns)} 列")
                logger.info(f"    {len(df)} rows, {len(df.columns)} columns")

                result.add_stat(f"rows_{filepath.name}", len(df))
                result.add_stat(f"cols_{filepath.name}", len(df.columns))

                # Check for all-NaN columns
                nan_cols = df.columns[df.isna().all()].tolist()
                if nan_cols:
                    result.add_warning(
                        f"文件 {filepath.name} 包含全NaN列 / "
                        f"File {filepath.name} contains all-NaN columns: {nan_cols}"
                    )

                # Check data ranges (basic sanity checks)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in ['T', 'E', 'ET', 'TEA_T', 'TEA_E', 'transpiration', 'evaporation']:
                        col_min = df[col].min()
                        col_max = df[col].max()
                        col_mean = df[col].mean()

                        logger.info(f"    {col}: 范围 / range = [{col_min:.4f}, {col_max:.4f}], "
                                  f"均值 / mean = {col_mean:.4f}")

                        # Warn if values are suspiciously large or all negative
                        if col_max > 50:  # ET components should typically be < 50 mm/day
                            result.add_warning(
                                f"{col} 存在异常大的值 / {col} has suspiciously large values: "
                                f"max = {col_max:.2f}"
                            )
                        if col_max < 0:
                            result.add_warning(
                                f"{col} 所有值为负 / {col} has all negative values"
                            )

            except Exception as e:
                result.add_warning(f"无法读取CSV文件 / Cannot read CSV file {filepath.name}: {str(e)}")


def test_uwue_method(base_path: Path, output_path: Path) -> TestResult:
    """
    测试uWUE方法

    Test uWUE method

    Args:
        base_path: 输入数据路径 / Input data path
        output_path: 输出路径 / Output path

    Returns:
        TestResult: 测试结果 / Test result
    """
    result = TestResult("uWUE")
    result.start()

    try:
        # Import uWUE batch processor
        from methods.uwue.batch import uWUEBatchProcessor

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize processor
        logger.info("初始化uWUE批处理器 / Initializing uWUE batch processor")
        processor = uWUEBatchProcessor(
            base_path=str(base_path),
            output_path=str(output_path),
            create_plots=True  # Enable plots for comprehensive testing
        )

        # Run processing
        logger.info("运行uWUE处理 / Running uWUE processing")
        processor.run()

        # Validate outputs
        validate_output(
            result,
            output_path,
            expected_patterns=["*.csv", "*.nc", "plots/*.png", "*.log"]
        )

        result.finish(success=True)

    except Exception as e:
        result.finish(success=False, error=str(e))
        logger.exception("uWUE测试失败 / uWUE test failed")

    return result


def test_tea_method(base_path: Path, output_path: Path) -> TestResult:
    """
    测试TEA方法

    Test TEA method

    Args:
        base_path: 输入数据路径 / Input data path
        output_path: 输出路径 / Output path

    Returns:
        TestResult: 测试结果 / Test result
    """
    result = TestResult("TEA")
    result.start()

    try:
        # Import TEA batch processor
        from methods.tea.batch import main as tea_batch_main

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare arguments (simulate command line args)
        class Args:
            base_path = str(base_path)
            output_path = str(output_path)
            pattern = r"FLX_.*_FLUXNET.*"

        # Run processing
        logger.info("运行TEA处理 / Running TEA processing")
        tea_batch_main(Args())

        # Validate outputs
        validate_output(
            result,
            output_path,
            expected_patterns=["*_TEA_results.csv"]
        )

        result.finish(success=True)

    except Exception as e:
        result.finish(success=False, error=str(e))
        logger.exception("TEA测试失败 / TEA test failed")

    return result


def test_perez_priego_method(base_path: Path, output_path: Path) -> TestResult:
    """
    测试Perez-Priego方法

    Test Perez-Priego method

    Args:
        base_path: 输入数据路径 / Input data path
        output_path: 输出路径 / Output path

    Returns:
        TestResult: 测试结果 / Test result
    """
    result = TestResult("Perez-Priego")
    result.start()

    try:
        # Import Perez-Priego batch processor
        from methods.perez_priego.batch import main as pp_batch_main

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare arguments (simulate command line args)
        class Args:
            base_path = str(base_path)
            output_path = str(output_path)
            site_metadata = None  # No metadata file, will use default altitude
            default_altitude = 0.5  # 500m default altitude

        # Run processing
        logger.info("运行Perez-Priego处理 / Running Perez-Priego processing")
        pp_batch_main(Args())

        # Validate outputs
        validate_output(
            result,
            output_path,
            expected_patterns=["*.csv", "*.png"]
        )

        result.finish(success=True)

    except Exception as e:
        result.finish(success=False, error=str(e))
        logger.exception("Perez-Priego测试失败 / Perez-Priego test failed")

    return result


def print_summary(results: list[TestResult]):
    """
    打印测试总结

    Print test summary

    Args:
        results: 测试结果列表 / List of test results
    """
    logger.info("\n" + "=" * 80)
    logger.info("测试总结 / TEST SUMMARY")
    logger.info("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = total_tests - passed_tests

    logger.info(f"总测试数 / Total tests: {total_tests}")
    logger.info(f"通过 / Passed: {passed_tests}")
    logger.info(f"失败 / Failed: {failed_tests}")
    logger.info("")

    # Detail for each method
    for result in results:
        status = "✓ 成功 / SUCCESS" if result.success else "✗ 失败 / FAILED"
        duration = result.end_time - result.start_time if result.end_time else 0

        logger.info(f"{result.method_name}: {status} ({duration:.2f}s)")

        if result.output_files:
            logger.info(f"  输出文件数 / Output files: {len(result.output_files)}")

        if result.warnings:
            logger.info(f"  警告数 / Warnings: {len(result.warnings)}")
            for warning in result.warnings[:3]:  # Show first 3 warnings
                logger.info(f"    - {warning}")
            if len(result.warnings) > 3:
                logger.info(f"    ... 还有 {len(result.warnings) - 3} 个警告 / "
                          f"and {len(result.warnings) - 3} more warnings")

        if result.error_message:
            logger.error(f"  错误 / Error: {result.error_message}")

        logger.info("")

    logger.info("=" * 80)

    if failed_tests == 0:
        logger.info("所有测试通过！/ ALL TESTS PASSED!")
    else:
        logger.error(f"{failed_tests} 个测试失败 / {failed_tests} test(s) FAILED")

    logger.info("=" * 80)

    # Return exit code
    return 0 if failed_tests == 0 else 1


def main():
    """
    主测试函数

    Main test function
    """
    logger.info("=" * 80)
    logger.info("ET蒸散发拆分方法 - 综合测试套件")
    logger.info("ET Partition Methods - Comprehensive Test Suite")
    logger.info("=" * 80)
    logger.info(f"测试开始时间 / Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Setup paths
    project_root = Path(__file__).parent.parent
    base_path = project_root / "data" / "test_site"
    output_base = project_root / "outputs" / "test_run"

    logger.info(f"项目根目录 / Project root: {project_root}")
    logger.info(f"测试数据路径 / Test data path: {base_path}")
    logger.info(f"输出基础路径 / Output base path: {output_base}")
    logger.info("")

    # Check if test data exists
    if not check_test_data_exists(base_path):
        logger.error("测试数据检查失败，退出测试 / Test data check failed, exiting")
        return 1

    logger.info("")

    # Run tests for all methods
    results = []

    # Test 1: uWUE
    uwue_output = output_base / "uwue"
    results.append(test_uwue_method(base_path, uwue_output))

    # Test 2: TEA
    tea_output = output_base / "tea"
    results.append(test_tea_method(base_path, tea_output))

    # Test 3: Perez-Priego
    pp_output = output_base / "perez_priego"
    results.append(test_perez_priego_method(base_path, pp_output))

    # Print summary
    exit_code = print_summary(results)

    logger.info(f"测试结束时间 / Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"测试日志已保存至 / Test log saved to: test_results.log")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
