"""
ET-Partition Methods Package
==============================

This package contains implementations of three widely-used evapotranspiration (ET)
partitioning methods that separate total ET into transpiration (T) and evaporation (E).

蒸散发拆分方法包
===============

本包包含三种广泛使用的蒸散发（ET）拆分方法的实现，用于将总ET分离为蒸腾（T）和蒸发（E）。

Available Methods / 可用方法:
------------------------------

1. **uWUE (Underlying Water Use Efficiency)** - methods.uwue
   - Author: Zhou et al. (2016)
   - Time resolution: Daily / 日尺度
   - Approach: Water use efficiency with quantile regression
   - 方法：基于水分利用效率的分位数回归

2. **TEA (Transpiration Estimation Algorithm)** - methods.tea
   - Author: Nelson et al. (2018)
   - Time resolution: Half-hourly / 半小时
   - Approach: Quantile Random Forest machine learning
   - 方法：分位数随机森林机器学习

3. **Perez-Priego (Optimality-based)** - methods.perez_priego
   - Author: Perez-Priego et al. (2018)
   - Time resolution: Half-hourly / 半小时
   - Approach: Stomatal conductance optimization
   - 方法：气孔导度最优化

Usage Example / 使用示例:
--------------------------

Command-line batch processing / 命令行批处理:

    # uWUE method
    python -m methods.uwue.batch --base-path data/test_site --output-path outputs/uwue

    # TEA method
    python -m methods.tea.batch --base-path data/test_site --output-path outputs/tea

    # Perez-Priego method
    python -m methods.perez_priego.batch --base-path data/test_site --output-path outputs/perez_priego

Python API / Python接口:

    from methods.uwue.batch import uWUEBatchProcessor
    from methods.tea.TEA import simplePartition
    from methods.perez_priego import et_partitioning_functions

See individual method READMEs for detailed documentation.
详细文档请参见各方法的README文件。

References / 参考文献:
-----------------------

- Zhou, S., et al. (2016). Water Resources Research, 52(2), 1160-1175.
  https://doi.org/10.1002/2015WR017766

- Nelson, J. A., et al. (2018). Biogeosciences, 15(8), 2433-2447.
  https://doi.org/10.5194/bg-15-2433-2018

- Perez-Priego, O., et al. (2018). JGR: Biogeosciences, 123(10), 3353-3370.
  https://doi.org/10.1029/2018JG004637
"""

__version__ = "0.1.0"
__author__ = "ET Partition Contributors"
__all__ = ["uwue", "tea", "perez_priego"]
