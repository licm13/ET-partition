# ET Partition Methods: Mathematical Foundations / ET蒸散发拆分方法数学原理

## Overview / 概述

This document provides a bilingual explanation of the mathematical concepts that underpin the three evapotranspiration (ET) partitioning schemes included in this repository: uWUE, TEA, and Perez-Priego. It also summarises how the new simulation utilities emulate these mechanisms to enable controlled experiments across plant functional types (PFTs).

本文档以中英文双语介绍本仓库中包含的三种ET蒸散发拆分方案（uWUE、TEA、Perez-Priego）的数学原理，并说明新增的模拟工具如何以不同植物功能型（PFT）场景对算法进行可控实验。

## uWUE Method / uWUE 方法

### Core relationships / 核心关系

The underlying concept of the underlying Water Use Efficiency (uWUE) approach is that the ratio between gross primary production (GPP) and transpiration (T) scales with the square root of vapour pressure deficit (VPD):

uWUE = GPP * sqrt(VPD) / T.

Rearranging yields an estimate for transpiration when GPP and VPD are known:

T = GPP * sqrt(VPD) / uWUE*.

where ``uWUE*`` is approximated by climatological or plant-type specific values. Evaporation is obtained as ``E = ET - T``.

uWUE 方法的核心思想是：光合速率（GPP）与蒸腾（T）之间的比例随饱和水汽压差（VPD）的平方根变化：

uWUE = GPP * sqrt(VPD) / T。

通过移项可得在已知 GPP 和 VPD 的情况下对蒸腾进行估算：

T = GPP * sqrt(VPD) / uWUE*。

其中 ``uWUE*`` 可由长期统计或植物功能型参数估算。蒸发通量则由 ``E = ET - T`` 获得。

### Simulation notes / 模拟说明

The surrogate emulator implemented in ``analysis/simulation.py`` scales ET by the normalised uWUE signal derived from ``GPP / sqrt(VPD)``. This preserves the theoretical proportionality while allowing controlled noise injection.

``analysis/simulation.py`` 中的替代算法通过 ``GPP / sqrt(VPD)`` 归一化信号对 ET 进行缩放，保持理论比例关系，并允许可控的噪声注入。

## TEA Method / TEA 方法

### Core relationships / 核心关系

The Transpiration and Evaporation Algorithm (TEA) links carbon and water fluxes through canopy conductance. A soil moisture stress factor ``β`` limits the stomatal conductance when water is scarce, resulting in

gc = gc_potential * β,

T ≈ (gc / (gc + ga)) * ET,

where ``gc`` is canopy conductance, ``ga`` aerodynamic conductance, and ``β`` is a function of soil water content. TEA typically infers ``β`` from observed soil moisture or latent energy partitioning.

TEA（Transpiration and Evaporation Algorithm）通过冠层导度连接碳水通量。当土壤水分不足时，水分胁迫因子 ``β`` 会降低气孔导度，因此：

gc = gc_potential * β,

T ≈ (gc / (gc + ga)) * ET，

其中 ``gc`` 为冠层导度，``ga`` 为气动导度，``β`` 由土壤含水量推断。TEA 通常利用土壤水分或能量分配信息来估计 ``β``。

### Simulation notes / 模拟说明

The emulator uses soil water content (``SWC``) as ``β`` and scales ET by ``GPP`` to account for biochemical control. This mimics the TEA mechanism where dry soils suppress transpiration relative to evaporation.

替代算法使用土壤含水量（``SWC``）作为 ``β``，并结合 ``GPP`` 对 ET 进行缩放，从而模拟干燥土壤抑制蒸腾、蒸发占比上升的行为。

## Perez-Priego Method / Perez-Priego 方法

### Core relationships / 核心关系

The Perez-Priego approach applies the novel transpiration fraction estimator derived from radiation partitioning and ecosystem conductances. The method exploits the observed covariance between shortwave radiation, aerodynamic coupling, and canopy water use to compute

T/ET ≈ f(Rn, Δ, γ, gc, ga),

where ``Δ`` is the slope of the saturation vapour pressure curve and ``γ`` the psychrometric constant. Radiation and conductance terms determine the fraction of energy that contributes to transpiration versus evaporation.

Perez-Priego 方法通过辐射与导度的分配关系估算蒸腾占比 ``T/ET``。该方法利用短波辐射、气动耦合和冠层水分利用之间的统计关系，近似计算：

T/ET ≈ f(Rn, Δ, γ, gc, ga)，

其中 ``Δ`` 为饱和水汽压曲线斜率，``γ`` 为湿空气常数。辐射与导度因子共同决定能量在蒸腾与蒸发之间的分配。

### Simulation notes / 模拟说明

In the surrogate model the radiation term ``f_rad`` serves as a proxy for the energy control of transpiration. Soil moisture modulates the final fraction to imitate canopy stress feedbacks.

在替代算法中，辐射因子 ``f_rad`` 表征能量对蒸腾的控制，并结合土壤含水量来模拟冠层胁迫反馈。

## Using the Advanced Comparison Utilities / 高级对比分析工具

The ``analysis`` package introduces a complete workflow for running synthetic experiments across multiple PFTs:

1. Define a list of :class:`analysis.simulation.PFTScenario` objects describing canopy conductance, VPD sensitivity, and structural biases.
2. Instantiate :class:`analysis.comparison.PartitionComparison` with the scenarios, number of simulation days, and random seed.
3. Call :meth:`PartitionComparison.run` to generate synthetic datasets, execute the method emulators, and compute error metrics.
4. Convert to tabular form or aggregate results with :meth:`PartitionComparison.results_to_dataframe` and :meth:`PartitionComparison.aggregate_metrics`.

``analysis`` 包提供了一套跨 PFT 场景执行合成实验的流程：

1. 定义若干 :class:`analysis.simulation.PFTScenario` 对象，描述冠层导度、VPD 敏感度以及结构偏差。
2. 使用这些情景、模拟天数和随机种子实例化 :class:`analysis.comparison.PartitionComparison`。
3. 调用 :meth:`PartitionComparison.run` 生成模拟数据、运行方法替代器，并计算误差指标。
4. 通过 :meth:`PartitionComparison.results_to_dataframe` 和 :meth:`PartitionComparison.aggregate_metrics` 获取详细表格或汇总结果。

The script ``examples/basic_usage.py`` now exposes an ``advanced_pft_comparison`` helper that writes detailed CSV diagnostics to ``outputs/advanced_analysis`` and prints aggregated RMSE statistics. Users can adapt the scenarios to match specific field experiments or benchmarking needs.

``examples/basic_usage.py`` 中新增 ``advanced_pft_comparison`` 函数，会将详细的指标导出至 ``outputs/advanced_analysis``，并在控制台输出综合 RMSE 统计。用户可根据具体场站或评估需求调整情景参数。

