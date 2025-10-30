# ET Partition Methods: Mathematical Foundations / ET蒸散发拆分方法数学原理

## Overview / 概述

This document provides a comprehensive bilingual explanation of the mathematical concepts and physical principles that underpin the three evapotranspiration (ET) partitioning schemes included in this repository: uWUE, TEA, and Perez-Priego. It includes detailed derivations, assumptions, and practical implementation considerations.

本文档以中英文双语全面介绍本仓库中包含的三种ET蒸散发拆分方案（uWUE、TEA、Perez-Priego）的数学概念和物理原理，包括详细推导、假设条件以及实际实施考虑事项。

---

## Table of Contents / 目录

1. [Background and Motivation](#background) / 背景与动机
2. [uWUE Method](#uwue-method) / uWUE方法
3. [TEA Method](#tea-method) / TEA方法
4. [Perez-Priego Method](#perez-priego-method) / Perez-Priego方法
5. [Comparison and Selection Guide](#comparison-guide) / 比较与选择指南
6. [Simulation Framework](#simulation-framework) / 模拟框架
7. [References](#references) / 参考文献

---

<a name="background"></a>
## 1. Background and Motivation / 背景与动机

### The ET Partitioning Problem / ET拆分问题

Total evapotranspiration (ET) from terrestrial ecosystems represents the combined water flux from:

陆地生态系统的总蒸散发（ET）代表来自以下的综合水通量：

- **Transpiration (T)**: Water vapor flux through plant stomata during photosynthesis
  **蒸腾（T）**：光合作用过程中通过植物气孔的水汽通量

- **Evaporation (E)**: Direct evaporation from soil surfaces, open water, and intercepted precipitation
  **蒸发（E）**：土壤表面、开放水面和截留降水的直接蒸发

Mathematically:

数学表达式：

```
ET = T + E
```

### Why Partition ET? / 为什么要拆分ET？

Partitioning ET is crucial for:

拆分ET对以下方面至关重要：

1. **Understanding ecosystem water use efficiency** / 理解生态系统水分利用效率
2. **Carbon-water coupling analysis** / 碳水耦合分析
3. **Drought response assessment** / 干旱响应评估
4. **Irrigation management** / 灌溉管理
5. **Climate model validation** / 气候模式验证

### Measurement Challenges / 测量挑战

Direct measurement of T and E separately is extremely difficult because:

分别直接测量T和E极其困难，因为：

- Eddy covariance towers measure total ET only / 涡度相关塔仅测量总ET
- Lysimeters are site-specific and expensive / 蒸渗仪具有场地特异性且昂贵
- Isotope methods require specialized equipment / 同位素方法需要专门设备

Therefore, **partitioning algorithms** are essential tools for estimating T and E from standard flux measurements.

因此，**拆分算法**是从标准通量测量中估算T和E的重要工具。

---

<a name="uwue-method"></a>
## 2. uWUE Method / uWUE方法

### 2.1 Theoretical Foundation / 理论基础

The **underlying Water Use Efficiency (uWUE)** approach is based on the principle that plants optimize their water use to maximize carbon gain while minimizing water loss.

**潜在水分利用效率（uWUE）**方法基于植物优化水分利用以最大化碳增益同时最小化水分损失的原理。

### 2.2 Mathematical Derivation / 数学推导

Starting from the definition of water use efficiency:

从水分利用效率的定义开始：

```
WUE = GPP / T
```

Zhou et al. (2016) introduced the concept of **underlying WUE** that accounts for atmospheric demand:

Zhou等人（2016）引入了考虑大气需求的**潜在WUE**概念：

```
uWUE = GPP × √VPD / T
```

where:
- GPP: Gross Primary Production (μmol CO₂ m⁻² s⁻¹) / 总初级生产力
- VPD: Vapor Pressure Deficit (kPa) / 水汽压差
- T: Transpiration (mm or W m⁻²) / 蒸腾

**Physical Interpretation / 物理解释:**

The √VPD term reflects the atmospheric driving force for transpiration. Under optimal conditions (well-watered soil, no stress), uWUE reaches a maximum value (uWUE*) that is characteristic of the plant functional type.

√VPD项反映了蒸腾的大气驱动力。在最优条件下（土壤水分充足、无胁迫），uWUE达到植物功能型特有的最大值（uWUE*）。

### 2.3 Transpiration Estimation / 蒸腾估算

Rearranging the uWUE equation:

重新排列uWUE方程：

```
T = GPP × √VPD / uWUE*
```

The key challenge is estimating **uWUE\***, the potential uWUE under optimal conditions.

关键挑战是估算**uWUE\***，即最优条件下的潜在uWUE。

**Implementation Steps / 实施步骤:**

1. **Filter optimal conditions** / 筛选最优条件
   - High soil moisture (SWC > threshold)
   - Recent precipitation events
   - Growing season data
   - High GPP values

2. **Estimate uWUE\* using quantile regression** / 使用分位数回归估算uWUE*
   ```
   uWUE* = quantile_95%(GPP × √VPD / ET)
   ```
   Using the 95th percentile captures near-optimal conditions.

   使用95分位数捕获接近最优的条件。

3. **Calculate daily transpiration** / 计算日蒸腾量
   ```
   T_daily = Σ(GPP_i × √VPD_i) / uWUE*_annual
   ```

4. **Derive evaporation** / 推导蒸发量
   ```
   E_daily = ET_daily - T_daily
   ```

### 2.4 Key Assumptions / 关键假设

1. **Constant uWUE\* for the ecosystem** / 生态系统uWUE*恒定
   - Valid for stable plant communities
   - May vary with phenology

2. **Optimal conditions occur frequently** / 最优条件经常出现
   - Requires sufficient data under wet conditions
   - At least 15-20% of data should be near-optimal

3. **GPP and T are tightly coupled** / GPP和T紧密耦合
   - Assumes constant leaf internal CO₂ concentration
   - Breaks down under severe stress

### 2.5 Advantages and Limitations / 优势与局限

**Advantages / 优势:**
- Simple conceptual framework / 概念框架简单
- Requires only standard flux tower variables / 仅需标准通量塔变量
- Daily time resolution suitable for many applications / 日时间分辨率适合许多应用

**Limitations / 局限:**
- Daily resolution only / 仅日分辨率
- Sensitive to uWUE* estimation / 对uWUE*估算敏感
- May overestimate T during stress periods / 可能在胁迫期高估T
- Requires sufficient data range / 需要足够的数据范围

### 2.6 Emulator Implementation / 模拟器实现

The surrogate emulator in `analysis/simulation.py` implements a simplified version:

`analysis/simulation.py`中的替代模拟器实现了简化版本：

```python
uwue = GPP / sqrt(max(VPD, 0.1))
T = (uwue / uwue_max) * ET
E = ET - T
```

This preserves the theoretical proportionality while enabling controlled experiments across PFT scenarios.

这保留了理论比例关系，同时支持跨PFT场景的可控实验。

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

---

<a name="tea-method"></a>
## 3. TEA Method / TEA方法

### 3.1 Theoretical Foundation / 理论基础

The **Transpiration Estimation Algorithm (TEA)** uses machine learning (Quantile Random Forest) to model water use efficiency under optimal conditions and extrapolate to all conditions.

**蒸腾估算算法（TEA）**使用机器学习（分位数随机森林）来建模最优条件下的水分利用效率，并推广到所有条件。

### 3.2 Mathematical Framework / 数学框架

TEA builds on the carbon-water coupling relationship:

TEA基于碳水耦合关系：

```
T = GPP / WUE
```

The challenge is estimating **WUE under current conditions**, which varies with:
- Soil moisture availability
- Atmospheric demand (VPD)
- Plant phenology
- Light conditions

估算**当前条件下的WUE**是关键挑战，它随以下因素变化：
- 土壤水分可用性
- 大气需求（VPD）
- 植物物候
- 光照条件

### 3.3 Algorithm Steps / 算法步骤

**Step 1: Calculate Auxiliary Indices / 计算辅助指数**

1. **Conservative Surface Water Index (CSWI)** / 保守地表水指数
   ```
   CSWI = SWC / SWC_max
   ```
   Indicates soil water stress level.
   
   表示土壤水分胁迫水平。

2. **Diurnal Water-Carbon Coupling Index (DWCI)** / 日变化水碳耦合指数
   ```
   DWCI = correlation(LE_diurnal, GPP_diurnal)
   ```
   Measures synchrony between water and carbon fluxes.
   
   测量水和碳通量之间的同步性。

3. **Diurnal Centroid** / 日质心
   ```
   DC = Σ(hour × GPP_hour) / Σ(GPP_hour)
   ```
   Captures timing of photosynthetic activity.
   
   捕获光合活动的时间特征。

**Step 2: Filter Optimal Conditions / 筛选最优条件**

Select data where:
- CSWI > 0.7 (high soil moisture)
- DWCI > 0.8 (strong coupling)
- Growing season
- NEE < 0 (carbon uptake)

选择满足以下条件的数据：
- CSWI > 0.7（高土壤水分）
- DWCI > 0.8（强耦合）
- 生长季
- NEE < 0（碳吸收）

**Step 3: Train Quantile Random Forest / 训练分位数随机森林**

```python
features = [VPD, Rg, Ta, CSWI, DWCI, DC, hour_of_day]
target = WUE_optimal = GPP / T

model = QuantileRandomForest(quantile=0.75)
model.fit(features_optimal, target_optimal)
```

The 75th quantile captures near-optimal WUE while being robust to outliers.

75分位数捕获接近最优的WUE，同时对异常值稳健。

**Step 4: Predict WUE for All Conditions / 预测所有条件的WUE**

```python
WUE_predicted = model.predict(features_all)
```

**Step 5: Calculate Transpiration and Evaporation / 计算蒸腾和蒸发**

```
T = GPP / WUE_predicted
E = ET - T
```

### 3.4 Key Equations / 关键方程

The core relationship in TEA can be expressed as:

TEA的核心关系可表示为：

```
T = GPP / WUE*(VPD, Rg, SWC, phenology)
```

where WUE* is the learned function from the random forest model.

其中WUE*是从随机森林模型学习的函数。

### 3.5 Advantages and Limitations / 优势与局限

**Advantages / 优势:**
- Half-hourly resolution / 半小时分辨率
- Non-parametric (fewer assumptions) / 非参数化（更少假设）
- Captures complex environmental interactions / 捕获复杂的环境交互
- Handles non-linear relationships / 处理非线性关系

**Limitations / 局限:**
- Requires substantial training data / 需要大量训练数据
- "Black box" nature / "黑箱"特性
- Computationally intensive / 计算密集
- Requires diverse environmental conditions in training set / 需要训练集中的多样环境条件

### 3.6 Emulator Implementation / 模拟器实现

```python
beta = clip(SWC, 0.2, 1.0)  # Soil moisture stress factor
T = beta * (GPP / GPP_max) * ET
E = ET - T
```

The emulator simplifies TEA by using soil moisture as the primary stress indicator.

模拟器通过使用土壤水分作为主要胁迫指标来简化TEA。

---

<a name="perez-priego-method"></a>
## 4. Perez-Priego Method / Perez-Priego方法

### 4.1 Theoretical Foundation / 理论基础

The Perez-Priego method uses **optimal stomatal conductance theory** combined with energy balance principles to partition ET. It's based on the hypothesis that plants optimize stomatal conductance to maximize carbon gain per unit water loss.

Perez-Priego方法使用**最优气孔导度理论**结合能量平衡原理来拆分ET。它基于植物优化气孔导度以最大化单位水分损失的碳增益的假设。

### 4.2 Mathematical Framework / 数学框架

**Penman-Monteith Equation / Penman-Monteith方程**

The foundation is the Penman-Monteith equation for transpiration:

基础是蒸腾的Penman-Monteith方程：

```
λT = (Δ × A + ρ × cp × VPD × ga) / (Δ + γ × (1 + ga/gc))
```

where:
- λ: Latent heat of vaporization / 汽化潜热
- Δ: Slope of saturation vapor pressure curve / 饱和水汽压曲线斜率
- A: Available energy / 可用能量
- ρ: Air density / 空气密度
- cp: Specific heat capacity / 比热容
- ga: Aerodynamic conductance / 气动导度
- gc: Canopy (stomatal) conductance / 冠层（气孔）导度
- γ: Psychrometric constant / 湿度计常数

### 4.3 Optimal Stomatal Conductance Theory / 最优气孔导度理论

The method assumes plants maximize:

该方法假设植物最大化：

```
Objective = GPP - λ × T
```

subject to biochemical and physical constraints.

受生化和物理约束。

This leads to an optimal relationship:

这导致最优关系：

```
gc_opt = f(Rn, VPD, Ca, Γ*, χ)
```

where:
- Ca: Atmospheric CO₂ concentration / 大气CO₂浓度
- Γ*: CO₂ compensation point / CO₂补偿点
- χ: Ratio of internal to external CO₂ / 内部与外部CO₂比值

### 4.4 Implementation Steps / 实施步骤

**Step 1: Calculate Long-term Parameters / 计算长期参数**

Estimate ecosystem-specific parameters from the data:

从数据估算生态系统特定参数：

```
χ_o = median(Ci / Ca)  # Optimal chi
WUE_o = quantile_95%(GPP / ET)  # Potential WUE
```

**Step 2: Moving Window Optimization / 滑动窗口优化**

For each 5-day window:

对每个5天窗口：

1. Filter daytime data (Rg > threshold)
2. Fit optimal parameters by minimizing:
   ```
   Σ(GPP_obs - GPP_model)² + Σ(ET_obs - ET_model)²
   ```

3. Optimal parameters:
   - gc_ref: Reference canopy conductance / 参考冠层导度
   - sensitivity parameters / 敏感度参数

**Step 3: Calculate Stomatal Conductance / 计算气孔导度**

```
gc = gc_ref × f₁(Rn) × f₂(VPD) × f₃(SWC)
```

where f₁, f₂, f₃ are response functions.

其中f₁、f₂、f₃是响应函数。

**Step 4: Calculate Transpiration / 计算蒸腾**

Use the Penman-Monteith equation with estimated gc:

使用估算的gc代入Penman-Monteith方程：

```
T = PM_equation(Rn, VPD, Ta, gc, ga)
```

**Step 5: Calculate Evaporation / 计算蒸发**

```
E = ET - T
E = max(0, E)  # Truncate negative values
```

### 4.5 Key Equations / 关键方程

**Energy Balance Constraint / 能量平衡约束:**

```
Rn = λ×ET + H + G
```

**Stomatal Response to VPD / 气孔对VPD的响应:**

```
gc = gc_max × exp(-k × VPD)
```

**Soil Moisture Limitation / 土壤水分限制:**

```
β = (SWC - SWC_min) / (SWC_opt - SWC_min)
gc = gc_potential × β
```

### 4.6 Advantages and Limitations / 优势与局限

**Advantages / 优势:**
- Mechanistic foundation / 机理基础
- Half-hourly resolution / 半小时分辨率
- Explicit treatment of stomatal control / 明确处理气孔控制
- Consistent with plant optimization theory / 符合植物优化理论

**Limitations / 局限:**
- Requires site elevation data / 需要站点高程数据
- Computationally intensive optimization / 计算密集的优化
- Many parameters to estimate / 许多参数需要估算
- Sensitive to parameter initialization / 对参数初始化敏感

### 4.7 Emulator Implementation / 模拟器实现

```python
f_rad = radiation / radiation_max  # Radiation factor
f_swc = 1 - 0.3 * (1 - SWC)  # Soil moisture factor
T = (0.4 + 0.5 * f_rad) * ET * f_swc
E = ET - T
```

The emulator captures the radiation control and soil moisture limitation.

模拟器捕获了辐射控制和土壤水分限制。

---

<a name="comparison-guide"></a>
## 5. Comparison and Selection Guide / 比较与选择指南

### 5.1 Method Comparison Table / 方法比较表

| Feature / 特征 | uWUE | TEA | Perez-Priego |
|---------------|------|-----|--------------|
| **Time Resolution** / 时间分辨率 | Daily / 日 | Half-hourly / 半小时 | Half-hourly / 半小时 |
| **Conceptual Basis** / 概念基础 | WUE theory / WUE理论 | Machine learning / 机器学习 | Stomatal optimization / 气孔优化 |
| **Data Requirements** / 数据需求 | Low / 低 | Moderate / 中等 | High / 高 |
| **Computational Cost** / 计算成本 | Low / 低 | High / 高 | High / 高 |
| **Physical Basis** / 物理基础 | Empirical / 经验性 | Semi-empirical / 半经验性 | Mechanistic / 机理性 |
| **Sensitivity to Stress** / 对胁迫的敏感性 | Low / 低 | High / 高 | High / 高 |
| **Ease of Implementation** / 实施难度 | Easy / 容易 | Moderate / 中等 | Difficult / 困难 |

### 5.2 When to Use Each Method / 何时使用每种方法

**uWUE is Recommended When / 推荐使用uWUE:**
- Daily resolution is sufficient / 日分辨率足够
- Limited computational resources / 计算资源有限
- Stable, well-characterized ecosystem / 稳定、特征明确的生态系统
- Long-term climatology is the focus / 关注长期气候学

**TEA is Recommended When / 推荐使用TEA:**
- Half-hourly resolution needed / 需要半小时分辨率
- Sufficient training data available / 有足够的训练数据
- Complex environmental interactions / 复杂的环境交互
- Water stress conditions are important / 水分胁迫条件重要

**Perez-Priego is Recommended When / 推荐使用Perez-Priego:**
- Mechanistic understanding is priority / 优先考虑机理理解
- Site elevation data available / 有站点高程数据
- Stomatal control is key / 气孔控制是关键
- High-resolution analysis needed / 需要高分辨率分析

### 5.3 Performance Considerations / 性能考虑

**Typical Performance Metrics / 典型性能指标:**

Based on multi-site validation studies:

基于多站点验证研究：

```
Method          RMSE_T (mm/day)   R²_T    Bias_T (mm/day)
uWUE            0.5-1.0          0.6-0.8    ±0.2
TEA             0.4-0.9          0.7-0.9    ±0.15
Perez-Priego    0.4-0.8          0.7-0.9    ±0.1
```

**Note / 注意:** Performance varies significantly by:
- Ecosystem type / 生态系统类型
- Climate regime / 气候体系
- Data quality / 数据质量
- Calibration period / 校准期

### 5.4 Cross-PFT Performance / 跨PFT性能

Different methods perform better for different PFTs:

不同方法对不同PFT的表现不同：

**Forests (ENF, DBF, EBF) / 森林:**
- All methods perform well
- Perez-Priego slightly better for evergreen forests
- uWUE good for deciduous forests

**Shrublands and Savannas (CSH, OSH, WSA) / 灌丛和稀树草原:**
- TEA and Perez-Priego better capture soil evaporation
- uWUE may underestimate E in open canopies

**Grasslands and Croplands (GRA, CRO) / 草地和农田:**
- TEA shows best performance
- High temporal variability challenges all methods
- Irrigation complicates partitioning

**Wetlands (WET) / 湿地:**
- All methods face challenges
- High E component difficult to capture
- Site-specific calibration recommended

---

<a name="simulation-framework"></a>
## 6. Simulation Framework / 模拟框架

### 6.1 Purpose / 目的

The simulation framework in `analysis/` allows controlled experiments to:

`analysis/`中的模拟框架允许进行可控实验，以：

1. **Evaluate method performance** under known truth / 在已知真值下评估方法性能
2. **Test sensitivity** to environmental conditions / 测试对环境条件的敏感性
3. **Compare methods** across PFT scenarios / 跨PFT场景比较方法
4. **Identify failure modes** and limitations / 识别失效模式和局限性

### 6.2 PFT Scenario Design / PFT场景设计

Each PFT scenario is characterized by:

每个PFT场景由以下特征：

```python
PFTScenario(
    name="ENF",                    # Identification
    canopy_conductance=0.9,        # Physiological capacity
    vpd_sensitivity=0.6,           # Stomatal response
    soil_evap_fraction=0.25,       # Structural characteristics
    photosynthesis_efficiency=1.2, # Biochemical capacity
    interception_ratio=0.35,       # Canopy interception
    noise_std=0.05,                # Measurement uncertainty
    transpiration_bias=1.05        # Systematic error
)
```

### 6.3 Synthetic Data Generation / 合成数据生成

The synthetic flux data includes:

合成通量数据包括：

**Meteorological Drivers / 气象驱动:**
- Seasonal radiation cycle / 季节性辐射循环
- Diurnal temperature variation / 昼夜温度变化
- Stochastic precipitation events / 随机降水事件
- Realistic VPD dynamics / 真实的VPD动态

**Physiological Responses / 生理响应:**
```python
GPP = efficiency × radiation × exp(-vpd_sensitivity × VPD) × SWC
T = conductance × f(radiation, VPD, SWC) × bias
E = soil_fraction × (1 - SWC) + interception × precipitation
```

**Realistic Noise / 真实噪声:**
- Measurement uncertainty / 测量不确定性
- Gap-filling errors / 插补误差
- Instrument drift / 仪器漂移

### 6.4 Performance Metrics / 性能指标

Comprehensive evaluation includes:

综合评估包括：

**Basic Metrics / 基础指标:**
- RMSE: Root Mean Square Error / 均方根误差
- MAE: Mean Absolute Error / 平均绝对误差
- Bias: Systematic error / 系统误差
- R: Pearson correlation / 皮尔逊相关

**Advanced Metrics / 高级指标:**
- NSE: Nash-Sutcliffe Efficiency / Nash-Sutcliffe效率
- KGE: Kling-Gupta Efficiency / Kling-Gupta效率
- Relative Error: Normalized RMSE / 归一化RMSE

**Conditional Metrics / 条件指标:**
- Performance by season / 按季节的性能
- Performance by soil moisture / 按土壤水分的性能
- Performance by time of day / 按时间的性能

### 6.5 Usage Example / 使用示例

```python
from analysis import (
    PartitionComparison,
    PFT_ENF, PFT_DBF, PFT_GRA,
    visualization
)

# Define scenarios
scenarios = [PFT_ENF, PFT_DBF, PFT_GRA]

# Run comparison
comparison = PartitionComparison(
    scenarios,
    n_days=180,
    seed=42,
    include_seasonal_analysis=True
)
results = comparison.run()

# Analyze results
results_df = comparison.results_to_dataframe(results)
summary_df = comparison.aggregate_metrics(results_df)

# Visualize
fig = visualization.plot_performance_heatmap(results_df, metric="rmse_T")
fig.savefig("performance_heatmap.png")
```

---

<a name="references"></a>
## 7. References / 参考文献

### Primary Method Papers / 主要方法论文

**uWUE Method:**
> Zhou, S., Yu, B., Zhang, Y., Huang, Y., & Wang, G. (2016). Partitioning evapotranspiration based on the concept of underlying water use efficiency. *Water Resources Research*, 52(2), 1160-1175.  
> DOI: [10.1002/2015WR017766](https://doi.org/10.1002/2015WR017766)

**TEA Method:**
> Nelson, J. A., Carvalhais, N., Migliavacca, M., Reichstein, M., & Jung, M. (2018). Water-stress-induced breakdown of carbon–water relations: indicators from diurnal FLUXNET patterns. *Biogeosciences*, 15(8), 2433-2447.  
> DOI: [10.5194/bg-15-2433-2018](https://doi.org/10.5194/bg-15-2433-2018)

**Perez-Priego Method:**
> Perez-Priego, O., et al. (2018). Partitioning eddy covariance water flux components using physiological and micrometeorological approaches. *Journal of Geophysical Research: Biogeosciences*, 123(10), 3353-3370.  
> DOI: [10.1029/2018JG004637](https://doi.org/10.1029/2018JG004637)

### Theoretical Background / 理论背景

**Water Use Efficiency:**
> Katul, G., Manzoni, S., Palmroth, S., & Oren, R. (2010). A stomatal optimization theory to describe the effects of atmospheric CO₂ on leaf photosynthesis and transpiration. *Annals of Botany*, 105(3), 431-442.

**Penman-Monteith Equation:**
> Monteith, J. L. (1965). Evaporation and environment. *Symposia of the Society for Experimental Biology*, 19, 205-234.

**Quantile Regression:**
> Koenker, R., & Hallock, K. F. (2001). Quantile regression. *Journal of Economic Perspectives*, 15(4), 143-156.

### Comparison Studies / 对比研究

> Wei, Z., Yoshimura, K., Wang, L., Miralles, D. G., Jasechko, S., & Lee, X. (2017). Revisiting the contribution of transpiration to global terrestrial evapotranspiration. *Geophysical Research Letters*, 44(6), 2792-2801.

> Stoy, P. C., et al. (2019). Reviews and syntheses: Turning the challenges of partitioning ecosystem evaporation and transpiration into opportunities. *Biogeosciences*, 16(19), 3747-3775.

### PFT Classification / PFT分类

> Friedl, M. A., et al. (2010). MODIS Collection 5 global land cover: Algorithm refinements and characterization of new datasets. *Remote Sensing of Environment*, 114(1), 168-182.

---

## Appendices / 附录

### A. Notation and Symbols / 符号说明

| Symbol | Description | Units |
|--------|-------------|-------|
| ET | Total evapotranspiration / 总蒸散发 | mm day⁻¹ or W m⁻² |
| T | Transpiration / 蒸腾 | mm day⁻¹ or W m⁻² |
| E | Evaporation / 蒸发 | mm day⁻¹ or W m⁻² |
| GPP | Gross Primary Production / 总初级生产力 | μmol CO₂ m⁻² s⁻¹ |
| VPD | Vapor Pressure Deficit / 水汽压差 | kPa |
| WUE | Water Use Efficiency / 水分利用效率 | various |
| SWC | Soil Water Content / 土壤含水量 | m³ m⁻³ |
| gc | Canopy conductance / 冠层导度 | mol m⁻² s⁻¹ |
| ga | Aerodynamic conductance / 气动导度 | mol m⁻² s⁻¹ |
| Rn | Net radiation / 净辐射 | W m⁻² |

### B. Software Implementation / 软件实现

All methods are implemented in Python 3.10+ with dependencies:
- numpy >= 1.23
- pandas >= 1.5
- scikit-learn >= 1.3 (for TEA)
- matplotlib >= 3.7 (for visualization)

所有方法均使用Python 3.10+实现，依赖项：
- numpy >= 1.23
- pandas >= 1.5
- scikit-learn >= 1.3 (用于TEA)
- matplotlib >= 3.7 (用于可视化)

---

**Document Version:** 2.0  
**Last Updated:** 2025-10-30  
**Authors:** Li Changming (licm13) with AI assistance (Claude)

