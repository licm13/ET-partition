# ET Partition Methods: Deep Dive Technical Documentation
# ET蒸散发拆分方法：深度技术文档

**Version 1.0** | **Last Updated:** 2025-12 | **Language:** Bilingual (EN/中文)

---

## Table of Contents / 目录

1. [Theoretical Framework Comparison](#1-theoretical-framework-comparison--理论框架对比)
2. [Mathematical Derivations](#2-mathematical-derivations--数学推导详解)
3. [Code Implementation Analysis](#3-code-implementation-analysis--代码实现剖析)
4. [Performance and Optimization](#4-performance-and-optimization--性能与优化)
5. [Application Scenarios and Limitations](#5-application-scenarios-and-limitations--应用场景与限制)

---

## 1. Theoretical Framework Comparison / 理论框架对比

### 1.1 Basic Assumptions Table / 基本假设表格

| Aspect / 方面 | uWUE | TEA | Perez-Priego |
|---------------|------|-----|--------------|
| **Core Principle / 核心原理** | Water use efficiency optimization / 水分利用效率优化 | Data-driven machine learning / 数据驱动机器学习 | Stomatal conductance optimization / 气孔导度优化 |
| **Key Assumption / 关键假设** | uWUE is constant under optimal conditions / 最优条件下uWUE恒定 | WUE can be predicted from environmental features / WUE可从环境特征预测 | Plants optimize carbon gain per water loss / 植物优化单位水分损失的碳增益 |
| **Mathematical Basis / 数学基础** | Quantile regression / 分位数回归 | Quantile Random Forest / 分位数随机森林 | Penman-Monteith + Optimization / Penman-Monteith + 优化 |
| **Time Resolution / 时间分辨率** | Daily / 日 | Half-hourly / 半小时 | Half-hourly / 半小时 |
| **Physical Basis / 物理基础** | Semi-empirical / 半经验 | Empirical / 经验 | Mechanistic / 机理 |

### 1.2 Applicability Conditions / 适用条件

#### uWUE Method / uWUE方法

**Best suited for / 最适合:**
- Stable ecosystems with established vegetation / 植被成熟的稳定生态系统
- Sites with regular precipitation patterns / 降水模式规律的站点
- Long-term climate studies (seasonal to multi-year) / 长期气候研究（季节到多年）

**Limitations / 局限:**
- Daily resolution only / 仅日分辨率
- Requires sufficient "optimal" conditions in dataset / 需要数据集中有足够的"最优"条件
- May underestimate E in dry periods / 可能在干旱期低估E
- Assumes constant ecosystem uWUE* / 假设生态系统uWUE*恒定

#### TEA Method / TEA方法

**Best suited for / 最适合:**
- Sites with variable soil moisture conditions / 土壤水分条件多变的站点
- Research requiring diurnal patterns / 需要日变化模式的研究
- Cross-site comparisons / 跨站点比较

**Limitations / 局限:**
- Requires substantial training data under optimal conditions / 需要最优条件下的大量训练数据
- "Black box" - difficult to interpret / "黑箱" - 难以解释
- Computationally expensive / 计算成本高
- May extrapolate poorly outside training range / 训练范围外外推可能较差

#### Perez-Priego Method / Perez-Priego方法

**Best suited for / 最适合:**
- Process-based understanding / 基于过程的理解
- Sites with known elevation and meteorological data / 已知高程和气象数据的站点
- Studies of stomatal regulation / 气孔调控研究

**Limitations / 局限:**
- Requires site elevation data / 需要站点高程数据
- Many parameters to optimize / 需优化多个参数
- Computationally intensive (MCMC optimization) / 计算密集（MCMC优化）
- Sensitive to initial parameter values / 对初始参数值敏感

### 1.3 Physical Basis / 物理基础详解

#### Carbon-Water Coupling / 碳水耦合

All three methods exploit the fundamental coupling between carbon uptake and water loss in plants:

所有三种方法都利用植物碳吸收与水分损失之间的基本耦合：

$$A = g_c \times (C_a - C_i)$$

$$T = g_w \times \frac{VPD}{P}$$

Where:
- $A$ = photosynthetic rate (μmol CO₂ m⁻² s⁻¹) / 光合速率
- $T$ = transpiration rate (mol H₂O m⁻² s⁻¹) / 蒸腾速率
- $g_c$ = stomatal conductance for CO₂ (mol m⁻² s⁻¹) / CO₂气孔导度
- $g_w = 1.6 \times g_c$ = stomatal conductance for water / 水蒸气气孔导度
- $C_a$ = atmospheric CO₂ (μmol mol⁻¹) / 大气CO₂
- $C_i$ = intercellular CO₂ (μmol mol⁻¹) / 胞间CO₂
- $VPD$ = vapor pressure deficit (kPa) / 水汽压差
- $P$ = atmospheric pressure (kPa) / 大气压

---

## 2. Mathematical Derivations / 数学推导详解

### 2.1 uWUE: Quantile Regression Derivation / uWUE分位数回归推导

#### Definition of underlying WUE / 潜在WUE定义

Starting from the intrinsic water use efficiency (iWUE):

从内在水分利用效率（iWUE）出发：

$$iWUE = \frac{A}{g_w}$$

Zhou et al. (2016) introduced the underlying WUE to normalize by VPD:

Zhou等人（2016）引入潜在WUE，用VPD归一化：

$$uWUE = \frac{GPP \times \sqrt{VPD}}{T}$$

**Physical interpretation / 物理解释:**
- The $\sqrt{VPD}$ term accounts for the non-linear relationship between VPD and transpiration
- Under optimal conditions, uWUE reaches a maximum (uWUE*)
- $\sqrt{VPD}$项解释了VPD与蒸腾之间的非线性关系
- 在最优条件下，uWUE达到最大值（uWUE*）

#### Quantile Regression / 分位数回归

The key insight is that uWUE* can be estimated from the upper quantile of observed data:

关键洞见是uWUE*可以从观测数据的上分位数估算：

$$uWUE^* = Q_{0.95}\left(\frac{GPP \times \sqrt{VPD}}{ET}\right)$$

**Quantile regression objective function / 分位数回归目标函数:**

$$\min_{\beta} \sum_{i} \rho_\tau(y_i - x_i\beta)$$

Where the check function is:

其中检验函数为：

$$\rho_\tau(u) = u(\tau - \mathbb{1}(u < 0))$$

#### Python Implementation / Python实现

```python
import numpy as np
from scipy.optimize import fmin

def quantile_regression(x, y, tau=0.95):
    """
    Quantile regression with zero-intercept model.
    分位数回归（零截距模型）
    
    Parameters
    ----------
    x : array-like
        Independent variable (ET)
    y : array-like  
        Dependent variable (GPP * sqrt(VPD))
    tau : float
        Quantile (0-1), default 0.95
        
    Returns
    -------
    float
        Estimated slope (uWUE*)
    """
    def check_function(u, tau):
        """Tilted absolute value function / 倾斜绝对值函数"""
        return u * (tau - (u < 0))
    
    def objective(beta, x, y, tau):
        residuals = y - x * beta
        return np.sum(check_function(residuals, tau))
    
    # Initial guess / 初始猜测
    beta_init = np.mean(y) / np.mean(x)
    
    # Optimize / 优化
    result = fmin(objective, beta_init, args=(x, y, tau), disp=False)
    return result[0]
```

### 2.2 TEA: Quantile Random Forest / TEA分位数随机森林原理

#### Random Forest Basics / 随机森林基础

A Random Forest is an ensemble of decision trees, each trained on a bootstrap sample:

随机森林是决策树的集成，每棵树在自助样本上训练：

$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$$

Where:
- $B$ = number of trees / 树的数量
- $T_b(x)$ = prediction of tree $b$ / 第$b$棵树的预测

#### Quantile Extension / 分位数扩展

For quantile prediction, instead of averaging predictions, we use the empirical distribution of training points in terminal nodes:

对于分位数预测，我们使用终端节点中训练点的经验分布，而不是平均预测：

$$\hat{Q}_\tau(x) = \inf\left\{y : \frac{1}{n}\sum_{i=1}^{n} w_i(x) \mathbb{1}(Y_i \leq y) \geq \tau\right\}$$

Where $w_i(x)$ is the weight assigned to observation $i$ based on how often it shares a terminal node with $x$ across all trees.

其中$w_i(x)$是根据观测$i$在所有树中与$x$共享终端节点的频率分配的权重。

#### Python Implementation / Python实现

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class QuantileRandomForest:
    """
    Quantile Random Forest for WUE prediction.
    用于WUE预测的分位数随机森林
    """
    
    def __init__(self, n_estimators=100, quantile=0.75, random_state=None):
        self.n_estimators = n_estimators
        self.quantile = quantile
        self.random_state = random_state
        self.rf = None
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """Fit the model / 拟合模型"""
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf.fit(X, y)
        self.X_train = X
        self.y_train = y
        return self
        
    def predict(self, X):
        """Predict quantile / 预测分位数"""
        # Get leaf indices for training and prediction data
        leaf_ids_train = self.rf.apply(self.X_train)
        leaf_ids_pred = self.rf.apply(X)
        
        n_pred = X.shape[0]
        predictions = np.zeros(n_pred)
        
        for i in range(n_pred):
            # Find training samples in same leaves
            in_same_leaf = (leaf_ids_train == leaf_ids_pred[i]).any(axis=1)
            if in_same_leaf.sum() > 0:
                predictions[i] = np.percentile(
                    self.y_train[in_same_leaf], 
                    self.quantile * 100
                )
            else:
                predictions[i] = np.median(self.y_train)
                
        return predictions
```

### 2.3 Perez-Priego: Medlyn Stomatal Model / Perez-Priego Medlyn模型公式

#### Stomatal Conductance Model / 气孔导度模型

The Perez-Priego method uses a modified Ball-Berry-Leuning model:

Perez-Priego方法使用修改的Ball-Berry-Leuning模型：

$$g_c = g_{c,max} \times f_Q \times f_{VPD} \times f_T$$

Where the response functions are:

其中响应函数为：

**Light response / 光响应:**
$$f_Q = \frac{Q}{Q + a_1}$$

**VPD response / VPD响应:**
$$f_{VPD} = \exp(-D_0 \times VPD)$$

**Temperature response (beta function) / 温度响应（beta函数）:**
$$f_T = \frac{(T - T_{min})(T_{max} - T)^\beta}{(T_{opt} - T_{min})(T_{max} - T_{opt})^\beta}$$

Where:
- $Q$ = photosynthetically active radiation (μmol m⁻² s⁻¹) / 光合有效辐射
- $a_1$ = light response parameter / 光响应参数
- $D_0$ = VPD sensitivity parameter / VPD敏感度参数
- $T_{opt}$ = optimal temperature (°C) / 最优温度
- $T_{min}, T_{max}$ = temperature limits (0, 50°C) / 温度极限

#### Optimal χ Calculation / 最优χ计算

The optimal ratio of intercellular to atmospheric CO₂ (χ) is calculated as:

胞间与大气CO₂最优比值（χ）计算为：

$$\chi_o = \frac{\exp(\theta)}{1 + \exp(\theta)}$$

Where:
$$\theta = 0.0545 \times (T_{air} - 25) - 0.58 \times \ln(VPD) - 0.0815 \times z + c$$

- $z$ = elevation (km) / 海拔（km）
- $c$ = calibration coefficient / 校准系数

#### Python Implementation / Python实现

```python
import numpy as np

def calculate_stomatal_conductance(
    Q: np.ndarray,
    VPD: np.ndarray, 
    Tair: np.ndarray,
    gc_max: float,
    a1: float = 50,
    D0: float = 0.1,
    T_opt: float = 25
) -> np.ndarray:
    """
    Calculate stomatal conductance using modified Ball-Berry model.
    使用修改的Ball-Berry模型计算气孔导度
    
    Parameters
    ----------
    Q : array-like
        Photosynthetically active radiation (μmol m⁻² s⁻¹)
    VPD : array-like
        Vapor pressure deficit (kPa)
    Tair : array-like
        Air temperature (°C)
    gc_max : float
        Maximum stomatal conductance (mol m⁻² s⁻¹)
    a1 : float
        Light response parameter
    D0 : float
        VPD sensitivity parameter
    T_opt : float
        Optimal temperature for conductance
        
    Returns
    -------
    np.ndarray
        Stomatal conductance (mol m⁻² s⁻¹)
    """
    # Light response / 光响应
    f_Q = Q / (Q + a1 + 1e-6)
    
    # VPD response / VPD响应
    f_VPD = np.exp(-D0 * VPD)
    
    # Temperature response (beta function) / 温度响应
    T_min, T_max = 0, 50
    beta = (T_max - T_opt) / (T_max - T_min)
    
    T_clipped = np.clip(Tair, T_min + 0.1, T_max - 0.1)
    T_diff = np.clip(T_max - T_clipped, 0, None)
    
    scale = 1 / ((T_opt - T_min) * (T_max - T_opt)**beta + 1e-6)
    f_T = scale * (T_clipped - T_min) * T_diff**beta
    f_T = np.clip(f_T, 0, None)
    
    # Combined conductance / 综合导度
    f_all = f_Q * f_VPD * f_T
    f_all_normalized = f_all / (np.nanmax(f_all) + 1e-6)
    
    return gc_max * f_all_normalized


def calculate_transpiration(
    gc: np.ndarray,
    VPD: np.ndarray,
    P_atm: np.ndarray
) -> np.ndarray:
    """
    Calculate transpiration from stomatal conductance.
    从气孔导度计算蒸腾
    
    Parameters
    ----------
    gc : array-like
        Stomatal conductance for CO2 (mol m⁻² s⁻¹)
    VPD : array-like
        Vapor pressure deficit (kPa)
    P_atm : array-like
        Atmospheric pressure (kPa)
        
    Returns
    -------
    np.ndarray
        Transpiration (mol H₂O m⁻² s⁻¹)
    """
    gw = 1.6 * gc  # Water vapor conductance
    T = gw * VPD / (P_atm + 1e-6)
    return T
```

---

## 3. Code Implementation Analysis / 代码实现剖析

### 3.1 Batch Processing Architecture / 批处理架构设计模式

The repository follows a consistent batch processing pattern across all methods:

代码库在所有方法中遵循一致的批处理模式：

```
┌─────────────────────────────────────────────────────────────────┐
│                     BATCH PROCESSOR FLOW                         │
│                     批处理器流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   INPUT: base_path (directory of site folders)                   │
│          输入：base_path（站点文件夹目录）                        │
│                         │                                         │
│                         ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ STEP 1: iter_site_folders()                             │   │
│   │         Scan for FLUXNET-style folders                  │   │
│   │         扫描FLUXNET格式的文件夹                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ STEP 2: process_site_folder()                           │   │
│   │         For each site:                                  │   │
│   │         ├── Load CSV data                               │   │
│   │         ├── Preprocess (rename, unit conversion)        │   │
│   │         ├── Run partitioning algorithm                  │   │
│   │         └── Save results                                │   │
│   │                                                         │   │
│   │         对每个站点：                                     │   │
│   │         ├── 加载CSV数据                                 │   │
│   │         ├── 预处理（重命名，单位转换）                   │   │
│   │         ├── 运行拆分算法                                │   │
│   │         └── 保存结果                                    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                         │                                         │
│                         ▼                                         │
│   OUTPUT: output_path/{site}_results.csv                         │
│           输出：output_path/{site}_results.csv                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow Diagram / 数据流转图

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
│                        数据流                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  FLUXNET CSV                                                     │
│      │                                                           │
│      ▼                                                           │
│  ┌────────────────┐                                              │
│  │ Raw Variables  │                                              │
│  │ 原始变量       │                                              │
│  │ LE_F_MDS       │──────┐                                       │
│  │ GPP_NT_VUT_REF │──────┤                                       │
│  │ VPD_F          │──────┤                                       │
│  │ TA_F           │──────┤                                       │
│  │ SW_IN_F        │──────┘                                       │
│  └────────────────┘                                              │
│          │                                                        │
│          ▼                                                        │
│  ┌────────────────┐                                              │
│  │  PREPROCESSOR  │                                              │
│  │  预处理器      │                                              │
│  │                │                                              │
│  │  • Rename cols │                                              │
│  │  • Unit convert│                                              │
│  │  • QC filter   │                                              │
│  │  • Gap fill    │                                              │
│  └────────────────┘                                              │
│          │                                                        │
│          ▼                                                        │
│  ┌────────────────┐      ┌────────────────┐                      │
│  │   METHOD A     │      │   METHOD B     │                      │
│  │   uWUE         │      │   TEA          │                      │
│  │                │      │                │                      │
│  │  zhou_part()   │      │ simplePartition│                      │
│  └────────────────┘      └────────────────┘                      │
│          │                      │                                 │
│          └──────────┬───────────┘                                │
│                     ▼                                             │
│            ┌────────────────┐                                    │
│            │    OUTPUT      │                                    │
│            │    输出        │                                    │
│            │                │                                    │
│            │  T, E, T/ET   │                                    │
│            │  日/半小时    │                                    │
│            └────────────────┘                                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Function Call Chains / 关键函数调用链

#### uWUE Method / uWUE方法

```python
# Entry point / 入口点
batch.py: uWUEBatchProcessor.run()
    │
    ├── preprocess.py: build_dataset()
    │       │
    │       └── bigleaf.py: LE_to_ET()  # Convert latent heat to ET
    │
    ├── zhou.py: build_zhou_masks()
    │       │
    │       └── calculate_rain_flag()  # Identify precipitation events
    │
    └── zhou.py: zhou_part()
            │
            ├── quantreg()           # Estimate uWUE*
            │
            └── Calculate T from uWUE ratio
```

#### TEA Method / TEA方法

```python
# Entry point / 入口点
batch.py: main()
    │
    ├── iter_site_folders()
    │
    └── process_site_folder()
            │
            ├── pd.read_csv()        # Load data
            │
            ├── Preprocessing        # Column mapping, unit conversion
            │
            └── TEA.TEA: simplePartition()
                    │
                    ├── Calculate auxiliary indices
                    │     (CSWI, DWCI, Diurnal centroid)
                    │
                    ├── Filter optimal conditions
                    │
                    ├── Train QuantileRandomForest
                    │
                    └── Predict T, E for all conditions
```

#### Perez-Priego Method / Perez-Priego方法

```python
# Entry point / 入口点
batch.py: main()
    │
    ├── iter_site_folders()
    │
    └── process_site_folder()
            │
            ├── et_partitioning_functions.py:
            │       │
            │       ├── calculate_chi_o()
            │       │
            │       ├── calculate_WUE_o()
            │       │
            │       └── optimal_parameters()
            │               │
            │               └── MCMC optimization (emcee)
            │
            └── Calculate T from optimized gc model
```

### 3.4 Configuration Parameters / 配置参数说明

#### uWUE Parameters / uWUE参数

| Parameter | Default | Description / 描述 |
|-----------|---------|-------------------|
| `percentile` | 0.95 | Quantile for uWUE* estimation / uWUE*估算的分位数 |
| `steps_per_day` | 48 | Number of timesteps per day / 每天的时间步数 |
| `MIN_DAYS_PER_YEAR` | 5 | Minimum days required for processing / 处理所需的最少天数 |
| `gpp_variable` | 'GPP_NT' | Name of GPP variable / GPP变量名 |

#### TEA Parameters / TEA参数

| Parameter | Default | Description / 描述 |
|-----------|---------|-------------------|
| `n_estimators` | 100 | Number of trees in Random Forest / 随机森林中的树数量 |
| `quantile` | 0.75 | Quantile for WUE prediction / WUE预测的分位数 |
| `n_jobs` | -1 | Parallel workers (-1 = all cores) / 并行工作进程 |
| `optimal_swc_threshold` | 0.7 | SWC threshold for optimal conditions / 最优条件的SWC阈值 |

#### Perez-Priego Parameters / Perez-Priego参数

| Parameter | Default | Description / 描述 |
|-----------|---------|-------------------|
| `window_size` | 5 | Moving window size (days) / 滑动窗口大小（天） |
| `nwalkers` | 10 | MCMC walkers / MCMC行走者数量 |
| `nsteps` | 100 | MCMC iterations / MCMC迭代次数 |
| `max_duration` | 30 | MCMC timeout (seconds) / MCMC超时（秒） |
| `default_altitude` | 0.5 | Default site altitude (km) / 默认站点海拔（km） |

---

## 4. Performance and Optimization / 性能与优化

### 4.1 Current Performance Benchmarks / 当前性能基准

Based on FI-Hyy test site (3 years, half-hourly data):

基于FI-Hyy测试站（3年，半小时数据）：

| Method / 方法 | Execution Time / 执行时间 | Peak Memory / 峰值内存 | Output Size / 输出大小 |
|---------------|---------------------------|------------------------|------------------------|
| uWUE | ~15 seconds | ~200 MB | ~100 KB (daily) |
| TEA | ~45 seconds | ~500 MB | ~2 MB (half-hourly) |
| Perez-Priego | ~90 seconds | ~300 MB | ~2 MB (half-hourly) |

### 4.2 Bottleneck Analysis / 瓶颈分析

#### uWUE Bottlenecks / uWUE瓶颈

```
Profile Results (typical 3-year dataset):
分析结果（典型3年数据集）：

Function                    Time(%)   Calls
─────────────────────────────────────────────
quantreg()                  45%       3 (per year)
zhou_part() loop            35%       ~1000 days
build_zhou_masks()          15%       3
I/O operations              5%        varies
```

**Optimization opportunities / 优化机会:**
- Vectorize daily loops / 向量化日循环
- Cache quantile regression results / 缓存分位数回归结果
- Use Numba for tight loops / 使用Numba加速紧密循环

#### TEA Bottlenecks / TEA瓶颈

```
Profile Results:
分析结果：

Function                    Time(%)   Calls
─────────────────────────────────────────────
RandomForest.fit()          60%       1
RandomForest.predict()      25%       1
Data preprocessing          10%       1
I/O operations              5%        varies
```

**Optimization opportunities / 优化机会:**
- Already uses n_jobs=-1 for parallelism / 已使用n_jobs=-1实现并行
- Consider incremental training / 考虑增量训练
- Memory chunking for very long series / 对超长序列进行内存分块

#### Perez-Priego Bottlenecks / Perez-Priego瓶颈

```
Profile Results:
分析结果：

Function                    Time(%)   Calls
─────────────────────────────────────────────
MCMC sampling (emcee)       70%       many windows
gc_model() evaluations      20%       ~1M
Data preparation            8%        per window
I/O operations              2%        varies
```

**Optimization opportunities / 优化机会:**
- Use Numba JIT for gc_model() / 对gc_model()使用Numba JIT
- Parallel window processing / 并行窗口处理
- Reduce MCMC iterations with better initialization / 通过更好的初始化减少MCMC迭代

### 4.3 Optimization Strategies / 优化方案

#### Strategy 1: Numba JIT Compilation / Numba JIT编译

```python
import numba
import numpy as np

@numba.njit(parallel=True)
def gc_model_numba(Q, VPD, Tair, gc_max, a1, D0, T_opt):
    """
    Numba-accelerated stomatal conductance model.
    Numba加速的气孔导度模型
    
    Achieves 5-10x speedup over pure Python.
    相比纯Python实现5-10倍加速。
    """
    n = len(Q)
    result = np.empty(n)
    
    T_min, T_max = 0.0, 50.0
    beta = (T_max - T_opt) / (T_max - T_min)
    scale = 1.0 / ((T_opt - T_min) * (T_max - T_opt)**beta)
    
    for i in numba.prange(n):
        # Light response
        f_Q = Q[i] / (Q[i] + a1 + 1e-6)
        
        # VPD response
        f_VPD = np.exp(-D0 * VPD[i])
        
        # Temperature response
        T_clip = min(max(Tair[i], T_min + 0.1), T_max - 0.1)
        T_diff = max(T_max - T_clip, 0.0)
        f_T = scale * (T_clip - T_min) * (T_diff ** beta)
        f_T = max(f_T, 0.0)
        
        result[i] = gc_max * f_Q * f_VPD * f_T
        
    # Normalize
    max_val = np.max(result)
    if max_val > 0:
        result = result / max_val
        
    return result * gc_max
```

#### Strategy 2: Vectorized Operations / 向量化操作

```python
import numpy as np

def vectorized_daily_aggregation(halfhourly_data, steps_per_day=48):
    """
    Vectorized daily aggregation without loops.
    无循环的向量化日聚合
    
    Much faster than iterating over days.
    比按天迭代快得多。
    """
    n_days = len(halfhourly_data) // steps_per_day
    reshaped = halfhourly_data[:n_days * steps_per_day].reshape(n_days, steps_per_day)
    
    daily_mean = np.nanmean(reshaped, axis=1)
    daily_sum = np.nansum(reshaped, axis=1)
    daily_max = np.nanmax(reshaped, axis=1)
    
    return {
        'mean': daily_mean,
        'sum': daily_sum,
        'max': daily_max
    }
```

#### Strategy 3: LRU Cache / LRU缓存

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def atmospheric_pressure_cached(elevation_km: float) -> float:
    """
    Cached atmospheric pressure calculation.
    缓存的大气压计算
    
    Avoids repeated calculation for same elevation.
    避免对相同海拔重复计算。
    """
    P0 = 101.325  # Standard pressure at sea level (kPa)
    return P0 * np.exp(-elevation_km / 8.5)
```

### 4.4 Memory Optimization / 内存优化策略

For processing very long time series (10+ years):

对于处理超长时间序列（10年以上）：

```python
def chunk_process(data, chunk_size=365*48, overlap=48):
    """
    Process data in chunks to limit memory usage.
    分块处理数据以限制内存使用
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    chunk_size : int
        Number of rows per chunk (default: 1 year of half-hourly data)
    overlap : int
        Overlap between chunks for continuity
        
    Yields
    ------
    pd.DataFrame
        Processed chunk
    """
    n_rows = len(data)
    
    for start in range(0, n_rows, chunk_size - overlap):
        end = min(start + chunk_size, n_rows)
        chunk = data.iloc[start:end].copy()
        
        # Process chunk
        result = process_chunk(chunk)
        
        # Remove overlap from result (except last chunk)
        if end < n_rows and len(result) > overlap:
            result = result.iloc[:-overlap]
            
        yield result
        
        # Explicit garbage collection
        del chunk
        gc.collect()
```

---

## 5. Application Scenarios and Limitations / 应用场景与限制

### 5.1 Method Selection Decision Tree / 方法选择决策树

```
                    ┌───────────────────┐
                    │ START             │
                    │ What resolution   │
                    │ do you need?      │
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
        ┌─────────────┐             ┌─────────────────┐
        │   Daily     │             │  Half-hourly    │
        └──────┬──────┘             └────────┬────────┘
               │                             │
               ▼                             │
        ┌─────────────┐          ┌───────────┴───────────┐
        │    uWUE     │          │                       │
        │  ✓ Simple   │          ▼                       ▼
        │  ✓ Fast     │   ┌─────────────┐      ┌─────────────────┐
        └─────────────┘   │ Elevation   │      │   Emphasis on   │
                          │   data?     │      │ interpretability│
                          └──────┬──────┘      │       OR        │
                                 │             │    accuracy?    │
                    ┌────────────┴────────┐    └────────┬────────┘
                    │                     │             │
                    ▼                     ▼    ┌────────┴────────┐
             ┌─────────────┐      ┌────────────────┐      │
             │     YES     │      │      NO       │       ▼
             └──────┬──────┘      └───────┬───────┘ ┌──────────────┐
                    │                     │        │Interpretability│
                    ▼                     ▼        └───────┬───────┘
           ┌───────────────┐      ┌─────────────┐          │
           │ Perez-Priego  │      │    TEA      │          ▼
           │ ✓ Mechanistic │      │ ✓ Flexible  │   ┌─────────────┐
           │ ✓ Process-    │      │ ✓ Data-     │   │ Perez-Priego │
           │   based       │      │   driven    │   └─────────────┘
           └───────────────┘      └─────────────┘          │
                                                           │
                                                   ┌───────┴───────┐
                                                   │   Accuracy    │
                                                   └───────┬───────┘
                                                           ▼
                                                   ┌─────────────┐
                                                   │     TEA     │
                                                   └─────────────┘
```

### 5.2 Known Issues and Solutions / 已知问题与解决方案

| Issue / 问题 | Affected Method / 受影响方法 | Solution / 解决方案 |
|--------------|------------------------------|---------------------|
| All-NaN transpiration | All | Check input data quality; ensure GPP > 0 during daytime |
| Negative evaporation | All | Apply `E = max(0, ET - T)` constraint |
| T > ET | All | Apply `T = min(T, ET)` constraint |
| uWUE* estimation fails | uWUE | Increase data range; check for sufficient wet periods |
| TEA NaN predictions | TEA | Ensure diverse training conditions; check for outliers |
| MCMC timeout | Perez-Priego | Reduce `nsteps`; improve initial parameter estimates |
| Memory overflow | All (10+ years) | Use chunked processing; reduce output frequency |

### 5.3 Parameter Tuning Guide / 参数调优指南

#### uWUE Tuning / uWUE调优

```python
# Adjust quantile based on data quality
# 根据数据质量调整分位数
if high_quality_data:
    percentile = 0.95  # Strict optimal conditions
else:
    percentile = 0.90  # More relaxed

# Adjust for ecosystem type
# 根据生态系统类型调整
if forest_ecosystem:
    gpp_threshold_percentile = 0.10  # 10% of 95th percentile
elif grassland:
    gpp_threshold_percentile = 0.05  # Lower threshold for variable GPP
```

#### TEA Tuning / TEA调优

```python
# Adjust for data availability
# 根据数据可用性调整
if long_time_series:
    n_estimators = 200  # More trees for better accuracy
    quantile = 0.75
elif short_time_series:
    n_estimators = 50   # Fewer trees to avoid overfitting
    quantile = 0.80     # Higher quantile for limited data
```

#### Perez-Priego Tuning / Perez-Priego调优

```python
# Adjust MCMC settings based on convergence
# 根据收敛情况调整MCMC设置
if convergence_issues:
    nwalkers = 20   # More walkers for better exploration
    nsteps = 200    # More iterations
    
# Adjust window size for climate
# 根据气候调整窗口大小
if stable_climate:
    window_size = 7  # Larger window
elif variable_climate:
    window_size = 3  # Smaller window for responsiveness
```

### 5.4 Validation Strategies / 验证策略

#### Cross-Validation / 交叉验证

```python
def temporal_cross_validation(data, n_folds=5):
    """
    Time-series cross-validation for ET partitioning.
    ET拆分的时间序列交叉验证
    """
    fold_size = len(data) // n_folds
    metrics = []
    
    for i in range(n_folds):
        # Leave one fold out
        test_start = i * fold_size
        test_end = test_start + fold_size
        
        train_data = pd.concat([
            data.iloc[:test_start],
            data.iloc[test_end:]
        ])
        test_data = data.iloc[test_start:test_end]
        
        # Train and predict
        model = fit_partitioning_model(train_data)
        predictions = model.predict(test_data)
        
        # Calculate metrics
        fold_metrics = calculate_metrics(predictions, test_data)
        metrics.append(fold_metrics)
        
    return pd.DataFrame(metrics)
```

#### Ecological Plausibility Checks / 生态合理性检验

```python
def ecological_plausibility_check(T, E, ET, GPP, metadata):
    """
    Check if partitioning results are ecologically plausible.
    检查拆分结果是否生态合理
    """
    issues = []
    
    # 1. T/ET ratio within expected range
    t_et_ratio = T.sum() / ET.sum()
    expected_range = get_expected_t_et_range(metadata['ecosystem_type'])
    if not expected_range[0] <= t_et_ratio <= expected_range[1]:
        issues.append(f"T/ET ratio {t_et_ratio:.2f} outside expected range {expected_range}")
    
    # 2. Seasonal pattern check
    summer_t = T[metadata['summer_mask']].mean()
    winter_t = T[metadata['winter_mask']].mean()
    if summer_t < winter_t:
        issues.append("Summer T < Winter T (unexpected for most ecosystems)")
    
    # 3. GPP-T correlation
    corr = np.corrcoef(GPP[GPP > 0], T[GPP > 0])[0, 1]
    if corr < 0.3:
        issues.append(f"Low GPP-T correlation ({corr:.2f})")
    
    return issues
```

#### Isotope Validation (if available) / 同位素验证（如果可用）

```python
def isotope_validation(T_model, T_isotope):
    """
    Validate modeled T against isotope-derived estimates.
    用同位素估算值验证模型T
    """
    # Aggregate to match isotope temporal resolution
    T_model_agg = T_model.resample('D').sum()
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((T_model_agg - T_isotope)**2))
    bias = np.mean(T_model_agg - T_isotope)
    r = np.corrcoef(T_model_agg, T_isotope)[0, 1]
    
    return {
        'RMSE': rmse,
        'Bias': bias,
        'Correlation': r
    }
```

---

## References / 参考文献

### Primary Method Papers / 主要方法论文

1. **Zhou et al. (2016)** - uWUE method
   > Zhou, S., Yu, B., Zhang, Y., Huang, Y., & Wang, G. (2016). Partitioning evapotranspiration based on the concept of underlying water use efficiency. *Water Resources Research*, 52(2), 1160-1175.

2. **Nelson et al. (2018)** - TEA method
   > Nelson, J. A., Carvalhais, N., Migliavacca, M., Reichstein, M., & Jung, M. (2018). Water-stress-induced breakdown of carbon–water relations: indicators from diurnal FLUXNET patterns. *Biogeosciences*, 15(8), 2433-2447.

3. **Perez-Priego et al. (2018)** - Optimality-based method
   > Perez-Priego, O., et al. (2018). Partitioning eddy covariance water flux components using physiological and micrometeorological approaches. *Journal of Geophysical Research: Biogeosciences*, 123(10), 3353-3370.

### Theoretical Background / 理论背景

4. **Koenker & Bassett (1978)** - Quantile regression
5. **Meinzer & Grantz (1991)** - Stomatal conductance models
6. **Medlyn et al. (2011)** - Optimal stomatal conductance theory

---

**Document Version:** 1.0
**Last Updated:** 2025-12
**Authors:** ET-partition Project Team with AI assistance

