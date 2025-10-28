# TEA模型批量处理代码文档



本文档旨在为 **TEA (Transpiration-Evaporation-WUE) 模型批量处理代码** 提供全面的架构分析、使用指南和维护信息，确保用户在长期中断后仍能快速理解和使用本工具。



## 1. 代码架构分析





### 1.1 整体架构图和模块关系



代码段

```
graph TD
    A[用户] --> B(TEA_batch.py);
    B --> C{TEA Library};
    C --> D[PreProc.py];
    C --> E[TEA.py];
    C --> F[PotRad.py];
    C --> G[CSWI.py];
    C --> H[DWCI.py];
    C --> I[DiurnalCentroid.py];
    E --> J[core.py];

    subgraph "主运行脚本"
        B
    end

    subgraph "核心TEA库"
        C
    end

    subgraph "数据预处理与特征工程"
        D
        F
        G
        H
        I
    end

    subgraph "核心分区算法"
        E
    end

    subgraph "机器学习模型"
        J
    end

    subgraph "数据输入/输出"
        K(AmeriFLUX 数据集) --> B;
        B --> L(分区结果CSV);
    end
```



### 1.2 主要组件和它们的职责



| 模块 (文件)                | 主要职责                                                     |
| -------------------------- | ------------------------------------------------------------ |
| **TEA_batch.py**           | **主控脚本**。负责遍历指定的AmeriFLUX数据文件夹，自动化执行数据读取、预处理、调用TEA模型，并保存结果。 |
| **TEA/PreProc.py**         | **数据预处理模块**。构建xarray数据集，计算派生变量（如CSWI, DWCI, 质心等），并应用数据质量过滤器。 |
| **TEA/TEA.py**             | **核心分区模块**。包含`partition`和`simplePartition`函数，执行基于随机森林的蒸散（ET）分区计算。 |
| **TEA/core.py**            | **分位数随机森林**。实现了`QuantileRandomForestRegressor`类，这是TEA算法进行非参数化建模的基础。 |
| **TEA/DWCI.py**            | **日变化水碳耦合指数 (DWCI)**。计算ET和GPP在日尺度上的耦合强度，作为模型的一个重要预测因子。 |
| **TEA/DiurnalCentroid.py** | **日质心计算模块**。计算通量的日加权平均时间，用于捕捉物候或胁迫引起的日循环模式变化。 |
| **TEA/CSWI.py**            | **保守地表水指数 (CSWI)**。基于简化的水量平衡模型计算，反映短期土壤湿度状况。 |
| **TEA/PotRad.py**          | **潜在辐射计算模块**。根据地理位置和时间计算潜在太阳辐射，用于数据过滤和标准化。 |



### 1.3 数据流向和处理流程



1. **启动**: 用户运行 `TEA_batch.py`。
2. **文件扫描**: 脚本扫描 `BASE_PATH` 目录下符合 `FOLDER_PATTERN` 命名规则的站点文件夹。
3. **数据加载**: 对每个站点，找到对应的半小时（HH）CSV数据文件并用Pandas加载。
4. **初步处理**: 在 `TEA_batch.py` 中，进行必要的列筛选、重命名和单位转换（如将 `LE` 从 W/m² 转换为 mm/30min）。
5. **调用核心库**: 将处理后的数据（作为NumPy数组）传递给 `TEA.TEA.simplePartition` 函数。
6. **构建数据集**: 在 `simplePartition` 内部，首先调用 `PreProc.build_dataset` 将NumPy数组封装成一个结构化的 `xarray.Dataset`。
7. **特征工程**: 接着，调用 `PreProc.preprocess` 计算一系列派生变量（特征），包括：
   - `CSWI` (来自 `CSWI.py`)
   - `DWCI` (来自 `DWCI.py`)
   - `C_Rg_ET` (来自 `DiurnalCentroid.py`)
   - `GPPgrad`、`Rgpotgrad` 等梯度和统计量。
8. **模型训练与预测**: `TEA.partition` 函数被调用。它根据预设的条件（如白 天、生长季等）筛选出高质量数据，用这些数据训练一个**分位数随机森林回归模型** (`core.QuantileRandomForestRegressor`)，目标变量是瞬时水分利用效率 (`inst_WUE`)。然后，该模型对整个数据集进行预测。
9. **ET分区**: 利用预测的 `WUE` (水分利用效率) 和输入的 `GPP` (总初级生产力)，计算出**蒸腾 (T)**。
10. **计算蒸发 (E)**: 通过总蒸散减去计算出的蒸腾得到蒸发 (`E = ET - T`)。
11. **结果返回**: `simplePartition` 返回计算出的 `T`, `E`, 和 `WUE`。
12. **保存输出**: `TEA_batch.py` 将返回的结果整理成一个DataFrame，并以站点名命名，保存为CSV文件到 `OUTPUT_PATH`。



### 1.4 依赖关系和调用链



- `TEA_batch.py` 依赖于 `TEA` 库（特别是 `TEA.TEA.simplePartition`）。
- `TEA.py` 依赖于 `TEA.PreProc` 和 `TEA.core`。
- `PreProc.py` 依赖于 `TEA.CSWI`, `TEA.DiurnalCentroid`, 和 `TEA.DWCI`。
- `core.py` 依赖于 `sklearn` 和 `numba` 来实现随机森林算法。
- 几乎所有模块都依赖于 `numpy` 和 `pandas`。

------



## 2. 功能说明





### 2.1 代码的主要功能和用途



本代码库的主要功能是实现**蒸散（Evapotranspiration, ET）分区**，即将其分解为**植被蒸腾（Transpiration, T）和地表蒸发（Evaporation, E）两个组分。它采用了一种名为TEA (Transpiration-Evaporation-WUE partitioning)** 的数据驱动方法，该方法基于机器学习（分位数随机森林）来估算水分利用效率（WUE），并依此进行分区。



### 2.2 解决的具体问题



在全球碳水循环研究中，准确区分T和E至关重要，因为T与光合作用（GPP）紧密耦合，而E则不受生物过程直接影响。本代码解决了如何利用涡度相关通量站点（如AmeriFLUX）的常规观测数据，在半小时尺度上对ET进行稳健、可重复的分区。



### 2.3 适用场景和限制条件



- **适用场景**:
  - 适用于处理来自**AmeriFLUX**或**FLUXNET**等通量网络的标准化半小时数据集。
  - 站点数据必须包含TEA模型所需的输入变量（详见3.1）。
  - 适合对大量站点进行自动化、批量的ET分区计算。
- **限制条件**:
  - **数据质量**: 算法对输入数据的质量敏感。缺失值、异常值或质量差的通量数据会影响分区结果的准确性。`TEA_batch.py`目前对数据质量的控制较为简单，仅跳过缺失必需列的文件。
  - **模型假设**: TEA方法假设在特定环境条件下（如高土壤湿度），观测到的`GPP/ET`比率可以代表潜在的`WUE`。此假设的有效性可能因生态系统类型和环境条件而异。
  - **计算资源**: 随机森林计算量较大，处理长时间序列或大量站点可能需要较长时间。

------



## 3. 输入输出规范





### 3.1 详细的输入参数说明



该程序的主要输入是标准的**AmeriFLUX全集（FULLSET）半小时（HH）CSV文件**。`TEA_batch.py`会自动从这些文件中提取所需的列。

| 原始列名         | 类型    | 格式/单位        | 说明                             |
| ---------------- | ------- | ---------------- | -------------------------------- |
| `LE_F_MDS`       | `float` | W m⁻²            | 潜热通量，将内部转换为蒸散（ET） |
| `GPP_NT_VUT_REF` | `float` | umol CO₂ m⁻² s⁻¹ | 总初级生产力（GPP）              |
| `TA_F_MDS`       | `float` | °C               | 空气温度                         |
| `RH`             | `float` | %                | 相对湿度                         |
| `VPD_F_MDS`      | `float` | hPa              | 饱和水汽压差                     |
| `P_ERA`          | `float` | mm               | 降水                             |
| `SW_IN_F`        | `float` | W m⁻²            | 向下短波辐射 (实测)              |
| `WS`             | `float` | m s⁻¹            | 风速                             |
| `SW_IN_POT`      | `float` | W m⁻²            | 潜在向下短波辐射                 |



### 3.2 输出结果格式和含义



程序会为每个成功处理的站点生成一个CSV文件，文件名格式为 `{SITENAME}_TEA_results.csv`。

| 输出列名    | 类型    | 格式/单位    | 含义                                                         |
| ----------- | ------- | ------------ | ------------------------------------------------------------ |
| `timestamp` | `int`   | 分钟         | 从0开始，步长为30的连续时间戳，用于与输入数据对齐。          |
| `TEA_T`     | `float` | mm/30min     | **蒸腾 (Transpiration)**：由TEA模型计算出的植被蒸腾水量。    |
| `TEA_E`     | `float` | mm/30min     | **蒸发 (Evaporation)**：通过 `ET - TEA_T` 计算出的地表蒸发水量。 |
| `TEA_WUE`   | `float` | g C / kg H₂O | **水分利用效率 (Water Use Efficiency)**：由随机森林模型预测的`GPP/T`比率。 |



### 3.3 错误处理和异常情况



- **文件夹/文件缺失**: 如果找不到站点文件夹或对应的CSV文件，将在控制台打印提示信息并跳过。
- **必需列缺失**: 如果CSV文件中缺少任何一个必需的输入列，将打印警告并跳过该文件。
- **CSV读取错误**: 如果文件格式损坏导致Pandas无法读取，将捕获异常，打印错误信息并跳过。
- **算法内部错误**: 如果TEA核心算法在计算过程中（例如，由于数据全为NaN）失败，可能导致输出结果为-9999或NaN。

------



## 4. 使用指南





### 4.1 环境配置和依赖安装



建议使用`conda`或`venv`创建独立的Python环境。

Bash

```
# 创建并激活conda环境
conda create -n tea_env python=3.9
conda activate tea_env

# 安装核心依赖
pip install pandas numpy scikit-learn xarray numba
```



### 4.2 逐步运行流程



1. **准备数据**:

   - 将所有AmeriFLUX站点文件夹（格式如`AMF_US-Ne1_FLUXNET_FULLSET_...`）放置在一个根目录下。
   - 确保每个文件夹内包含对应的 `..._FULLSET_HH_....csv` 文件。

2. **配置脚本**:

   - 打开 `TEA_batch.py` 文件。
   - 修改 **配置区 (Configuration Section)** 的以下变量：
     - `BASE_PATH`: 设置为存放AmeriFLUX站点文件夹的根目录路径。
     - `OUTPUT_PATH`: 设置为您希望保存结果的目录路径。
     - `FOLDER_PATTERN`: 如果您的文件夹命名规则与默认不同，请修改此正则表达式。

3. **运行程序**:

   - 在激活了Python环境的终端中，导航到 `TEA_batch.py` 所在的目录。

   - 执行脚本：

     Bash

     ```
     python TEA_batch.py
     ```

4. **检查结果**:

   - 程序运行期间，控制台会实时显示当前正在处理的文件夹和进度。
   - 运行结束后，前往您设置的 `OUTPUT_PATH` 目录，检查生成的 `_TEA_results.csv` 文件。



### 4.3 配置文件说明



本程序没有独立的配置文件，所有配置项都在 `TEA_batch.py` 的**配置区**内，以Python变量的形式存在。这种方式简化了部署，但修改配置需要直接编辑代码。



### 4.4 常见用法示例



**场景：处理一批下载自AmeriFLUX网站的Tier 1数据**

1. 将下载的所有站点文件夹解压到 `D:\AmeriFLUX_Data`。

2. 在 `TEA_batch.py` 中修改配置：

   Python

   ```
   BASE_PATH = r'D:\AmeriFLUX_Data'
   OUTPUT_PATH = r'D:\AmeriFLUX_Data\TEA_Results'
   ```

3. 打开终端，`cd`到脚本目录，运行 `python TEA_batch.py`。

4. 等待程序执行完毕，结果将出现在 `D:\AmeriFLUX_Data\TEA_Results` 文件夹中。

------



## 5. 核心逻辑解析





### 5.1 关键算法和实现原理



- **核心算法**: **分位数随机森林回归 (Quantile Random Forest Regression)**
  - **原理**: 标准的随机森林通过平均所有决策树的叶节点值来做预测。而分位数随机森林则保留了每个预测点对应的所有叶节点值，形成一个完整的预测分布。通过对这个分布取分位数（例如75%），可以得到一个更稳健的、对异常值不敏感的预测。
  - **实现**: `core.py` 中的 `QuantileRandomForestRegressor` 类包装了 `sklearn` 的随机森林，但重写了 `predict` 方法。它首先获取每个预测样本在所有树中落入的叶节点ID，然后收集这些叶节点在**训练时**所包含的所有`trainy`（目标变量）值，最后使用`numpy.quantile`计算所需的分位数。
  - **加速**: 关键的聚合步骤 (`find_quant`) 使用 `numba` 的 `@jit` 装饰器进行了即时编译优化，显著提升了计算速度。
- **TEA分区逻辑**:
  1. **识别理想条件**: 算法假设在水分充足（通过`CSWI`判断）且生态系统活跃（通过`GPPFlag`, `tempFlag`等判断）的条件下，蒸发E被抑制到最小，此时测得的ET几乎全部是蒸腾T。
  2. **训练模型**: 在这些理想条件下，瞬时水分利用效率 (`inst_WUE = GPP / ET`) 近似等于潜在（或内在）水分利用效率 (`WUE = GPP / T`)。算法使用这些“理想”数据点来训练随机森林模型，学习 `WUE` 与一系列环境变量（`RFmod_vars`）之间的非线性关系。
  3. **全局预测**: 将训练好的模型应用于整个数据集，预测出所有时间点的`WUE`。
  4. **计算T和E**: 利用公式 `T = GPP / WUE` 和 `E = ET - T` 完成分区。



### 5.2 重要函数的作用和参数



- `TEA.TEA.simplePartition(...)`:
  - **作用**: 封装了从原始数据到最终分区结果的全过程，是 `TEA_batch.py` 调用的主入口。
  - **参数**: 接收一系列NumPy数组作为输入，如`ET`, `GPP`, `Tair`等。
- `TEA.TEA.partition(ds, ...)`:
  - **作用**: 执行核心的随机森林训练和预测。
  - **参数**:
    - `ds`: 预处理后的xarray数据集。
    - `percs`: 指定预测的`WUE`分位数，默认75%。
    - `CSWIlims`: 定义用于筛选训练数据的`CSWI`阈值。
    - `RFmod_vars`: 一个列表，定义了哪些变量被用作随机森林的预测因子（特征）。



### 5.3 核心业务逻辑流程



核心逻辑集中在`TEA.TEA.partition`函数中：

1. **构建特征矩阵 `RFxs`**: 从输入的`ds`中提取`RFmod_vars`中指定的变量，并堆叠成一个 `(n_samples, n_features)` 的NumPy数组。
2. **定义数据过滤器**:
   - `Baseflag`: 一个布尔掩码，整合了多个条件（白天、生长季、数据有效等），用于筛选高质量数据。
   - `CurFlag`: 在`Baseflag`的基础上，进一步根据`CSWI`阈值筛选出水分充足的、用于模型训练的数据点。
3. **训练随机森林**: 如果`CurFlag`中有足够的数据点（>240），则实例化`QuantileRandomForestRegressor`并使用`RFxs[CurFlag]`和`ds.inst_WUE.values[CurFlag]`进行`fit`。
4. **进行分位数预测**: 调用`qrf.predict()`方法，对**所有**有效数据点（`DayNightFlag` & `nanflag`）进行预测，得到`TEA_WUE`。
5. **计算T和E**: 根据公式进行最终的计算。



### 5.4 性能考量和优化点



- **Numba加速**: `core.py`中对`find_quant`的JIT编译是关键的性能优化，否则分位数计算会非常缓慢。
- **内存**: `xarray`和`numpy`操作是内存效率较高的。但处理超长时序（数十年）数据时，仍需关注内存占用。
- **并行计算**: `QuantileRandomForestRegressor`通过`n_jobs`参数支持多线程训练随机森林，可以显著缩短训练时间。
- **可优化点**: `TEA_batch.py` 目前是串行处理每个站点。对于多核服务器，可以很容易地修改主循环，使用`multiprocessing`库来并行处理多个站点，从而大幅提升整体处理效率。

------



## 6. 维护信息



- **创建时间**: 2025-07-16
- **作者**: Changming Li & Gemini (AI assistant)
- **版本信息**: 1.1 (根据 `TEA_batch.py` 内的注释)
  - **1.0 (Initial)**: 初始批处理版本。
  - **1.1 (2025-07-12)**: 增加了详细的注释，改进了代码结构和可读性。
- **已知问题**:
  - 数据质量控制较为初级，依赖于输入数据的预先筛选。
  - 站点处理为串行，在大规模数据处理时效率有提升空间。
- **TODO项**:
  - [ ] 实现基于`multiprocessing`的站点级并行处理。
  - [ ] 增加更详细的日志记录功能，方便问题排查。
  - [ ] 允许用户通过命令行参数指定`BASE_PATH`和`OUTPUT_PATH`，而不是硬编码。