# uWUE 模型批量处理代码文档



本文档旨在为 **uWUE (underlying Water Use Efficiency) 模型批量处理代码** 提供全面的架构分析、使用指南和维护信息，确保用户在长期中断后仍能快速理解和使用本工具。



## 1. 代码架构分析





### 1.1 整体架构图和模块关系



代码段

```
graph TD
    A[用户] --> B(uwue_batch.py);
    B --> C{preprocess.py};
    B --> D{bigleaf.py};
    B --> E{zhou.py};

    subgraph "主控与批处理"
        B
    end

    subgraph "数据加载与预处理"
        C
    end

    subgraph "物理/生理计算"
        D
    end

    subgraph "核心分区算法"
        E
    end

    subgraph "输入/输出与配置"
        F(FLUXNET CSV 数据) --> B;
        G(BerkeleyConversion.json) --> C;
        H(Units.json) --> C;
        I(LongNames.json) --> C;
        B --> J(分析结果 .csv/.nc);
        B --> K(可视化图表 .png);
        B --> L(日志 .log);
    end
```



### 1.2 主要组件和它们的职责



| 模块 (文件)         | 主要职责                                                     |
| ------------------- | ------------------------------------------------------------ |
| **`uwue_batch.py`** | **主控脚本与批处理器**。封装了整个处理流程，负责： 1. 扫描数据目录，识别有效站点。 2. 循环处理每个站点。 3. 调用其他模块执行具体计算。 4. 保存结果、生成图表和总结报告。 |
| **`preprocess.py`** | **数据加载与预处理模块**。`build_dataset_modified` 函数负责：<br>1. 从FLUXNET格式的CSV文件中高效加载所需数据。<br>2. 利用`BerkeleyConversion.json`进行列重命名。<br>3. 将数据转换为结构化的`xarray.Dataset`。<br>4. 利用`Units.json`和`LongNames.json`为变量添加元数据（单位和长名称）。 |
| **`bigleaf.py`**    | **生物气象学计算库**。提供一系列物理和生理生态学计算函数，例如： 1. `LE_to_ET`：将潜热通量（LE）转换为蒸散（ET）。<br>2. `PET`：计算潜在蒸散发（PET）。 3. 其他辅助函数，如计算饱和水汽压、空气密度等。 |
| **`zhou.py`**       | **Zhou uWUE 分区算法实现**。包含实现Zhou et al. (2016)方法的核心逻辑： 1. `zhouFlags`：根据降雨、生长季、数据质量等条件生成用于区分“实际”和“潜在”uWUE计算的数据掩码。<br>2. `quantreg`：实现分位数回归，用于估算潜在水分利用效率（uWUEp）。<br>3. `zhou_part`：执行核心分区计算，估算蒸腾（T）。 |
| **JSON 配置文件**   | **配置与元数据**： 1. `BerkeleyConversion.json`：定义了从原始CSV列名到程序内部变量名的映射。<br>2. `Units.json` / `LongNames.json`：为变量提供单位和详细描述，增强数据的可解释性。 |



### 1.3 数据流向和处理流程



1. **启动与扫描**：用户运行 `uwue_batch.py`。`uWUEBatchProcessor` 类被实例化，扫描 `BASE_PATH` 寻找符合 `folder_pattern` 的站点文件夹。
2. **数据加载**：对于每个有效站点，`process_single_site` 函数调用 `preprocess.build_dataset_modified`。该函数读取对应的 `..._FULLSET_HH_....csv` 文件，并根据 `BerkeleyConversion.json` 筛选和重命名列，最终返回一个 `xarray.Dataset` 对象 (`ec`)。
3. **物理量计算**：在 `_perform_uwue_analysis` 方法中，脚本调用 `bigleaf.py` 中的函数进行计算：
   - `bigleaf.LE_to_ET` 将 `LE` 和 `TA` 转换为 `ET` (mm/timestep)。
   - 缺失的 `NETRAD` 通过能量平衡闭合（`LE` + `H` + `G`）进行填充。
   - `bigleaf.PET` 根据 `TA`, `PA`, `NETRAD` 等计算潜在蒸散发 `PET`。
4. **uWUE 掩码生成**：调用 `zhou.zhouFlags`，根据降雨、数据质量、生长季等条件，生成用于区分 uWUEa 和 uWUEp 计算的布尔掩码 (`uWUEa_Mask`, `uWUEp_Mask`)。
5. **核心分区计算**：脚本按年份循环处理数据。在每年循环中：
   - 调用 `zhou.zhou_part` 函数。
   - 内部，`zhou.quantreg` 使用分位数回归（默认95%）和 `uWUEp_Mask` 筛选的数据来估算该年度的**潜在水分利用效率 (uWUEp)**。
   - `zhou.zhou_part` 在日尺度或8日滑动窗口上，使用普通最小二乘法和 `uWUEa_Mask` 筛选的数据估算**实际水分利用效率 (uWUEa)**。
   - 最终，**蒸腾与蒸散的比率 (T/ET)** 被估算为 `uWUEa / uWUEp`，并计算出每日的蒸腾量（`zhou_T` 和 `zhou_T_8day`）。
6. **结果整合与保存**：
   - 计算出的每日蒸腾量被存入一个新的 `xarray.Dataset` (`ds_zhou`)。
   - `_save_results` 方法将结果保存为 `.csv` 和 `.nc` 文件。
7. **可视化与报告**：
   - 如果 `create_plots` 为 `True`，`_create_visualization` 方法会生成包含时间序列、散点图和月平均值的分析图表并保存为 `.png`。
   - 所有站点处理完毕后，`generate_summary_report` 生成一个包含处理统计信息的文本报告。



### 1.4 依赖关系和调用链



- `uwue_batch.py` 是顶层控制器，直接调用 `preprocess.py`、`bigleaf.py` 和 `zhou.py`。
- `preprocess.py` 依赖于三个 `.json` 配置文件来加载和格式化数据。
- `zhou.py` 依赖于 `numpy` 和 `scipy.optimize` 来实现分位数回归。
- `bigleaf.py` 依赖于 `sympy` 和 `numpy` 来进行符号计算和数值运算。
- 整个项目依赖于 `pandas`, `xarray`, `numpy`, `matplotlib`, 和 `seaborn`。

------



## 2. 功能说明





### 2.1 代码的主要功能和用途



本代码库的主要功能是**批量处理FLUXNET通量数据，并应用Zhou et al. (2016) 提出的基于潜在水分利用效率（uWUE）的方法，对生态系统总蒸散发（Evapotranspiration, ET）进行分区**，即将其分解为**植被蒸腾（Transpiration, T）和地表蒸发（Evaporation, E）**。



### 2.2 解决的具体问题



在全球碳水循环研究中，准确区分T和E至关重要，因为T与光合作用（GPP）紧密耦合，而E则主要受物理因素驱动。该代码库提供了一个自动化、可重复的框架，用于从标准的通量塔观测数据中估算T和E，为生态水文学、气候模型评估等研究提供关键数据产品。



### 2.3 适用场景和限制条件



- **适用场景**:
  - 处理遵循FLUXNET2015或类似格式的半小时涡度相关通量数据。
  - 对大量站点进行自动化、标准化的ET分区计算。
  - 需要生成ET分区结果及其可视化图表的研究。
- **限制条件**:
  - **数据要求**: 输入的CSV文件必须包含`BerkeleyConversion.json`中定义的所有变量列，否则`preprocess.py`在加载时会失败。
  - **方法假设**: uWUE方法的准确性依赖于其核心假设，即在非降雨、生长季期间观测到的WUE可以代表潜在WUE。此假设在某些干旱或特殊生态系统中可能不完全成立。
  - **数据质量**: 分区结果对输入数据（特别是`LE`, `GPP`, `VPD`, `P`）的质量非常敏感。代码内部虽然有QC标志的检查，但无法完全修正原始数据的系统性偏差。
  - **按年计算**: uWUEp是按年计算的，这假设了植被性状在年内是相对稳定的，可能不适用于年际变化剧烈的生态系统。

------



## 3. 输入输出规范





### 3.1 详细的输入参数说明



该程序的主要输入是标准的**FLUXNET2015全集（FULLSET）半小时（HH）CSV文件**。`preprocess.build_dataset_modified` 会自动根据 `BerkeleyConversion.json` 提取所需列。关键输入变量包括：

| 原始列名             | 类型    | 格式/单位       | 说明                   |
| -------------------- | ------- | --------------- | ---------------------- |
| `TIMESTAMP_START`    | `int`   | YYYYMMDDHHMM    | 时间戳                 |
| `LE_F_MDS`           | `float` | W m⁻²           | 潜热通量               |
| `GPP_NT_VUT_USTAR50` | `float` | umolCO₂ m⁻² s⁻¹ | 夜间法估算的GPP        |
| `VPD_F_MDS`          | `float` | hPa             | 饱和水汽压差           |
| `P`                  | `float` | mm              | 降水                   |
| `TA_F_MDS`           | `float` | °C              | 空气温度               |
| `PA`                 | `float` | kPa             | 大气压                 |
| `NETRAD`             | `float` | W m⁻²           | 净辐射                 |
| `H_F_MDS`            | `float` | W m⁻²           | 感热通量               |
| `G_F_MDS`            | `float` | W m⁻²           | 土壤热通量             |
| `*_QC`               | `int`   | 0-3             | 对应变量的质量控制标志 |



### 3.2 输出结果格式和含义



对于每个成功处理的站点，程序会生成三种输出文件：

1. **CSV结果文件 (`{sitename}_uWUE_output.csv`)**:
   - 包含每日的`ET`, `zhou_T`, 和 `zhou_T_8day`。

| 输出列名      | 类型       | 格式/单位  | 含义                                   |
| ------------- | ---------- | ---------- | -------------------------------------- |
| `time`        | `datetime` | YYYY-MM-DD | 日期                                   |
| `ET`          | `float`    | mm d⁻¹     | 每日总蒸散发                           |
| `zhou_T`      | `float`    | mm d⁻¹     | uWUE估算的每日蒸腾（日尺度uWUEa）      |
| `zhou_T_8day` | `float`    | mm d⁻¹     | uWUE估算的每日蒸腾（8日滑动窗口uWUEa） |

1. **NetCDF结果文件 (`{sitename}_uWUE_output.nc`)**:
   - 与CSV内容相同，但以`xarray.Dataset`格式存储，保留了变量的元数据（如`long_name`, `units`）和全局属性（如处理日期、uWUEp值等）。
2. **可视化图表 (`plots/{sitename}_uWUE_analysis.png`)**:
   - 一个PNG图像，包含四个子图，直观展示分区结果。



### 3.3 错误处理和异常情况



- **文件/目录缺失**: 如果 `BASE_PATH` 不存在或找不到对应的CSV文件，程序会记录错误并跳过。
- **数据加载失败**: 如果CSV文件损坏或缺少`BerkeleyConversion.json`中定义的列，`build_dataset_modified`会抛出异常，`process_single_site`会捕获该异常，记录错误并跳过该站点。
- **计算失败**: 如果在uWUE分析过程中出现意外的数值问题（例如，数据全为NaN），可能会导致结果为NaN。
- **绘图失败**: 如果一个站点处理后没有有效的数值结果，绘图将被跳过，并记录警告。

------



## 4. 使用指南





### 4.1 环境配置和依赖安装



建议使用`conda`或`venv`创建独立的Python环境。

Bash

```
# 创建并激活conda环境
conda create -n uwue_env python=3.9
conda activate uwue_env

# 安装核心依赖
pip install pandas numpy xarray scipy sympy matplotlib seaborn
```



### 4.2 逐步运行流程



1. **准备数据**:

   - 将所有FLUXNET站点文件夹（格式如`FLX_SITE-ID_FLUXNET2015_...`）放置在一个根目录下。
   - 确保每个文件夹内包含对应的 `..._FULLSET_HH_....csv` 文件。

2. **配置脚本**:

   - 打开 `uwue_batch.py` 文件。
   - 在 `main()` 函数中，修改以下变量：
     - `BASE_PATH`: 设置为存放FLUXNET站点文件夹的根目录路径。
     - `OUTPUT_PATH`: 设置为您希望保存结果的目录路径。
     - `CREATE_PLOTS`: 设置为 `True` 或 `False`，以决定是否生成图表。

3. **运行程序**:

   - 在激活了Python环境的终端中，导航到 `uwue_batch.py` 所在的目录。

   - 执行脚本：

     Bash

     ```
     python uwue_batch.py
     ```

4. **检查结果**:

   - 程序运行期间，控制台和日志文件会实时显示处理进度和信息。
   - 运行结束后，前往您设置的 `OUTPUT_PATH` 目录，检查生成的 `.csv`, `.nc`, `.png` 文件以及 `processing_summary.txt` 报告。



### 4.3 配置文件说明



本程序使用三个JSON文件进行配置：

- `BerkeleyConversion.json`: **必须配置**。定义了从CSV加载哪些列以及如何重命名它们。如果CSV中缺少此文件定义的列，程序会出错。
- `Units.json` & `LongNames.json`: **可选但建议配置**。为变量提供元数据，如果一个变量没有在这里定义，它在输出的NetCDF文件中将没有单位和长名称信息。



### 4.4 常见用法示例



**场景：处理一批下载自FLUXNET2015的Tier-2数据**

1. 将下载的所有站点文件夹解压到 `C:\Data\FLUXNET2015_Tier2`。

2. 在 `uwue_batch.py` 的 `main` 函数中修改配置：

   Python

   ```
   BASE_PATH = 'C:\\Data\\FLUXNET2015_Tier2'
   OUTPUT_PATH = 'C:\\Data\\uWUE_Results'
   CREATE_PLOTS = True
   ```

3. 打开终端，`cd`到脚本目录，运行 `python uwue_batch.py`。

4. 等待程序执行完毕，结果将出现在 `C:\Data\uWUE_Results` 文件夹中。

------



## 5. 核心逻辑解析





### 5.1 关键算法和实现原理



- **核心算法**: **基于潜在水分利用效率（uWUE）的分区方法**
  - **原理**: 该方法基于一个核心假设：生态系统的水分利用效率（WUE = GPP/T）在不受水分胁迫时会达到一个由VPD主导的潜在上限（uWUEp）。而在其他时候，实际的WUE（uWUEa）会低于这个上限。因此，蒸腾占总蒸散的比例（T/ET）可以近似为 `uWUEa / uWUEp`。
  - **uWUE的定义**: uWUE 被定义为 `(GPP * VPD^0.5) / T`。这个公式移除了VPD对WUE的主要影响，使得uWUE更能反映植被的内在生理特性。由于T难以直接测量，在计算中用ET近似，即 `uWUE ≈ (GPP * VPD^0.5) / ET`。
  - **uWUEp的估算**: uWUEp（潜在值）被认为是uWUE分布的**上边界**。代码通过**分位数回归**（`quantreg`函数，默认对95%分位数进行拟合）来估算这个上边界。该回归只在特定的“理想”条件下（`uWUEp_Mask`：非降雨、生长旺盛期）进行。
  - **uWUEa的估算**: uWUEa（实际值）通过在每日或8日滑动窗口内对 `GPP * VPD^0.5` 和 `ET` 进行普通最小二乘回归来估算，使用的是更宽松的数据筛选条件（`uWUEa_Mask`）。



### 5.2 重要函数的作用和参数



- `zhou.zhou_part(ET, GxV, uWUEa_Mask, uWUEp_Mask, ...)`:
  - **作用**: 执行核心的分区计算，是Zhou方法的直接实现。
  - **参数**:
    - `ET`: 蒸散发数组。
    - `GxV`: `GPP * VPD^0.5` 的数组。
    - `uWUEa_Mask`, `uWUEp_Mask`: 用于筛选数据的布尔掩码。
    - `rho`: 分位数回归的分位数，默认为0.95。
- `zhou.quantreg(x, y, rho, ...)`:
  - **作用**: 实现分位数回归，用于找到uWUEp。它通过最小化一个“倾斜绝对值”目标函数来找到拟合参数，这是分位数回归的标准方法。



### 5.3 核心业务逻辑流程



`uwue_batch.py`中的`_perform_uwue_analysis`方法是核心业务逻辑的体现：

1. **数据准备**：加载数据后，计算派生变量 `ET` 和 `PET`，并填充 `NETRAD`。
2. **条件筛选**：调用`zhou.zhouFlags`，根据物理和生物条件创建两个关键的数据子集：
   - `uWUEp_Mask`：用于估算**潜在能力**的“黄金标准”数据（晴天、生长季、无水分胁迫）。
   - `uWUEa_Mask`：用于估算**实际表现**的、更广泛的数据集。
3. **分层计算**：按年份进行循环，确保uWUEp的估算是针对特定年份的植被状况。
4. **模型拟合与分区**：在年循环中调用`zhou.zhou_part`：
   - 使用`uWUEp_Mask`的数据和分位数回归，确定当年的`uWUEp`。
   - 使用`uWUEa_Mask`的数据和线性回归，计算每日的`uWUEa`。
   - 计算`T = ET * (uWUEa / uWUEp)`。
5. **结果聚合**：将每年计算出的每日T值拼接起来，形成完整时间序列的结果。



### 5.4 性能考量和优化点



- **数据加载**: `preprocess.py`中的`build_dataset_modified`通过`usecols`参数只读取必要的列，避免了加载整个大型CSV文件，提升了IO和内存效率。
- **计算**: `xarray`和`numpy`的向量化操作是高效的。主要的计算瓶颈在于`zhou.quantreg`中的`fmin`优化函数，因为它是一个迭代过程。
- **并行化**: 当前脚本是**串行处理**每个站点。一个显著的优化点是修改`run`方法的主循环，使用Python的`multiprocessing`库来并行处理多个站点文件夹，这在多核CPU上可以大幅缩短总处理时间。
- **内存管理**: 对于非常长的记录（例如，>20年），一次性加载所有数据到内存可能成为问题。虽然当前实现没有分块处理，但`xarray`与`dask`的结合可以实现这一点，不过会增加代码的复杂性。

------



## 6. 维护信息



- **创建时间**: 2025-07-11 (根据`uwue_batch.py`内的注释)
- **作者**: LCM (协助: Gemini & Claude AI)
- **版本信息**: 未明确标明版本号，但文件注释清晰，结构良好。
- **已知问题**:
  - 站点处理为串行，处理大量站点时效率有待提升。
  - 对输入数据格式的依赖性强，如果CSV列名或格式与`BerkeleyConversion.json`不符，程序会失败。
- **TODO项**:
  - [ ] 实现基于`multiprocessing`的站点级并行处理。
  - [ ] 增加通过命令行参数配置`BASE_PATH`和`OUTPUT_PATH`的功能。
  - [ ] 对`zhou.quantreg`的收敛性进行更详细的检查和报告。
  - [ ] 考虑在数据加载时增加更灵活的列名映射机制，以适应不同数据源。