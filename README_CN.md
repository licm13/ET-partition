# ET蒸散发拆分参考实现

[![Python版本](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/license-Mixed-green.svg)](#许可证)

[English](README.md) | [中文](README_CN.md)

本仓库整合了三种广泛使用的蒸散发(ET)拆分方法，构建为一个结构良好的Python项目。
每个方法都可以独立运行，同时共享通用的仓库布局、样本数据和包元数据。

**ET拆分**是将总蒸散发分离为两个主要组分的过程：
- **蒸腾(Transpiration, T)**：通过植物气孔蒸发的水分
- **蒸发(Evaporation, E)**：土壤和水面的直接蒸发

这种区分对于理解生态系统水分利用、碳水耦合以及对环境变化的响应至关重要。

## 包含的方法

| 方法 | 目录 | 原始文献 | 时间分辨率 | 关键特性 |
| ---- | ---- | -------- | ---------- | -------- |
| **uWUE** | `methods/uwue` | [Zhou et al. (2016)](#引用文献) | 日尺度 | 基于水分利用效率，分位数回归 |
| **TEA** | `methods/tea` | [Nelson et al. (2018)](#引用文献) | 半小时 | 机器学习，分位数随机森林 |
| **Perez-Priego** | `methods/perez_priego` | [Perez-Priego et al. (2018)](#引用文献) | 半小时 | 最优化理论，气孔导度模型 |

每个方法目录都包含批处理入口脚本以及重现已发表工作流程所需的支持模块。

## 项目结构

```
ET-partition/
├── data/                 # 示例FLUXNET格式输入数据
│   ├── test_site/       # 测试站点数据（FI-Hyy，2008-2010）
│   └── tea_reference/   # TEA参考NetCDF数据
├── methods/              # 每个拆分方法的Python实现
│   ├── perez_priego/    # Perez-Priego最优化方法
│   ├── tea/             # 蒸腾估算算法(TEA)
│   └── uwue/            # 潜在水分利用效率(uWUE)
├── notebooks/            # Jupyter交互式教程
│   ├── Zhou_tutorial.ipynb
│   ├── TEA_tutorial.ipynb
│   └── Perez-Priego_tutorial.ipynb
├── tests/                # 测试脚本
│   └── test_all_methods.py
├── outputs/              # 运行时创建的结果目录（被忽略）
│   ├── uwue/
│   ├── tea/
│   └── perez_priego/
└── third_party/          # 归档的源材料和参考包
```

## 安装

本仓库包含`pyproject.toml`文件，定义了Python依赖。创建虚拟环境并以可编辑模式安装项目：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate     # Windows

# 安装项目依赖
pip install -e .
```

安装后会暴露`methods`包，允许使用`python -m`执行批处理脚本。

### 依赖项

- **numpy** >= 1.23 - 数值计算
- **pandas** >= 1.5 - 数据处理
- **matplotlib** >= 3.7 - 数据可视化
- **seaborn** >= 0.12 - 统计图表
- **xarray** >= 2023.1 - 科学数据处理
- **scikit-learn** >= 1.3 - 机器学习（TEA）
- **numba** >= 0.57 - JIT编译加速
- **netCDF4** >= 1.6 - NetCDF文件I/O
- **openpyxl** >= 3.1 - Excel读写

## 示例数据

`data/test_site`目录包含一个FLUXNET2015站点文件夹
(`FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3`)，可用于测试运行。

**测试站点信息：**
- **站点代码**：FI-Hyy（芬兰海门林卡，欧洲云杉林）
- **时间范围**：2008-01-01 至 2010-12-31（3年）
- **数据记录**：52,609条半小时观测数据
- **数据格式**：FLUXNET2015 FULLSET标准CSV格式

> **注意**：批处理脚本期望每个站点位于符合FLUXNET命名规范的文件夹中，
> 例如 `FLX_<站点>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#`。

## 运行批处理工作流

每个方法都提供了带有合理默认值的命令行接口。所有脚本都接受指向站点文件夹目录的
`--base-path`参数和写入结果的`--output-path`参数。

### Perez-Priego方法

```bash
python -m methods.perez_priego.batch \
    --base-path data/test_site \
    --output-path outputs/perez_priego
```

**可选参数：**

* `--site-metadata` – 包含`SITE_ID`和`LOCATION_ELEV`列的Excel表格
* `--default-altitude` – 元数据缺失时的默认海拔高度（单位：千米）

**输出内容**：每个处理站点的日尺度蒸腾/蒸发时间序列和诊断图表。

### TEA方法

```bash
python -m methods.tea.batch \
    --base-path data/test_site \
    --output-path outputs/tea
```

**可选参数：**

* `--pattern` – 用于匹配FLUXNET/AmeriFlux文件夹的自定义正则表达式

**输出内容**：每个站点的半小时蒸腾(`TEA_T`)、蒸发(`TEA_E`)和
水分利用效率(`TEA_WUE`)估算值。

### uWUE方法

```bash
python -m methods.uwue.batch \
    --base-path data/test_site \
    --output-path outputs/uwue
```

**可选参数：**

* `--no-plots` – 跳过诊断图表的创建
* `--pattern` – 覆盖默认的FLUXNET2015文件夹表达式

**输出内容**：结果同时写入CSV和NetCDF文件，详细的日志文件与输出存储在一起。

## 教程

`notebooks/`目录中的三个Jupyter笔记本镜像了原始方法文档，提供逐步演示。
安装依赖后使用JupyterLab启动：

```bash
pip install jupyterlab
jupyter lab
```

**可用教程：**
- `Zhou_tutorial.ipynb` - uWUE方法详细教程
- `TEA_tutorial.ipynb` - TEA分位数随机森林教程
- `Perez-Priego_tutorial.ipynb` - 最优化理论和参数拟合教程

## 测试

提供了综合测试脚本，使用示例数据验证所有方法：

```bash
python tests/test_all_methods.py
```

这将在FI-Hyy测试站点上运行所有三种方法并验证输出。

## 方法详细说明

### uWUE（潜在水分利用效率）

**原理**：使用分位数回归估算最优条件下（高土壤湿度、降雨后）的潜在水分利用效率。
拆分基于实际与潜在uWUE的比率。

**关键公式**：`T/ET = uWUEa / uWUEp`，其中 `uWUE = GPP × √VPD / T`

**计算流程**：
1. 计算每日ET和PET（潜在蒸散发）
2. 生成uWUE掩码（基于降雨、土壤湿度等条件）
3. 使用分位数回归（95%分位）估算年度uWUEp
4. 使用线性回归估算日尺度uWUEa
5. 计算 T = ET × (uWUEa / uWUEp)

**输出**：
- 日尺度蒸腾和蒸发
- 四子图诊断图表（uWUE散点图、时间序列等）
- 带元数据的NetCDF文件

### TEA（蒸腾估算算法）

**原理**：使用分位数随机森林建模理想条件下（高土壤湿度、生长季）的水分利用效率。
为所有条件预测WUE，然后计算 `T = GPP / WUE`。

**关键特性**：
- 非参数化机器学习方法
- 半小时时间分辨率
- 多种衍生指数（CSWI、DWCI、日质心）

**计算流程**：
1. 数据预处理和特征工程
2. 计算保守地表水指数(CSWI)和日变化水碳耦合指数(DWCI)
3. 筛选理想条件数据
4. 训练分位数随机森林模型（75%分位）
5. 全数据集WUE预测
6. 计算 T = GPP / WUE，E = ET - T

**输出**：
- 半小时蒸腾(T)、蒸发(E)和水分利用效率(WUE)时间序列

### Perez-Priego（基于最优化理论）

**原理**：基于气孔导度最优化理论。使用5天滑动窗口拟合最优参数，
从气孔模型估算蒸腾。

**关键特性**：
- 生物物理最优化框架
- 需要站点海拔元数据
- 滑动窗口参数估计

**计算流程**：
1. 计算长期参数（Chi_o、WUE_o）
2. 按5天滑动窗口分割数据
3. 筛选白天数据
4. 参数优化
5. 气孔导度和蒸腾计算
6. 负值截断处理

**输出**：
- 半小时蒸腾和蒸发估算值
- 日平均通量曲线诊断图

## 数据要求

所有方法都期望**FLUXNET2015或AmeriFlux**格式的CSV文件，包含半小时观测数据。
必需变量包括：

**能量通量：**
- `LE_F_MDS` - 潜热通量（W/m²）
- `H_F_MDS` - 感热通量（W/m²）
- `G_F_MDS` - 土壤热通量（W/m²）
- `NETRAD` - 净辐射（W/m²）

**气象变量：**
- `TA_F_MDS` - 空气温度（°C）
- `VPD_F_MDS` - 水汽压差（hPa）
- `RH` - 相对湿度（%）
- `P` 或 `P_ERA` - 降水（mm）
- `WS` - 风速（m/s）

**碳通量：**
- `GPP_NT_VUT_REF` 或 `GPP_NT_VUT_USTAR50` - 总初级生产力（μmol CO₂/m²/s）

**质量控制：**
- `*_QC` - 各变量的质量控制标志

参见`data/test_site/`获取示例数据集（FI-Hyy站点，2008-2010）。

## 引用文献

如果使用这些方法，请引用原始论文：

**uWUE方法：**
> Zhou, S., Yu, B., Zhang, Y., Huang, Y., & Wang, G. (2016). Partitioning evapotranspiration
> based on the concept of underlying water use efficiency. *Water Resources Research*,
> 52(2), 1160-1175. https://doi.org/10.1002/2015WR017766

**TEA方法：**
> Nelson, J. A., Carvalhais, N., Migliavacca, M., Reichstein, M., & Jung, M. (2018).
> Water-stress-induced breakdown of carbon–water relations: indicators from diurnal
> FLUXNET patterns. *Biogeosciences*, 15(8), 2433-2447.
> https://doi.org/10.5194/bg-15-2433-2018

**Perez-Priego方法：**
> Perez-Priego, O., et al. (2018). Partitioning eddy covariance water flux components
> using physiological and micrometeorological approaches. *Journal of Geophysical
> Research: Biogeosciences*, 123(10), 3353-3370. https://doi.org/10.1029/2018JG004637

## 第三方材料

`third_party/`目录包含来自原始项目的未修改资源（如R包、归档发布和支持JSON文件）。
它们被保留用于可追溯性，但默认不会被导入。

## 贡献

欢迎贡献！请参阅[CONTRIBUTING.md](CONTRIBUTING.md)了解指南。

## 常见问题

**问：缺少列错误**
答：确保您的数据遵循FLUXNET2015命名规范。检查`methods/uwue/BerkeleyConversion.json`
中的列映射或方法特定文档。

**问：内存不足错误**
答：使用`--base-path data/site_folder`单独处理站点，而不是批量处理。

**问：TEA预测全部为NaN**
答：检查您在理想条件下（生长季、降雨事件后）是否有足够的高质量数据。
TEA需要最优条件下的训练数据。

**问：如何处理自己的数据？**
答：确保数据格式符合FLUXNET2015标准，包含所有必需列。将数据放在符合命名规范的文件夹中：
`FLX_<站点代码>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#/`

**问：各方法的优缺点是什么？**
答：
- **uWUE**：计算简单，但仅提供日尺度结果
- **TEA**：高时间分辨率，需要充足的理想条件数据
- **Perez-Priego**：基于生物物理机制，需要站点海拔信息

## 联系方式

如有问题和议题，请使用GitHub issue tracker。

## 致谢

本整合项目由李昌明创建，并得到AI工具（Gemini & Claude）的协助。
原始方法实现归功于各自的作者（参见[引用文献](#引用文献)）。

## 许可证

本仓库包含从上游项目继承的多种开源许可证下发布的代码。
在重新分发软件之前，请参阅每个方法目录和`third_party/`文件夹中保留的原始LICENSE文件以获取完整详情。
