# Quick Start Guide / 快速开始指南

[English](#english) | [中文](#中文)

---

## English

### Installation (5 minutes)

**Step 1: Clone the repository**
```bash
git clone https://github.com/your-username/ET-partition.git
cd ET-partition
```

**Step 2: Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Step 3: Install dependencies**
```bash
pip install -e .
```

That's it! You're ready to go.

### Run Your First Test (2 minutes)

Test all methods with the included sample data:

```bash
python tests/test_all_methods.py
```

This will process the FI-Hyy test site (2008-2010) and create outputs in `outputs/test_run/`.

### Run Individual Methods

**Option 1: uWUE (Daily partitioning)**
```bash
python -m methods.uwue.batch \
    --base-path data/test_site \
    --output-path outputs/uwue
```

Expected outputs:
- `outputs/uwue/FI-Hyy_uWUE_output.csv` - Daily T and E
- `outputs/uwue/plots/FI-Hyy_uWUE_analysis.png` - Diagnostic plots

**Option 2: TEA (Half-hourly partitioning)**
```bash
python -m methods.tea.batch \
    --base-path data/test_site \
    --output-path outputs/tea
```

Expected outputs:
- `outputs/tea/FI-Hyy_TEA_results.csv` - Half-hourly T, E, and WUE

**Option 3: Perez-Priego (Half-hourly partitioning)**
```bash
python -m methods.perez_priego.batch \
    --base-path data/test_site \
    --output-path outputs/perez_priego
```

Expected outputs:
- `outputs/perez_priego/FI-Hyy_pp_output.csv` - Half-hourly T and E
- `outputs/perez_priego/FI-Hyy_plot.png` - Diagnostic plot

### Explore Tutorials

Launch Jupyter to explore interactive tutorials:

```bash
pip install jupyterlab
jupyter lab
```

Then open:
- `notebooks/Zhou_tutorial.ipynb` - uWUE method walkthrough
- `notebooks/TEA_tutorial.ipynb` - TEA method walkthrough
- `notebooks/Perez-Priego_tutorial.ipynb` - Perez-Priego method walkthrough

### Use Your Own Data

**Requirements:**
1. FLUXNET2015 or AmeriFlux CSV format
2. Half-hourly timesteps
3. Required columns: LE, GPP, TA, VPD, RH, precipitation, etc.
4. Folder naming: `FLX_<SITE>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#/`

**Example:**
```bash
# Place your data in data/my_sites/
python -m methods.uwue.batch \
    --base-path data/my_sites \
    --output-path outputs/my_analysis
```

### Common Issues

**Issue: "ModuleNotFoundError"**
```bash
# Solution: Reinstall in editable mode
pip install -e .
```

**Issue: "Missing column XYZ"**
```bash
# Solution: Check your CSV has FLUXNET2015 standard column names
# See methods/uwue/BerkeleyConversion.json for column mappings
```

**Issue: "Out of memory"**
```bash
# Solution: Process one site at a time
python -m methods.tea.batch \
    --base-path data/single_site_folder \
    --output-path outputs/tea
```

### Next Steps

- Read the [full README](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Report issues on GitHub

---

## 中文

### 安装（5分钟）

**步骤1：克隆仓库**
```bash
git clone https://github.com/your-username/ET-partition.git
cd ET-partition
```

**步骤2：创建虚拟环境**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows系统: .venv\Scripts\activate
```

**步骤3：安装依赖**
```bash
pip install -e .
```

完成！您已准备就绪。

### 运行第一个测试（2分钟）

使用包含的示例数据测试所有方法：

```bash
python tests/test_all_methods.py
```

这将处理FI-Hyy测试站点（2008-2010）并在`outputs/test_run/`中创建输出。

### 运行单个方法

**选项1：uWUE（日尺度拆分）**
```bash
python -m methods.uwue.batch \
    --base-path data/test_site \
    --output-path outputs/uwue
```

预期输出：
- `outputs/uwue/FI-Hyy_uWUE_output.csv` - 日尺度蒸腾和蒸发
- `outputs/uwue/plots/FI-Hyy_uWUE_analysis.png` - 诊断图表

**选项2：TEA（半小时拆分）**
```bash
python -m methods.tea.batch \
    --base-path data/test_site \
    --output-path outputs/tea
```

预期输出：
- `outputs/tea/FI-Hyy_TEA_results.csv` - 半小时蒸腾、蒸发和水分利用效率

**选项3：Perez-Priego（半小时拆分）**
```bash
python -m methods.perez_priego.batch \
    --base-path data/test_site \
    --output-path outputs/perez_priego
```

预期输出：
- `outputs/perez_priego/FI-Hyy_pp_output.csv` - 半小时蒸腾和蒸发
- `outputs/perez_priego/FI-Hyy_plot.png` - 诊断图表

### 探索教程

启动Jupyter以探索交互式教程：

```bash
pip install jupyterlab
jupyter lab
```

然后打开：
- `notebooks/Zhou_tutorial.ipynb` - uWUE方法演练
- `notebooks/TEA_tutorial.ipynb` - TEA方法演练
- `notebooks/Perez-Priego_tutorial.ipynb` - Perez-Priego方法演练

### 使用您自己的数据

**要求：**
1. FLUXNET2015或AmeriFlux CSV格式
2. 半小时时间步长
3. 必需列：LE、GPP、TA、VPD、RH、降水等
4. 文件夹命名：`FLX_<站点>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#/`

**示例：**
```bash
# 将数据放在 data/my_sites/
python -m methods.uwue.batch \
    --base-path data/my_sites \
    --output-path outputs/my_analysis
```

### 常见问题

**问题："ModuleNotFoundError"**
```bash
# 解决方案：以可编辑模式重新安装
pip install -e .
```

**问题："缺少列XYZ"**
```bash
# 解决方案：检查CSV是否具有FLUXNET2015标准列名
# 参见 methods/uwue/BerkeleyConversion.json 了解列映射
```

**问题："内存不足"**
```bash
# 解决方案：一次处理一个站点
python -m methods.tea.batch \
    --base-path data/single_site_folder \
    --output-path outputs/tea
```

### 下一步

- 阅读[完整README](README_CN.md)了解详细文档
- 查看[CONTRIBUTING.md](CONTRIBUTING.md)以贡献代码
- 在GitHub上报告问题

---

## Performance Expectations / 性能预期

| Method | Test Site (3 years) | Typical Speed | Memory |
|--------|---------------------|---------------|--------|
| uWUE | ~30-60 seconds | ~1 year/5s | Low |
| TEA | ~2-5 minutes | ~1 year/30s | Medium |
| Perez-Priego | ~1-3 minutes | ~1 year/20s | Low |

*Tested on: Intel i7, 16GB RAM, SSD*

---

## Directory Structure After First Run / 首次运行后的目录结构

```
ET-partition/
├── outputs/
│   ├── uwue/
│   │   ├── FI-Hyy_uWUE_output.csv
│   │   ├── FI-Hyy_uWUE_output.nc
│   │   ├── plots/
│   │   │   └── FI-Hyy_uWUE_analysis.png
│   │   ├── processing_summary.txt
│   │   └── uwue_processing_*.log
│   ├── tea/
│   │   └── FI-Hyy_TEA_results.csv
│   └── perez_priego/
│       ├── FI-Hyy_pp_output.csv
│       └── FI-Hyy_plot.png
└── test_results.log
```

---

## Getting Help / 获取帮助

- **Documentation**: See [README.md](README.md) or [README_CN.md](README_CN.md)
- **Issues**: https://github.com/your-username/ET-partition/issues
- **Tutorials**: Check `notebooks/` directory
- **原始论文**: See [Citations](README.md#citations) section

---

**Enjoy partitioning ET! / 祝您使用愉快！**
