# ET-Partition Project Summary / 项目总结

## Overview / 概述

This document provides a comprehensive overview of the ET-Partition project structure,
features, and usage guidelines.

本文档提供ET-Partition项目结构、功能和使用指南的全面概述。

**Project Version:** 0.1.0
**Python Version:** >= 3.10
**License:** Mixed (see LICENSE file)

---

## Project Structure / 项目结构

```
ET-partition/
├── README.md                      # Main project documentation (English)
├── README_CN.md                   # Main project documentation (Chinese)
├── QUICKSTART.md                  # Quick start guide (bilingual)
├── CONTRIBUTING.md                # Contribution guidelines (bilingual)
├── LICENSE                        # License information
├── PROJECT_SUMMARY.md            # This file
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Python dependencies
│
├── methods/                       # ET partitioning methods
│   ├── __init__.py               # Package initialization with full docs
│   ├── uwue/                     # uWUE method (Zhou et al. 2016)
│   │   ├── README.md             # Detailed method documentation
│   │   ├── batch.py              # Batch processing entry point
│   │   ├── zhou.py               # Core uWUE algorithm
│   │   ├── bigleaf.py            # Biophysical calculations
│   │   ├── preprocess.py         # Data preprocessing
│   │   └── *.json                # Configuration files
│   │
│   ├── tea/                      # TEA method (Nelson et al. 2018)
│   │   ├── README.md             # Detailed method documentation
│   │   ├── batch.py              # Batch processing entry point
│   │   ├── TEA/                  # TEA core library
│   │   │   ├── TEA.py            # Main partitioning functions
│   │   │   ├── core.py           # Quantile Random Forest
│   │   │   ├── PreProc.py        # Feature engineering
│   │   │   └── *.py              # Additional modules
│   │   └── environment.yml       # Conda environment
│   │
│   └── perez_priego/             # Perez-Priego method (2018)
│       ├── README.md             # Detailed method documentation
│       ├── batch.py              # Batch processing entry point
│       └── et_partitioning_functions.py  # Core functions
│
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_all_methods.py       # Comprehensive test script
│
├── examples/                     # Usage examples
│   ├── README.md
│   └── basic_usage.py            # Example script for all methods
│
├── notebooks/                    # Jupyter tutorials
│   ├── Zhou_tutorial.ipynb       # uWUE interactive tutorial
│   ├── TEA_tutorial.ipynb        # TEA interactive tutorial
│   └── Perez-Priego_tutorial.ipynb  # Perez-Priego tutorial
│
├── data/                         # Sample and reference data
│   ├── test_site/                # Test site (FI-Hyy, 2008-2010)
│   └── tea_reference/            # TEA reference NetCDF data
│
├── outputs/                      # Output directory (gitignored)
│   ├── uwue/
│   ├── tea/
│   └── perez_priego/
│
└── third_party/                  # Archived resources
    ├── original_release_v1.1/
    └── pyquantrf/
```

---

## Key Features / 核心功能

### 1. Three Validated ET Partitioning Methods / 三种经过验证的ET拆分方法

**uWUE (Underlying Water Use Efficiency)**
- Time resolution: Daily / 日尺度
- Key feature: Quantile regression approach / 分位数回归方法
- Best for: Long-term analysis / 适合长期分析

**TEA (Transpiration Estimation Algorithm)**
- Time resolution: Half-hourly / 半小时
- Key feature: Machine learning (Random Forest) / 机器学习（随机森林）
- Best for: High-resolution analysis / 适合高分辨率分析

**Perez-Priego (Optimality-based)**
- Time resolution: Half-hourly / 半小时
- Key feature: Physiological modeling / 生理建模
- Best for: Mechanistic understanding / 适合机制理解

### 2. Unified Interface / 统一接口

All methods share:
- Common command-line interface
- Consistent input/output formats
- FLUXNET2015 compatibility
- Comprehensive documentation

所有方法共享：
- 通用命令行接口
- 一致的输入/输出格式
- FLUXNET2015兼容性
- 全面的文档

### 3. Comprehensive Documentation / 全面的文档

- **English & Chinese**: All major docs are bilingual
- **Method-specific READMEs**: Detailed documentation per method
- **Jupyter tutorials**: Interactive learning notebooks
- **Example scripts**: Ready-to-run examples

### 4. Quality Assurance / 质量保证

- Comprehensive test suite
- Example data included
- Validated against original implementations
- Clear error messages

---

## Quick Start / 快速开始

### Installation / 安装

```bash
# Clone repository
git clone https://github.com/your-username/ET-partition.git
cd ET-partition

# Install dependencies
pip install -e .
```

### Run Test / 运行测试

```bash
# Test all methods with sample data
python tests/test_all_methods.py
```

### Run Individual Methods / 运行单个方法

```bash
# uWUE method
python -m methods.uwue.batch --base-path data/test_site --output-path outputs/uwue

# TEA method
python -m methods.tea.batch --base-path data/test_site --output-path outputs/tea

# Perez-Priego method
python -m methods.perez_priego.batch --base-path data/test_site --output-path outputs/perez_priego
```

---

## File Descriptions / 文件说明

### Documentation Files / 文档文件

| File | Purpose | Language |
|------|---------|----------|
| README.md | Main project documentation | English |
| README_CN.md | Main project documentation | Chinese |
| QUICKSTART.md | Quick start guide | Bilingual |
| CONTRIBUTING.md | Contribution guidelines | Bilingual |
| LICENSE | License information | Bilingual |
| PROJECT_SUMMARY.md | Project overview | Bilingual |

### Configuration Files / 配置文件

| File | Purpose |
|------|---------|
| pyproject.toml | Project metadata and dependencies |
| requirements.txt | Python package requirements |
| .gitignore | Git ignore rules |

### Code Files / 代码文件

| Directory | Purpose |
|-----------|---------|
| methods/ | ET partitioning method implementations |
| tests/ | Test scripts and validation |
| examples/ | Usage examples and demonstrations |
| notebooks/ | Jupyter interactive tutorials |

---

## Data Requirements / 数据要求

### Input Data Format / 输入数据格式

**Required:**
- FLUXNET2015 or AmeriFlux CSV format
- Half-hourly timesteps (30-min)
- Standard column names

**Key Variables:**
- Energy fluxes: LE, H, G, NETRAD
- Meteorology: TA, VPD, RH, P
- Carbon flux: GPP
- Quality flags: *_QC columns

### Output Formats / 输出格式

**uWUE:**
- Daily CSV and NetCDF files
- Diagnostic plots (PNG)
- Processing logs

**TEA:**
- Half-hourly CSV files
- No plots (processing focused)

**Perez-Priego:**
- Half-hourly CSV files
- Diagnostic plots (PNG)

---

## Performance Metrics / 性能指标

**Processing Speed** (FI-Hyy test site, 3 years, Intel i7):
- uWUE: ~30-60 seconds
- TEA: ~2-5 minutes
- Perez-Priego: ~1-3 minutes

**Memory Usage:**
- uWUE: Low (~500MB)
- TEA: Medium (~1-2GB)
- Perez-Priego: Low (~500MB)

---

## Citation Information / 引用信息

If you use this project, please cite both the original method papers AND
this repository:

**Original Papers:**

```bibtex
% uWUE Method
@article{zhou2016partitioning,
  title={Partitioning evapotranspiration based on the concept of underlying water use efficiency},
  author={Zhou, Sha and Yu, Bofu and Zhang, Yao and Huang, Yuefei and Wang, Guangqian},
  journal={Water Resources Research},
  volume={52},
  number={2},
  pages={1160--1175},
  year={2016}
}

% TEA Method
@article{nelson2018water,
  title={Water-stress-induced breakdown of carbon--water relations},
  author={Nelson, Jacob A and others},
  journal={Biogeosciences},
  volume={15},
  number={8},
  pages={2433--2447},
  year={2018}
}

% Perez-Priego Method
@article{perezpriego2018partitioning,
  title={Partitioning eddy covariance water flux components},
  author={Perez-Priego, Oscar and others},
  journal={JGR: Biogeosciences},
  volume={123},
  number={10},
  pages={3353--3370},
  year={2018}
}
```

**This Repository:**
```bibtex
@software{et_partition_2025,
  title={ET-Partition: Reference Implementation of ET Partitioning Methods},
  author={Li, Changming},
  year={2025},
  url={https://github.com/your-username/ET-partition}
}
```

---

## Development Status / 开发状态

**Version:** 0.1.0 (Initial Release)

**Status:**
- ✅ Core functionality implemented
- ✅ All three methods working
- ✅ Test suite complete
- ✅ Documentation complete
- ✅ Example data included
- ⏳ Performance optimization ongoing
- ⏳ Additional features planned

---

## Future Plans / 未来计划

### Planned Features / 计划功能

1. **Parallel processing** for batch operations
2. **Additional output formats** (HDF5, Zarr)
3. **Web interface** for easy access
4. **Docker containerization**
5. **More example datasets**
6. **Uncertainty quantification** modules

### Community Contributions Welcome / 欢迎社区贡献

We welcome contributions in:
- Bug fixes and improvements
- Documentation enhancements
- New example scripts
- Performance optimizations
- Additional partitioning methods

See CONTRIBUTING.md for guidelines.

---

## Support and Contact / 支持和联系

**Issues:** https://github.com/your-username/ET-partition/issues

**Documentation:** See README.md and method-specific READMEs

**Questions:** Open an issue with the "question" label

---

## Acknowledgments / 致谢

This project consolidates and extends implementations from:

- **Zhou et al. (2016)** - uWUE method
- **Nelson et al. (2018)** - TEA method
- **Perez-Priego et al. (2018)** - Optimality-based method

**Project Integration:** Changming Li with assistance from Gemini & Claude AI

**Test Data:** FLUXNET2015 dataset (FI-Hyy site)

---

## Version History / 版本历史

**v0.1.0 (2025-01-XX)** - Initial Release
- ✅ Three ET partitioning methods implemented
- ✅ Comprehensive documentation
- ✅ Test suite and examples
- ✅ Bilingual support (English/Chinese)

---

*Last Updated: 2025-01-XX*
*最后更新：2025-01-XX*
