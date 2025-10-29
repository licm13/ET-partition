# Examples / 使用示例

This directory contains example scripts demonstrating how to use the ET-partition methods.

本目录包含演示如何使用ET拆分方法的示例脚本。

## Available Examples / 可用示例

### basic_usage.py

Comprehensive demonstration of all three methods with result comparison.

演示所有三种方法并比较结果的综合脚本。

**Features / 功能:**
- Individual method demonstrations (uWUE, TEA, Perez-Priego)
- Result visualization
- Method comparison plots
- Statistics calculation

**Usage / 使用:**
```bash
python examples/basic_usage.py
```

**Prerequisites / 前提条件:**
- Install project: `pip install -e .`
- Test data available in `data/test_site/`

## Adding Your Own Examples / 添加自己的示例

To contribute examples:

1. Create a new Python script in this directory
2. Add clear docstrings in both English and Chinese
3. Include proper error handling
4. Document the example in this README

贡献示例的步骤：

1. 在此目录中创建新的Python脚本
2. 添加中英文清晰的文档字符串
3. 包含适当的错误处理
4. 在本README中记录示例

## Example Output Structure / 示例输出结构

```
outputs/
├── example_uwue/
│   ├── *_uWUE_output.csv
│   ├── *_uWUE_output.nc
│   └── plots/
├── example_tea/
│   └── *_TEA_results.csv
├── example_pp/
│   ├── *_pp_output.csv
│   └── *_plot.png
└── example_comparison/
    ├── method_comparison.png
    └── method_comparison.csv
```
