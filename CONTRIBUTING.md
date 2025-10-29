# Contributing to ET-Partition / 贡献指南

Thank you for your interest in contributing to the ET-Partition project! This document
provides guidelines for contributing to the codebase.

感谢您对ET-Partition项目的关注！本文档提供了代码库贡献指南。

[English](#english) | [中文](#中文)

---

## English

### How to Contribute

We welcome contributions in the following areas:

1. **Bug reports and fixes**
2. **Documentation improvements**
3. **New features or methods**
4. **Performance optimizations**
5. **Test coverage**
6. **Examples and tutorials**

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ET-partition.git
   cd ET-partition
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e .
   pip install pytest black flake8  # Optional dev tools
   ```

### Development Workflow

1. **Make your changes** in your feature branch
2. **Test your changes**:
   ```bash
   python tests/test_all_methods.py
   ```
3. **Commit your changes** with clear messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** on GitHub

### Code Style Guidelines

- **PEP 8**: Follow Python PEP 8 style guide
- **Docstrings**: Use clear docstrings for all functions and classes
- **Comments**: Add bilingual comments (English/Chinese) where helpful
- **Type hints**: Use type hints for function signatures (Python 3.10+)
- **Variable names**: Use descriptive names

**Example:**
```python
def calculate_transpiration(et: float, gpp: float, vpd: float) -> float:
    """
    Calculate transpiration from ET and environmental variables.

    根据ET和环境变量计算蒸腾量。

    Args:
        et: Evapotranspiration (mm/timestep)
        gpp: Gross primary productivity (μmol CO₂ m⁻² s⁻¹)
        vpd: Vapor pressure deficit (hPa)

    Returns:
        Transpiration (mm/timestep)
    """
    # Implementation here
    pass
```

### Testing Guidelines

- **Write tests** for new features
- **Run existing tests** before submitting PR
- **Ensure tests pass** on your local machine
- **Test with example data** in `data/test_site/`

### Documentation Guidelines

- **Update README** if adding new features
- **Add docstrings** to all public functions
- **Include examples** for new functionality
- **Bilingual preferred**: English and Chinese

### Pull Request Process

1. **Ensure tests pass** and code follows style guidelines
2. **Update documentation** as needed
3. **Describe your changes** clearly in the PR description
4. **Reference issues** if applicable (e.g., "Fixes #123")
5. **Wait for review** from maintainers
6. **Address feedback** promptly

### Reporting Bugs

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **System information** (OS, Python version)
- **Error messages** or stack traces
- **Sample data** if applicable (but no sensitive data)

**Bug report template:**
```markdown
**Description:**
Brief description of the bug

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What you expected to happen

**Actual Behavior:**
What actually happened

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.10.5]
- ET-partition version: [e.g., 0.1.0]

**Error Message:**
```
Paste error message here
```
```

### Feature Requests

For feature requests, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** and its benefits
3. **Provide use cases** or examples
4. **Discuss implementation** if you have ideas

### Code Review Process

- Maintainers will review PRs within 1-2 weeks
- Feedback will be provided via GitHub comments
- Changes may be requested before merging
- Once approved, maintainers will merge the PR

### Community Guidelines

- **Be respectful** and constructive
- **Help others** when possible
- **Give credit** to original authors
- **Follow open source** best practices

---

## 中文

### 如何贡献

我们欢迎以下方面的贡献：

1. **错误报告和修复**
2. **文档改进**
3. **新功能或方法**
4. **性能优化**
5. **测试覆盖率**
6. **示例和教程**

### 入门指南

1. 在GitHub上**Fork仓库**
2. 在本地**克隆您的fork**：
   ```bash
   git clone https://github.com/your-username/ET-partition.git
   cd ET-partition
   ```
3. 为您的功能**创建新分支**：
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **安装开发依赖**：
   ```bash
   pip install -e .
   pip install pytest black flake8  # 可选的开发工具
   ```

### 开发工作流程

1. 在功能分支中**进行更改**
2. **测试您的更改**：
   ```bash
   python tests/test_all_methods.py
   ```
3. **提交更改**并附上清晰的消息：
   ```bash
   git add .
   git commit -m "添加功能：简要描述"
   ```
4. **推送到您的fork**：
   ```bash
   git push origin feature/your-feature-name
   ```
5. 在GitHub上**打开Pull Request**

### 代码风格指南

- **PEP 8**：遵循Python PEP 8风格指南
- **文档字符串**：为所有函数和类使用清晰的文档字符串
- **注释**：在有帮助的地方添加双语注释（英语/中文）
- **类型提示**：为函数签名使用类型提示（Python 3.10+）
- **变量名**：使用描述性名称

### 测试指南

- 为新功能**编写测试**
- 在提交PR之前**运行现有测试**
- **确保测试通过**在本地机器上
- 使用`data/test_site/`中的**示例数据测试**

### 文档指南

- 如果添加新功能，**更新README**
- 为所有公共函数**添加文档字符串**
- 为新功能**包含示例**
- **优先双语**：英语和中文

### Pull Request流程

1. **确保测试通过**且代码遵循风格指南
2. 根据需要**更新文档**
3. 在PR描述中**清楚描述您的更改**
4. 如果适用，**引用问题**（例如，"Fixes #123"）
5. **等待维护者审核**
6. **及时处理反馈**

### 报告错误

报告错误时，请包括：

- 问题的**清晰描述**
- **重现问题的步骤**
- **预期与实际行为**
- **系统信息**（操作系统、Python版本）
- **错误消息**或堆栈跟踪
- 如果适用，**示例数据**（但不要包含敏感数据）

### 功能请求

对于功能请求，请：

1. **检查现有问题**以避免重复
2. **描述功能**及其好处
3. **提供用例**或示例
4. 如果有想法，**讨论实现**

### 代码审查流程

- 维护者将在1-2周内审查PR
- 反馈将通过GitHub评论提供
- 在合并之前可能会要求更改
- 一旦批准，维护者将合并PR

### 社区指南

- **尊重**和建设性
- 尽可能**帮助他人**
- 给原作者**署名**
- 遵循**开源**最佳实践

---

## License / 许可证

By contributing to ET-Partition, you agree that your contributions will be licensed
under the same license as the project (Mixed licenses from upstream projects).

通过为ET-Partition贡献，您同意您的贡献将根据与项目相同的许可证（来自上游项目的混合许可证）授权。

---

## Questions? / 有疑问？

If you have questions about contributing:

- Open an issue on GitHub
- Check existing documentation
- Contact the maintainers

如果您对贡献有疑问：

- 在GitHub上打开issue
- 查看现有文档
- 联系维护者

---

**Thank you for contributing! / 感谢您的贡献！**
