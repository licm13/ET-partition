# ET Partition Reference Implementation

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Mixed-green.svg)](#licensing)

[English](README.md) | [中文](README_CN.md)

This repository consolidates three widely used evapotranspiration (ET) partitioning
approaches into a single, well-structured Python project. Each method can be
run independently while sharing a common repository layout, sample data, and
packaging metadata.

**ET partitioning** is the process of separating total evapotranspiration into its
two main components:
- **Transpiration (T)**: Water evaporated through plant stomata
- **Evaporation (E)**: Direct evaporation from soil and water surfaces

This distinction is critical for understanding ecosystem water use, carbon-water
coupling, and responses to environmental changes.

## Included methods

| Method | Directory | Original reference | Time Resolution | Key Features |
| ------ | --------- | ------------------ | --------------- | ------------ |
| **uWUE** | `methods/uwue` | [Zhou et al. (2016)](#citations) | Daily | Water use efficiency based, quantile regression |
| **TEA** | `methods/tea` | [Nelson et al. (2018)](#citations) | Half-hourly | Machine learning, quantile random forest |
| **Perez-Priego** | `methods/perez_priego` | [Perez-Priego et al. (2018)](#citations) | Half-hourly | Optimality theory, stomatal conductance |

Each method directory contains a batch processing entry point together with the
supporting modules required to reproduce the published workflow.

## Project layout

```
├── data/                 # Example Fluxnet-style input data
├── methods/              # Python implementations of each partition method
│   ├── perez_priego/
│   ├── tea/
│   └── uwue/
├── notebooks/            # Jupyter tutorials for interactive exploration
├── outputs/              # Created at runtime for method results (ignored)
└── third_party/          # Archived source material and reference packages
```

## Installation

The repository ships with a `pyproject.toml` defining the Python dependencies.
Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\\Scripts\\activate` on Windows
pip install -e .
```

The installation exposes the `methods` package, allowing the batch runners to
be executed with `python -m`.

## Sample data

The `data/test_site` directory contains a single Fluxnet2015 station folder
(`FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3`) that can be used for smoke
tests. The datasets are identical to those shipped with the original releases
and remain in CSV format to simplify experimentation.

> **Note**  The batch runners expect each site to live in a folder matching the
> Fluxnet naming convention, e.g. `FLX_<SITE>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#`.

## Running the batch workflows

Each method exposes a command-line interface with sensible defaults. All
scripts accept a `--base-path` pointing to a directory of site folders and an
`--output-path` where results will be written.

### Perez-Priego

```bash
python -m methods.perez_priego.batch \
    --base-path data/test_site \
    --output-path outputs/perez_priego
```

Optional arguments:

* `--site-metadata` – Excel sheet containing `SITE_ID` and `LOCATION_ELEV` columns.
* `--default-altitude` – fallback altitude in kilometres when metadata is missing.

Outputs consist of a daily transpiration/evaporation time series and diagnostic
plots for each processed site.

### TEA

```bash
python -m methods.tea.batch \
    --base-path data/test_site \
    --output-path outputs/tea
```

Optional arguments:

* `--pattern` – custom regular expression used to match Fluxnet/AmeriFlux folders.

The TEA workflow produces half-hourly transpiration (`TEA_T`), evaporation
(`TEA_E`), and water use efficiency (`TEA_WUE`) estimates for each site.

### uWUE

```bash
python -m methods.uwue.batch \
    --base-path data/test_site \
    --output-path outputs/uwue
```

Optional arguments:

* `--no-plots` – skip creation of diagnostic figures.
* `--pattern` – override the default Fluxnet2015 folder expression.

Results are written to both CSV and NetCDF files, and detailed log files are
stored alongside the outputs.

## Tutorials

Three Jupyter notebooks in the `notebooks/` directory mirror the original
method documentation and provide step-by-step demonstrations. Launch them with
JupyterLab after installing the dependencies.

```bash
pip install jupyterlab
jupyter lab
```

## Third-party material

The `third_party/` directory contains unmodified resources from the original
projects (e.g. R packages, archived releases, and supporting JSON files). They
are preserved for traceability but are not imported by default.

## Testing

A comprehensive test script is provided to validate all methods using the sample data:

```bash
python tests/test_all_methods.py
```

This will run all three methods on the FI-Hyy test site and verify the outputs.

## Method descriptions

A bilingual deep-dive into the mathematics of each algorithm and the new
synthetic benchmarking workflow is available in
[`docs/partition_methods_math.md`](docs/partition_methods_math.md).

### uWUE (Underlying Water Use Efficiency)

**Principle**: Estimates potential water use efficiency under optimal conditions (high
soil moisture, post-precipitation) using quantile regression. Partitioning is based
on the ratio of actual to potential uWUE.

**Key equation**: `T/ET = uWUEa / uWUEp`, where `uWUE = GPP × √VPD / T`

**Outputs**: Daily transpiration and evaporation, diagnostic plots, NetCDF with metadata

### TEA (Transpiration Estimation Algorithm)

**Principle**: Uses quantile random forest to model water use efficiency under ideal
conditions (high soil moisture, growing season). Predicts WUE for all conditions
and calculates `T = GPP / WUE`.

**Key features**:
- Non-parametric machine learning approach
- Half-hourly time resolution
- Multiple derived indices (CSWI, DWCI, diurnal centroid)

**Outputs**: Half-hourly T, E, and WUE time series

### Perez-Priego (Optimality-based)

**Principle**: Based on stomatal conductance optimization theory. Uses 5-day moving
windows to fit optimal parameters and estimate transpiration from stomatal models.

**Key features**:
- Biophysical optimization framework
- Requires site elevation metadata
- Moving window parameter estimation

**Outputs**: Half-hourly transpiration and evaporation estimates with diagnostic plots

## Data requirements

All methods expect **FLUXNET2015 or AmeriFlux** formatted CSV files with half-hourly
observations. Required variables include:

- **Energy fluxes**: Latent heat (LE), sensible heat (H), ground heat (G), net radiation
- **Meteorology**: Air temperature, vapor pressure deficit, relative humidity, precipitation
- **Carbon flux**: Gross primary productivity (GPP)
- **Quality flags**: QC indicators for flux measurements

See `data/test_site/` for an example dataset (FI-Hyy site, 2008-2010).

## Citations

If you use these methods, please cite the original papers:

**uWUE:**
> Zhou, S., Yu, B., Zhang, Y., Huang, Y., & Wang, G. (2016). Partitioning evapotranspiration
> based on the concept of underlying water use efficiency. *Water Resources Research*,
> 52(2), 1160-1175. https://doi.org/10.1002/2015WR017766

**TEA:**
> Nelson, J. A., Carvalhais, N., Migliavacca, M., Reichstein, M., & Jung, M. (2018).
> Water-stress-induced breakdown of carbon–water relations: indicators from diurnal
> FLUXNET patterns. *Biogeosciences*, 15(8), 2433-2447.
> https://doi.org/10.5194/bg-15-2433-2018

**Perez-Priego:**
> Perez-Priego, O., et al. (2018). Partitioning eddy covariance water flux components
> using physiological and micrometeorological approaches. *Journal of Geophysical
> Research: Biogeosciences*, 123(10), 3353-3370. https://doi.org/10.1029/2018JG004637

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Troubleshooting

**Q: Missing columns error**
A: Ensure your data follows FLUXNET2015 naming conventions. Check column mappings in
`methods/uwue/BerkeleyConversion.json` or method-specific documentation.

**Q: Out of memory errors**
A: Process sites individually using `--base-path data/site_folder` instead of batch processing.

**Q: TEA predictions are all NaN**
A: Check that you have sufficient high-quality data during ideal conditions (growing season,
after precipitation events). TEA requires training data under optimal conditions.

## Contact

For issues and questions, please use the GitHub issue tracker.

## Acknowledgments

This consolidation was created by Changming Li with assistance from AI tools (Gemini & Claude).
Original method implementations are credited to their respective authors (see [Citations](#citations)).

## Licensing

The repository contains code released under a combination of open-source
licenses inherited from the upstream projects. Refer to the original LICENSE
files preserved inside each method directory and the `third_party/` folder for
full details before redistributing the software.
