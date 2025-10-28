# ET Partition Reference Implementation

This repository consolidates three widely used evapotranspiration (ET) partitioning
approaches into a single, well-structured Python project. Each method can be
run independently while sharing a common repository layout, sample data, and
packaging metadata.

## Included methods

| Method | Directory | Original reference |
| ------ | --------- | ------------------ |
| Perez-Priego optimality-based partitioning | `methods/perez_priego` | Perez-Priego et al. (2018) |
| Transpiration Estimation Algorithm (TEA) | `methods/tea` | Nelson et al. (2018) |
| Underlying Water Use Efficiency (uWUE) | `methods/uwue` | Zhou et al. (2016) |

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

## Licensing

The repository contains code released under a combination of open-source
licenses inherited from the upstream projects. Refer to the original LICENSE
files preserved inside each method directory and the `third_party/` folder for
full details before redistributing the software.
