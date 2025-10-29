# TEA Method - Transpiration Estimation Algorithm

[![DOI](https://zenodo.org/badge/121650199.svg)](https://zenodo.org/badge/latestdoi/121650199)

## Overview

The TEA (Transpiration Estimation Algorithm) uses quantile random forest machine learning
to partition evapotranspiration into transpiration (T) and evaporation (E) based on
water use efficiency under ideal conditions.

**Original Publication:**
> Nelson, J. A., Carvalhais, N., Migliavacca, M., Reichstein, M., & Jung, M. (2018).
> Water-stress-induced breakdown of carbon–water relations: indicators from diurnal
> FLUXNET patterns. *Biogeosciences*, 15(8), 2433-2447.
> https://doi.org/10.5194/bg-15-2433-2018

**Related Publication:**
> Nelson, Jacob A., Nuno Carvalhais, Matthias Cuntz, Nicolas Delpierre, Jürgen Knauer,
> Jérome Ogée, Mirco Migliavacca, Markus Reichstein, and Martin Jung. Coupling Water
> and Carbon Fluxes to Constrain Estimates of Transpiration: The TEA Algorithm.
> *Journal of Geophysical Research: Biogeosciences*, December 21, 2018.
> https://doi.org/10.1029/2018JG004727

**Original Repository:** https://github.com/jnelson18/ecosystem-transpiration

## Method Principle

TEA is based on the following concepts:

1. **Water Use Efficiency (WUE)**: Under ideal conditions (high soil moisture, growing season),
   the GPP/ET ratio represents potential WUE
2. **Machine Learning**: Quantile Random Forest (QRF) models WUE as a function of
   meteorological and derived variables
3. **Partitioning**: Once WUE is predicted, transpiration is calculated as `T = GPP / WUE`

**Key Workflow:**
```
1. Identify ideal conditions (high CSWI, growing season)
2. Train QRF on ideal condition data → WUE_potential
3. Predict WUE for all conditions
4. Calculate T = GPP / WUE
5. Calculate E = ET - T
```

## Key Features

- **High temporal resolution**: Half-hourly estimates
- **Non-parametric**: No assumptions about functional forms
- **Multiple indices**: CSWI, DWCI, diurnal centroid for enhanced prediction
- **Uncertainty quantification**: QRF provides prediction intervals

## Processing Workflow

```
Input: FLUXNET half-hourly CSV
        ↓
1. Data Loading & Preprocessing (PreProc.build_dataset)
   - Load required variables
   - Unit conversions (LE → ET)
   - Basic QC filtering
        ↓
2. Feature Engineering (PreProc.preprocess)
   - CSWI: Conservative Surface Water Index
   - DWCI: Diurnal Water-Carbon coupling Index
   - Diurnal Centroid: Time-weighted GPP/ET center
   - Potential radiation (PotRad.potential_radiation)
        ↓
3. Ideal Condition Selection
   Criteria:
   - High CSWI (moist conditions)
   - Growing season (high GPP)
   - Valid ET, GPP, meteorology
        ↓
4. Quantile Random Forest Training (core.QuantileRandomForestRegressor)
   - Target: WUE = GPP / ET
   - Features: Tair, VPD, RH, Rg, Rg_pot, CSWI, DWCI, centroid, u
   - Quantile: 75th percentile (conservative estimate)
   - Trees: 50-100 (default: 50)
        ↓
5. WUE Prediction for All Data
   - Apply trained QRF to full dataset
   - Returns: WUE_pred (g C / kg H₂O)
        ↓
6. T and E Calculation (TEA.simplePartition)
   - T = GPP / WUE_pred
   - E = ET - T
   - Handle edge cases (E < 0 → E = 0, T = ET)
        ↓
Output: Half-hourly T, E, WUE time series (CSV)
```

## Installation

TEA is part of the ET-partition project. Install dependencies:

```bash
pip install -e .
```

Or install specific dependencies:
```bash
pip install numpy pandas scikit-learn numba
```

**Optional for tutorials:**
```bash
conda env create -f environment.yml
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
python -m methods.tea.batch \
    --base-path data/test_site \
    --output-path outputs/tea
```

**Advanced options:**
```bash
python -m methods.tea.batch \
    --base-path /path/to/fluxnet/sites \
    --output-path /path/to/outputs \
    --pattern "FLX_.*_FLUXNET.*"
```

**Arguments:**
- `--base-path`: Directory containing FLUXNET site folders (default: `data/test_site`)
- `--output-path`: Output directory (default: `outputs/tea`)
- `--pattern`: Regular expression for matching site folders (default: FLUXNET/AmeriFlux pattern)

### Python API

**Simple partitioning (recommended):**
```python
from methods.tea.TEA import simplePartition
import numpy as np

# Prepare input arrays (all half-hourly)
timestamp = np.arange(0, len(data)) * 1800  # seconds since start
ET = data['LE_F_MDS'].values / 28.94  # Convert W/m² to mm/30min
GPP = data['GPP_NT_VUT_REF'].values
RH = data['RH'].values
Rg = data['SW_IN_F'].values
Rg_pot = data['SW_IN_POT'].values
Tair = data['TA_F_MDS'].values
VPD = data['VPD_F_MDS'].values
precip = data['P_ERA'].values
u = data['WS'].values

# Run TEA
TEA_T, TEA_E, TEA_WUE = simplePartition(
    timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u
)
```

**Advanced usage with custom parameters:**
```python
from methods.tea.TEA import partition

T, E, WUE = partition(
    timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u,
    # Optional parameters:
    quantile=0.75,           # QRF quantile (default: 0.75)
    n_trees=100,             # Number of trees (default: 50)
    min_samples_leaf=5,      # Min samples per leaf (default: 5)
    cswi_percentile=50       # CSWI threshold percentile (default: 50)
)
```

## Input Data Requirements

**Format:** FLUXNET2015 or AmeriFlux half-hourly CSV files

**Required columns** (standard names):
- `TIMESTAMP_START`, `TIMESTAMP_END` - Timestamps (YYYYMMDDHHMM)
- `LE_F_MDS` - Latent heat flux (W m⁻²)
- `GPP_NT_VUT_REF` or `GPP_NT_VUT_MEAN` - GPP (μmol CO₂ m⁻² s⁻¹)
- `TA_F_MDS` or `TA_F` - Air temperature (°C)
- `VPD_F_MDS` or `VPD_F` - Vapor pressure deficit (hPa)
- `RH` - Relative humidity (%)
- `SW_IN_F` - Incoming shortwave radiation (W m⁻²)
- `SW_IN_POT` - Potential incoming shortwave (W m⁻²)
- `P_ERA` or `P` - Precipitation (mm per timestep)
- `WS` - Wind speed (m s⁻¹)

## Output Files

For each processed site (e.g., FI-Hyy):

```
outputs/tea/
└── FI-Hyy_TEA_results.csv
```

### Output Variables

**CSV columns:**
- `timestamp` - Time since start (seconds)
- `datetime` - Date and time (YYYY-MM-DD HH:MM:SS)
- `TEA_T` - Transpiration (mm per 30-min)
- `TEA_E` - Evaporation (mm per 30-min)
- `TEA_WUE` - Water use efficiency (g C / kg H₂O)

## Method Details

### CSWI (Conservative Surface Water Index)

Tracks cumulative water availability:
```
CSWI = cumsum(P - ET)
```

High CSWI → moist conditions → ideal for training WUE model

### DWCI (Diurnal Water-Carbon coupling Index)

Measures consistency of daytime GPP-ET relationship:
```
DWCI = 1 - |R_gpp - R_et|
```

High DWCI → tight GPP-ET coupling → unstressed conditions

### Quantile Random Forest

Non-parametric regression that predicts conditional quantiles. Returns upper bound (75th percentile)
estimate of WUE for robust transpiration estimates.

## Module Descriptions

**batch.py** - Batch processing entry point
**TEA/TEA.py** - Main partitioning functions
**TEA/core.py** - Quantile Random Forest implementation (Numba-optimized)
**TEA/PreProc.py** - Data preprocessing and feature engineering
**TEA/PotRad.py** - Solar geometry and potential radiation
**TEA/CSWI.py**, **DWCI.py**, **DiurnalCentroid.py** - Derived indices

## Examples

### Process Single Site

```bash
python -m methods.tea.batch \
    --base-path data/test_site \
    --output-path outputs/tea_test
```

### Python Script

```python
import pandas as pd
from methods.tea.TEA import simplePartition

# Load data
df = pd.read_csv('FI-Hyy_data.csv')

# Prepare inputs (convert timestamps to seconds since start)
timestamp = (pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M') -
             pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')
ET = df['LE_F_MDS'] / 28.94  # W/m² to mm/30min
GPP = df['GPP_NT_VUT_REF']
# ... (other variables)

# Run TEA
T, E, WUE = simplePartition(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u)

# Save results
result = pd.DataFrame({'timestamp': timestamp, 'T': T, 'E': E, 'WUE': WUE})
result.to_csv('tea_output.csv', index=False)
```

## Troubleshooting

**All NaN outputs:**
Insufficient ideal condition data for QRF training. Ensure data includes growing season
with precipitation events.

**QRF training failed:**
Too few valid samples. Use longer time series (2-3 years recommended).

**Many negative E values:**
QRF overestimates WUE during dry periods. Code automatically sets E=0, T=ET.

**Very high WUE values:**
Check input data quality. Filter output to reasonable WUE range (1-50 g C/kg H₂O).

## Performance

**Typical processing time** (Intel i7, 16GB RAM):
- 1 site, 3 years: ~2-5 minutes
- 10 sites: ~20-40 minutes

**Memory usage:** ~1-2GB per site during QRF training

## Tutorials

Interactive Jupyter tutorial available:

```bash
jupyter lab notebooks/TEA_tutorial.ipynb
```

Or see the original tutorial in `tutorial.py`.

## References

```bibtex
@article{nelson2018water,
  title={Water-stress-induced breakdown of carbon--water relations: indicators from diurnal FLUXNET patterns},
  author={Nelson, Jacob A and Carvalhais, Nuno and Migliavacca, Mirco and Reichstein, Markus and Jung, Martin},
  journal={Biogeosciences},
  volume={15},
  number={8},
  pages={2433--2447},
  year={2018},
  doi={10.5194/bg-15-2433-2018}
}
```

## License

See [LICENSE](LICENSE) file for details.

## Support

- **Issues:** Open a GitHub issue
- **Original method:** https://github.com/jnelson18/ecosystem-transpiration
- **Paper:** Nelson et al. (2018), Biogeosciences
