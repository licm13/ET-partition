# Perez-Priego Method - Optimality-based ET Partitioning

## Overview

The Perez-Priego method partitions evapotranspiration using stomatal conductance models
based on optimality theory. It estimates transpiration from physiological principles
and derives evaporation as the residual.

**Original Publication:**
> Perez-Priego, O., Tjoelker, M. G., Rambal, S., Migliavacca, M., Ladreiter-Knauss, T.,
> Christiansen, J. R., Escudero, A., Morillas, L., & Loustau, D. (2018). Partitioning eddy
> covariance water flux components using physiological and micrometeorological approaches.
> *Journal of Geophysical Research: Biogeosciences*, 123(10), 3353-3370.
> https://doi.org/10.1029/2018JG004637

## Method Principle

The method is based on the following concepts:

1. **Stomatal optimization**: Plants optimize stomatal conductance to maximize carbon gain
   while minimizing water loss
2. **Physiological model**: Stomatal conductance (gs) is modeled as a function of environmental
   variables (VPD, Tair, photosynthesis)
3. **Parameter estimation**: Optimal parameters (Chi_o, WUE_o) are estimated from observed
   data using moving windows
4. **Transpiration calculation**: T is calculated from modeled gs and environmental conditions

**Key equations:**
```
Chi = Ci / Ca  (ratio of internal to ambient CO₂)
gs = g1 * A / Ca * (1 + RH^g2) / VPD^g3  (stomatal conductance model)
T = gs * VPD / P  (transpiration from gs)
E = ET - T  (evaporation as residual)
```

Where:
- `gs` = Stomatal conductance (mol m⁻² s⁻¹)
- `A` = Net photosynthesis (μmol CO₂ m⁻² s⁻¹)
- `Ca` = Ambient CO₂ concentration (ppm)
- `VPD` = Vapor pressure deficit (kPa)
- `RH` = Relative humidity (fraction)
- `P` = Atmospheric pressure (kPa)
- `g1, g2, g3` = Empirical parameters

## Processing Workflow

```
Input: FLUXNET half-hourly CSV
        ↓
1. Data Loading & Preprocessing
   - Load required variables
   - Filter daytime data (NIGHT == 0)
   - Load site metadata (elevation)
        ↓
2. Long-term Parameter Calculation
   - Calculate Chi_o (optimal Chi)
   - Calculate WUE_o (optimal WUE)
   - Use full dataset statistics
        ↓
3. Moving Window Processing (5-day windows)
   For each window:
   - Filter valid daytime data
   - Optimize parameters (g1, g2, g3, Lambda)
   - Calculate stomatal conductance
   - Estimate transpiration
        ↓
4. Post-processing
   - Clip negative evaporation to 0
   - Aggregate to daily means
   - Generate diagnostic plots
        ↓
Output: Half-hourly T and E, diagnostic plots
```

## Installation

Perez-Priego method is part of the ET-partition project. Install dependencies:

```bash
pip install -e .
```

Or install specific dependencies:
```bash
pip install numpy pandas scipy matplotlib openpyxl
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
python -m methods.perez_priego.batch \
    --base-path data/test_site \
    --output-path outputs/perez_priego
```

**With site metadata:**
```bash
python -m methods.perez_priego.batch \
    --base-path /path/to/fluxnet/sites \
    --output-path /path/to/outputs \
    --site-metadata site_metadata.xlsx \
    --default-altitude 0.5
```

**Arguments:**
- `--base-path`: Directory containing FLUXNET site folders (default: `data/test_site`)
- `--output-path`: Output directory (default: `outputs/perez_priego`)
- `--site-metadata`: Excel file with `SITE_ID` and `LOCATION_ELEV` columns (optional)
- `--default-altitude`: Default elevation in km if metadata missing (default: 0.5)

### Python API

```python
from methods.perez_priego import et_partitioning_functions as ppf
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('site_data.csv')

# Calculate long-term parameters
chi_o = ppf.calculate_chi_o(df['GPP'].values, df['VPD'].values)
wue_o = ppf.calculate_WUE_o(df['GPP'].values, df['LE'].values)

# Process 5-day window
window_data = df.iloc[0:240]  # 5 days of half-hourly data

# Optimize parameters
params = ppf.optimal_parameters(
    window_data['GPP'].values,
    window_data['VPD'].values,
    window_data['TA'].values,
    altitude_km=0.5
)

# Calculate transpiration
T = ppf.transpiration_model(
    window_data['GPP'].values,
    window_data['VPD'].values,
    window_data['TA'].values,
    params,
    altitude_km=0.5
)

# Calculate evaporation
E = window_data['LE'].values / 28.94 - T  # Convert LE to mm/30min
E = np.maximum(E, 0)  # Clip negative values
```

### Core Functions

```python
from methods.perez_priego import et_partitioning_functions as ppf

# Calculate optimal Chi (Ci/Ca ratio)
chi_o = ppf.calculate_chi_o(GPP, VPD)

# Calculate optimal water use efficiency
wue_o = ppf.calculate_WUE_o(GPP, LE)

# Stomatal conductance model
gs = ppf.gc_model(GPP, VPD, RH, params)

# Optimize model parameters
params = ppf.optimal_parameters(GPP, VPD, Tair, altitude_km)

# Transpiration from stomatal model
T = ppf.transpiration_model(GPP, VPD, Tair, params, altitude_km)
```

## Input Data Requirements

**Format:** FLUXNET2015 half-hourly CSV files

**Required columns** (standard FLUXNET2015 names):
- `TIMESTAMP_START`, `TIMESTAMP_END` - Timestamps (YYYYMMDDHHMM)
- `LE_F_MDS` or `LE_F` - Latent heat flux (W m⁻²)
- `GPP_NT_VUT_MEAN` or `GPP_NT_VUT_REF` - GPP (μmol CO₂ m⁻² s⁻¹)
- `VPD_F` or `VPD_F_MDS` - Vapor pressure deficit (hPa)
- `TA_F` or `TA_F_MDS` - Air temperature (°C)
- `RH` - Relative humidity (%) [if available]
- `NIGHT` - Nighttime flag (0 = day, 1 = night)

**Optional:**
- Site metadata Excel file with columns:
  - `SITE_ID` - Site code (e.g., "FI-Hyy")
  - `LOCATION_ELEV` - Elevation in meters

**File structure:**
```
data/
└── site_folder/
    └── FLX_<SITE>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#/
        └── FLX_<SITE>_FLUXNET2015_FULLSET_HH_YYYY-YYYY_#-#.csv
```

## Output Files

For each processed site (e.g., FI-Hyy):

```
outputs/perez_priego/
├── FI-Hyy_pp_output.csv           # Processed data with T and E estimates
├── FI-Hyy_plot.png                # Diagnostic plot (daily mean fluxes)
└── missing_altitude_sites.csv     # List of sites with missing elevation
```

### Output Variables

**CSV columns:**
- Original columns from input data, plus:
- `transpiration` - Half-hourly transpiration (mm per 30-min)
- `evaporation` - Half-hourly evaporation (mm per 30-min)
- `T_ET_ratio` - Transpiration fraction (0-1)

### Diagnostic Plots

The diagnostic plot shows daily mean time series:
- Blue: Total ET
- Green: Transpiration (T)
- Red: Evaporation (E)

## Method Details

### Optimality Theory

The method assumes plants operate near an optimal point where the marginal carbon gain
per unit water loss is maximized. This leads to a relationship between stomatal conductance,
photosynthesis, and environmental conditions.

### Moving Window Approach

Parameters are estimated using 5-day sliding windows:
- Each window: 240 half-hourly measurements (daytime only)
- Optimization: Minimize error between predicted and observed fluxes
- Overlap: Windows slide by 1 day

### Parameter Estimation

Four parameters are optimized per window:
- **g1**: Base stomatal conductance sensitivity
- **g2**: RH response exponent
- **g3**: VPD response exponent
- **Lambda**: Marginal water use efficiency

Optimization uses scipy's differential evolution algorithm.

### Elevation Correction

Atmospheric pressure (required for transpiration calculation) is estimated from elevation:
```python
P = 101.325 * (1 - 0.0065 * elev / 288.15)^5.255
```

Where `elev` is in meters.

## Module Descriptions

**batch.py** (247 lines)
- Main batch processing script
- Handles directory scanning and site processing
- Loads site metadata for elevation
- Generates diagnostic plots

**et_partitioning_functions.py** (200+ lines)
- Core partitioning functions
- Stomatal conductance models
- Parameter optimization routines
- Transpiration calculation

**main_debug.py**
- Debug/development script
- Single-site testing

## Examples

### Example 1: Process Single Site

```bash
python -m methods.perez_priego.batch \
    --base-path data/test_site/FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3 \
    --output-path outputs/pp_test
```

### Example 2: Batch Process with Metadata

Prepare `site_metadata.xlsx`:
```
SITE_ID     LOCATION_ELEV
FI-Hyy      181
US-Ha1      340
DE-Tha      385
```

Run batch processing:
```bash
python -m methods.perez_priego.batch \
    --base-path /data/fluxnet_sites \
    --output-path /results/pp_global \
    --site-metadata site_metadata.xlsx
```

### Example 3: Custom Default Elevation

```bash
python -m methods.perez_priego.batch \
    --base-path /data/mountain_sites \
    --output-path /results/pp_mountains \
    --default-altitude 2.0  # 2000m default
```

## Troubleshooting

**Problem: "Site elevation not found"**
- **Solution**: Provide `--site-metadata` Excel file or adjust `--default-altitude`.
  Missing sites are listed in `missing_altitude_sites.csv`.

**Problem: "Optimization failed"**
- **Cause**: Insufficient valid data in 5-day window or poor initial conditions
- **Solution**: Check data quality. The method requires several good-quality daytime
  measurements per window.

**Problem: "Many negative E values"**
- **Cause**: Stomatal model overestimates T
- **Solution**: This is handled by clipping E to 0. If widespread, check:
  - Site elevation is correct
  - GPP and LE data quality
  - VPD measurements

**Problem: "All NaN outputs"**
- **Cause**: No daytime data or missing NIGHT flag
- **Solution**: Ensure input CSV has `NIGHT` column with 0 = day, 1 = night.

## Performance

**Typical processing time** (Intel i7, 16GB RAM):
- 1 site, 3 years: ~1-3 minutes
- 10 sites: ~10-30 minutes

**Memory usage:**
- Low (~200-500MB per site)

**Optimization:**
- Daytime-only processing reduces computation
- Moving window approach parallelizable (future work)

## Advantages and Limitations

**Advantages:**
- Physically-based approach (stomatal physiology)
- Half-hourly time resolution
- Relatively low computational cost

**Limitations:**
- Requires site elevation (affects pressure estimate)
- Assumes C3 plant physiology
- Moving window may smooth short-term variations
- Negative E values clipped to 0 (loss of information)

## References

**Primary citation:**
```bibtex
@article{perezpriego2018partitioning,
  title={Partitioning eddy covariance water flux components using physiological and micrometeorological approaches},
  author={Perez-Priego, Oscar and Tjoelker, Mark G and Rambal, Serge and Migliavacca, Mirco and Ladreiter-Knauss, Thomas and Christiansen, Jesper R and Escudero, Adri{\'a}n and Morillas, Lourdes and Loustau, Denis},
  journal={Journal of Geophysical Research: Biogeosciences},
  volume={123},
  number={10},
  pages={3353--3370},
  year={2018},
  doi={10.1029/2018JG004637}
}
```

**Related work:**
- Medlyn stomatal conductance model: Medlyn et al. (2011), Global Change Biology
- Optimality theory: Cowan & Farquhar (1977), Symposia of the Society for Experimental Biology

## Support

For issues with this implementation, open a GitHub issue.

For questions about the original method, refer to Perez-Priego et al. (2018) or contact
the authors.
