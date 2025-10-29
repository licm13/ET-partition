# uWUE Method - Underlying Water Use Efficiency

## Overview

The uWUE (Underlying Water Use Efficiency) method partitions evapotranspiration into
transpiration (T) and evaporation (E) based on the concept of potential water use efficiency.

**Original Publication:**
> Zhou, S., Yu, B., Zhang, Y., Huang, Y., & Wang, G. (2016). Partitioning evapotranspiration
> based on the concept of underlying water use efficiency. *Water Resources Research*,
> 52(2), 1160-1175. https://doi.org/10.1002/2015WR017766

## Method Principle

The method is based on the following concepts:

1. **Water Use Efficiency (WUE)**: The ratio of carbon gain to water loss
2. **Underlying WUE (uWUE)**: Potential WUE under optimal conditions (high soil moisture, after precipitation)
3. **Partitioning**: The ratio T/ET equals the ratio of actual to potential uWUE

**Key Equation:**
```
uWUE = GPP × √VPD / T
T/ET = uWUEa / uWUEp
```

Where:
- `GPP` = Gross Primary Productivity (μmol CO₂ m⁻² s⁻¹)
- `VPD` = Vapor Pressure Deficit (hPa)
- `T` = Transpiration (mm day⁻¹)
- `uWUEa` = Actual uWUE (daily)
- `uWUEp` = Potential uWUE (annual, 95th percentile)

## Processing Workflow

```
Input: FLUXNET2015 half-hourly CSV
        ↓
1. Data Loading & Column Mapping (BerkeleyConversion.json)
        ↓
2. Physical Quantity Calculations
   - ET from LE (bigleaf.LE_to_ET)
   - PET (Priestley-Taylor, bigleaf.PT)
   - NETRAD gap-filling
        ↓
3. uWUE Mask Generation (zhou.zhouFlags)
   Conditions:
   - Non-rainy periods (>48h after rain)
   - Quality flags pass
   - Valid GPP, VPD, ET data
        ↓
4. Annual Potential uWUE Estimation (zhou.quantreg)
   - Group by year
   - Quantile regression (95th percentile)
   - uWUEp = slope of GPP×√VPD vs T
        ↓
5. Daily Actual uWUE Estimation
   - Group by day
   - Linear regression
   - uWUEa = slope (if R² > threshold)
        ↓
6. T Calculation (zhou.zhou_part)
   - T = ET × (uWUEa / uWUEp)
   - E = ET - T
        ↓
Output: Daily T, E, diagnostic plots, NetCDF
```

## Installation

The uWUE method is part of the ET-partition project. Install dependencies:

```bash
pip install -e .
```

Or install specific dependencies:
```bash
pip install numpy pandas matplotlib seaborn xarray netCDF4
```

## Usage

### Command Line Interface

**Basic usage:**
```bash
python -m methods.uwue.batch \
    --base-path data/test_site \
    --output-path outputs/uwue
```

**Advanced options:**
```bash
python -m methods.uwue.batch \
    --base-path /path/to/fluxnet/sites \
    --output-path /path/to/outputs \
    --no-plots \
    --pattern "FLX_.*_FLUXNET2015_.*"
```

**Arguments:**
- `--base-path`: Directory containing FLUXNET site folders (default: `data/test_site`)
- `--output-path`: Output directory (default: `outputs/uwue`)
- `--no-plots`: Skip diagnostic plot generation (faster processing)
- `--pattern`: Regular expression for matching site folders

### Python API

```python
from methods.uwue.batch import uWUEBatchProcessor

# Initialize processor
processor = uWUEBatchProcessor(
    base_path="data/my_sites",
    output_path="outputs/uwue_results",
    create_plots=True
)

# Run batch processing
processor.run()

# Generate summary report
processor.generate_summary_report()
```

### Core Functions

```python
from methods.uwue import zhou, bigleaf

# Calculate ET from latent heat flux
ET = bigleaf.LE_to_ET(LE, Tair)  # Returns mm/timestep

# Calculate potential ET (Priestley-Taylor)
PET = bigleaf.PT(Rn, Tair, pa=101.325)

# Generate uWUE mask for optimal conditions
mask = zhou.zhouFlags(
    df,
    quality_flags=['LE_QC', 'NEE_QC'],
    precip_col='P',
    rain_thresh=0.5,
    lag_days=2
)

# Perform uWUE partitioning
T, E, uWUEp_vals, uWUEa_vals = zhou.zhou_part(
    daily_df,
    et_col='ET',
    gpp_col='GPP_NT',
    vpd_col='VPD',
    mask_col='uWUE_mask',
    quantile=0.95
)
```

## Input Data Requirements

**Format:** FLUXNET2015 half-hourly CSV files

**Required columns** (standard FLUXNET2015 names):
- `TIMESTAMP_START`, `TIMESTAMP_END` - Timestamps (YYYYMMDDHHMM)
- `LE_F_MDS` - Latent heat flux (W m⁻²)
- `H_F_MDS` - Sensible heat flux (W m⁻²)
- `G_F_MDS` - Ground heat flux (W m⁻²)
- `NETRAD` - Net radiation (W m⁻²) [optional, will be calculated if missing]
- `GPP_NT_VUT_USTAR50` - GPP nighttime partitioning (μmol CO₂ m⁻² s⁻¹)
- `VPD_F_MDS` - Vapor pressure deficit (hPa)
- `TA_F_MDS` - Air temperature (°C)
- `PA` - Atmospheric pressure (kPa)
- `P` - Precipitation (mm per timestep)
- `LE_F_MDS_QC` - LE quality flag
- `GPP_NT_VUT_USTAR50_QC` - GPP quality flag [if available]

**Column mapping:** If your data uses different column names, edit `BerkeleyConversion.json`.

**File structure:**
```
data/
└── site_folder/
    └── FLX_<SITE>_FLUXNET2015_FULLSET_YYYY-YYYY_#-#/
        └── FLX_<SITE>_FLUXNET2015_FULLSET_HH_YYYY-YYYY_#-#.csv
```

## Output Files

For each processed site (e.g., FI-Hyy), the following outputs are generated:

```
outputs/uwue/
├── FI-Hyy_uWUE_output.csv          # Daily results (CSV)
├── FI-Hyy_uWUE_output.nc           # Daily results (NetCDF with metadata)
├── plots/
│   └── FI-Hyy_uWUE_analysis.png   # Four-panel diagnostic plot
├── processing_summary.txt          # Processing statistics
└── uwue_processing_<timestamp>.log # Detailed processing log
```

### Output Variables

**CSV/NetCDF columns:**
- `date` - Date (YYYY-MM-DD)
- `year`, `month`, `day` - Date components
- `ET` - Total evapotranspiration (mm day⁻¹)
- `T` - Transpiration (mm day⁻¹)
- `E` - Evaporation (mm day⁻¹)
- `T_ET_ratio` - T/ET ratio (0-1)
- `GPP_NT` - Daily GPP (μmol CO₂ m⁻² s⁻¹)
- `VPD` - Daily mean VPD (hPa)
- `uWUEp` - Potential uWUE (annual value)
- `uWUEa` - Actual uWUE (daily value)
- `PET` - Potential ET (mm day⁻¹)
- `uWUE_mask` - Data quality mask (0/1)
- `uWUEa_r2` - R² of daily uWUE regression

### Diagnostic Plots

The four-panel diagnostic plot includes:

1. **Top-left**: uWUE scatter plot (GPP×√VPD vs T) with regression line
   - Shows potential uWUE estimation
   - 95th percentile quantile regression
   - Color-coded by year

2. **Top-right**: Time series of daily T, E, and ET
   - Helps visualize seasonal patterns
   - Identifies data gaps

3. **Bottom-left**: T/ET ratio time series
   - Shows transpiration fraction over time
   - Typical range: 0.4-0.9

4. **Bottom-right**: ET vs PET scatter
   - Validates energy balance
   - Points should fall near 1:1 line

## Configuration Files

**BerkeleyConversion.json**
Maps FLUXNET2015 column names to internal variable names:
```json
{
  "LE_F_MDS": "LE",
  "GPP_NT_VUT_USTAR50": "GPP_NT",
  "VPD_F_MDS": "VPD",
  ...
}
```

**Units.json**
Specifies units for each variable:
```json
{
  "LE": "W m-2",
  "GPP_NT": "umolCO2 m-2 s-1",
  ...
}
```

**LongNames.json**
Provides descriptive names for variables:
```json
{
  "LE": "Latent Heat Flux",
  "GPP_NT": "Gross Primary Productivity (Nighttime partitioning)",
  ...
}
```

## Module Descriptions

**batch.py** (496 lines)
- Main entry point for batch processing
- Class `uWUEBatchProcessor` orchestrates workflow
- Handles directory scanning, parallel processing, logging

**zhou.py** (200+ lines)
- Core uWUE partitioning algorithms
- Key functions: `zhouFlags`, `quantreg`, `zhou_part`
- Implements Zhou et al. (2016) equations

**bigleaf.py** (450+ lines)
- Biophysical calculations from bigleaf R package
- Functions: `LE_to_ET`, `PT`, `Rn_calc`, etc.
- Handles unit conversions and constants

**preprocess.py** (80+ lines)
- Data loading and preprocessing
- Applies JSON configuration mappings
- Timestamp parsing

**self_modify_version.py**
- Modified/experimental implementations
- For development and testing

## Examples

### Example 1: Process Single Site

```bash
python -m methods.uwue.batch \
    --base-path data/test_site/FLX_FI-Hyy_FLUXNET2015_FULLSET_2008-2010_1-3 \
    --output-path outputs/fi_hyy_test
```

### Example 2: Batch Process Multiple Sites Without Plots

```bash
python -m methods.uwue.batch \
    --base-path /data/fluxnet2015_sites \
    --output-path /results/uwue_global \
    --no-plots
```

### Example 3: Custom Pattern Matching

```bash
python -m methods.uwue.batch \
    --base-path /data/mixed_sites \
    --output-path /results/subset \
    --pattern "FLX_(US|CA)-.*_FLUXNET.*"
```

## Troubleshooting

**Problem: "Missing required column: LE"**
- **Solution**: Check that your CSV has FLUXNET2015 standard column names, or
  edit `BerkeleyConversion.json` to map your column names.

**Problem: "Not enough valid data for regression"**
- **Cause**: Insufficient data passing quality flags or uWUE mask conditions
- **Solution**: Check data quality flags and precipitation data. The method requires
  at least a few days of good-quality, non-rainy data per year.

**Problem: "All NaN outputs"**
- **Cause**: No valid daily data after aggregation
- **Solution**: Verify that half-hourly data has valid ET, GPP, and VPD values.
  Check quality flags are not too restrictive.

**Problem: "Negative transpiration values"**
- **Cause**: uWUEa > uWUEp (can happen with noisy data)
- **Solution**: This is expected in some cases. The code clips negative T to 0.
  If it's widespread, check data quality.

## Performance

**Typical processing time** (Intel i7, 16GB RAM):
- 1 site, 3 years: ~30-60 seconds (with plots)
- 10 sites, 3 years each: ~5-10 minutes
- 100 sites: ~1-2 hours

**Memory usage:**
- Low (~500MB per site)
- Can process hundreds of sites in batch

**Optimization tips:**
- Use `--no-plots` to skip plot generation (2x faster)
- Process sites in parallel using shell scripts or workflow managers

## References

**Primary citation:**
```bibtex
@article{zhou2016partitioning,
  title={Partitioning evapotranspiration based on the concept of underlying water use efficiency},
  author={Zhou, Sha and Yu, Bofu and Zhang, Yao and Huang, Yuefei and Wang, Guangqian},
  journal={Water Resources Research},
  volume={52},
  number={2},
  pages={1160--1175},
  year={2016},
  doi={10.1002/2015WR017766}
}
```

**Related methods:**
- bigleaf R package: https://CRAN.R-project.org/package=bigleaf
- FLUXNET2015 dataset: https://fluxnet.org/data/fluxnet2015-dataset/

## Support

For issues specific to this implementation, please open an issue on the GitHub repository.

For questions about the original method, refer to the Zhou et al. (2016) paper or contact
the authors.
