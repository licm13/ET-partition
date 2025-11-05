import numpy as np
from scipy.optimize import fmin

# Configuration constants
MIN_DAYS_PER_YEAR = 5  # Minimum number of days required in a year for Zhou partitioning
MIN_HALFHOURS_PER_DAY = 1  # Minimum number of half-hours needed for daily T calculation
MIN_HALFHOURS_PER_8DAY = 1  # Minimum number of half-hours needed for 8-day T calculation

# Main functions:

def calculate_rain_flag(precipitation, potential_et, steps_per_day, hourly_mask):
    """
    Calculate precipitation flag for Zhou partitioning.
    
    Excludes days with precipitation and subsequent days based on 
    precipitation amount relative to potential evapotranspiration.
    
    Parameters
    ----------
    precipitation : array
        Precipitation values
    potential_et : array
        Potential evapotranspiration values
    steps_per_day : int
        Number of timesteps per day (typically 48 for half-hourly)
    hourly_mask : bool array
        Mask for valid timesteps
        
    Returns
    -------
    array
        Boolean mask (True = exclude from uWUEp calculation)
    """
    daily_precip = precipitation[hourly_mask].reshape(-1, steps_per_day).sum(axis=1)
    daily_pet = potential_et[hourly_mask].reshape(-1, steps_per_day).sum(axis=1)

    precip_mask = np.isfinite(daily_precip) & np.isfinite(daily_pet)

    for day_idx in range(precip_mask.shape[0]):
        if daily_precip[day_idx] > 0:
            precip_mask[day_idx] = False
            # Exclude next day if precip > PET
            if (daily_precip[day_idx] > daily_pet[day_idx]) and (daily_precip.shape[0] - day_idx > 1):
                precip_mask[day_idx + 1] = False
            # Exclude 2 days if precip > 2*PET
            if (daily_precip[day_idx] > daily_pet[day_idx] * 2) and (daily_precip.shape[0] - day_idx > 2):
                precip_mask[day_idx + 2] = False
    
    return precip_mask.repeat(steps_per_day)

def build_zhou_masks(dataset, steps_per_day=48, hourly_mask=None, gpp_variable='GPP_NT'):
    """
    Build quality control and condition masks for Zhou partitioning.
    
    Creates masks to identify valid data for actual (uWUEa) and potential (uWUEp)
    water use efficiency calculations.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Input dataset with required variables
    steps_per_day : int, optional
        Number of timesteps per day (default: 48 for half-hourly)
    hourly_mask : bool array, optional
        Pre-existing mask for valid timesteps
    gpp_variable : str, optional
        Name of GPP variable in dataset (default: 'GPP_NT')
        
    Returns
    -------
    tuple of arrays
        (actual_wue_mask, potential_wue_mask) - Boolean masks for data selection
    """
    if (hourly_mask is None) and (steps_per_day == 48):
        hourly_mask = np.ones(dataset.LE.shape).astype(bool)

    quality_mask = np.ones(dataset.LE.shape).astype(bool)
    nonzero_mask = np.ones(dataset.LE.shape).astype(bool)

    # Build quality control mask from QC flags
    for var in ['NEE', 'LE', 'TA', 'VPD']:
        qc_values = dataset[var + '_QC'].values
        qc_values[qc_values < 0] = 3
        qc_values[~np.isfinite(qc_values)] = 3
        quality_mask &= qc_values < 2
    
    # Ensure required variables are finite and valid
    for var in [gpp_variable, 'ET', 'TA', 'VPD', 'NETRAD']:
        quality_mask &= np.isfinite(dataset[var].values)
        quality_mask &= dataset[var].values > -9000

    # Build mask for non-zero values
    for var in [gpp_variable, 'ET', 'NETRAD', 'VPD']:
        nonzero_mask &= dataset[var].values > 0

    # Build growing season mask (GPP > 10% of 95th percentile)
    gpp_daily = dataset[gpp_variable].values[hourly_mask].reshape(-1, steps_per_day).mean(axis=1)
    gpp_threshold = 0.10 * np.percentile(gpp_daily, 95)
    season_mask = np.repeat(gpp_daily > gpp_threshold, steps_per_day)

    # Build precipitation mask
    precip_mask = calculate_rain_flag(
        dataset.P.values, dataset.PET.values, steps_per_day, hourly_mask
    )

    # Combine masks
    actual_wue_mask = nonzero_mask & quality_mask
    potential_wue_mask = nonzero_mask & quality_mask & precip_mask & season_mask

    return actual_wue_mask, potential_wue_mask



def quantreg(x,y,PolyDeg=1,rho=0.95,weights=None):
    '''quantreg(x,y,PolyDeg=1,rho=0.95)

    Quantile regression

    Fits a polynomial function (of degree PolyDeg) using quantile regression based on a percentile (rho).
    Based on script by Dr. Phillip M. Feldman, and based on method by Koenker, Roger, and
    Gilbert Bassett Jr. “Regression Quantiles.” Econometrica: Journal of
    the Econometric Society, 1978, 33–50.


    Parameters
    ----------
    x : list or list like
        independent variable
    y : list or list like
        dependent variable
    PolyDeg : int
        The degree of the polynomial function
    rho : float between 0-1
        The percentile to fit to, must be between 0-1
    weights : list or list like
        Vector to weight each point, must be same size as x

     Returns
    -------
    list
        The resulting parameters in order of degree from low to high
    '''
    def model(x, beta):
       """
       This example defines the model as a polynomial, where the coefficients of the
       polynomial are passed via `beta`.
       """
       if PolyDeg == 0:
           return x*beta
       else:
           return polyval(x, beta)

    N_coefficients=PolyDeg+1

    def tilted_abs(rho, x, weights):
       """
       OVERVIEW

       The tilted absolute value function is used in quantile regression.


       INPUTS

       rho: This parameter is a probability, and thus takes values between 0 and 1.

       x: This parameter represents a value of the independent variable, and in
       general takes any real value (float) or NumPy array of floats.
       """

       return weights * x * (rho - (x < 0))

    def objective(beta, rho, weights):
       """
       The objective function to be minimized is the sum of the tilted absolute
       values of the differences between the observations and the model.
       """
       return tilted_abs(rho, y - model(x, beta), weights).sum()

    # Build weights if they don't exits:
    if weights is None:
        weights=np.ones(x.shape)

    # Define starting point for optimization:
    beta_0= np.zeros(N_coefficients)
    if N_coefficients >= 2:
       beta_0[1]= 1.0

    # `beta_hat[i]` will store the parameter estimates for the quantile
    # corresponding to `fractions[i]`:
    beta_hat= []

    #for i, fraction in enumerate(fractions):
    beta_hat.append( fmin(objective, x0=beta_0, args=(rho,weights), xtol=1e-8,
      disp=False, maxiter=3000) )
    return(beta_hat)


def zhou_part(evapotranspiration, gpp_times_vpd_sqrt, actual_mask, potential_mask, 
              steps_per_day=48, hourly_mask=None, percentile=0.95):
    """
    ET partitioning based on Zhou et al. 2016.
    
    Calculates two estimates of underlying water use efficiency (uWUE):
    - uWUEa: actual WUE based on daily or 8-day window
    - uWUEp: potential WUE based on single year
    Then calculates T/ET ratio as uWUEa/uWUEp.

    Parameters
    ----------
    evapotranspiration : array
        Evapotranspiration (mm per timestep)
    gpp_times_vpd_sqrt : array
        GPP * sqrt(VPD) in (gC hPa^0.5 m^-2 d^-1)
    actual_mask : bool array
        Boolean mask where True indicates timesteps for calculating uWUEa
    potential_mask : bool array
        Boolean mask where True indicates timesteps for calculating uWUEp
    steps_per_day : int, optional
        Number of timesteps in a day (48 for half-hourly, 24 for hourly)
    hourly_mask : bool array, optional
        Boolean mask for hourly averaged dataset
    percentile : float, optional
        Percentile for quantile regression (0-1), default 0.95

    Returns
    -------
    potential_wue : float
        Potential underlying water use efficiency (uWUEp)
    daily_transpiration : array
        Estimated daily transpiration (mm d^-1)
    transpiration_8day : array
        Estimated transpiration using 8-day moving window (mm d^-1)

    References
    ----------
    Zhou, S., Yu, B., Zhang, Y., Huang, Y., & Wang, G. (2016). 
    Partitioning evapotranspiration based on the concept of underlying 
    water use efficiency. Water Resources Research, 52(2), 1160–1175. 
    https://doi.org/10.1002/2015WR017766
    """
    if (hourly_mask is None) and (steps_per_day == 48):
        hourly_mask = np.ones(evapotranspiration.shape).astype(bool)

    # Calculate potential uWUE using quantile regression
    potential_wue = quantreg(
        evapotranspiration[potential_mask],
        gpp_times_vpd_sqrt[potential_mask],
        PolyDeg=0,
        rho=percentile
    )[0][0]

    # Reshape arrays once for efficiency
    et_daily = evapotranspiration.reshape(-1, steps_per_day)
    gxv_daily = gpp_times_vpd_sqrt.reshape(-1, steps_per_day)
    num_days = et_daily.shape[0]
    
    # Update mask to include finite values
    valid_mask = np.isfinite(evapotranspiration) & np.isfinite(gpp_times_vpd_sqrt)
    valid_mask_daily = valid_mask.reshape(-1, steps_per_day)
    
    # Calculate daily actual uWUE
    actual_wue_daily = np.full(num_days, np.nan)
    
    for day_idx in range(num_days):
        day_mask = valid_mask_daily[day_idx]
        if day_mask.sum() >= MIN_HALFHOURS_PER_DAY:
            et_valid = et_daily[day_idx][day_mask][:, np.newaxis]
            gxv_valid = gxv_daily[day_idx][day_mask][:, np.newaxis]
            slope, _, _, _ = np.linalg.lstsq(et_valid, gxv_valid, rcond=None)
            actual_wue_daily[day_idx] = slope[0]
    
    # Calculate 8-day window actual uWUE
    actual_wue_8day = np.full(num_days, np.nan)
    
    for day_idx in range(num_days):
        # Determine window boundaries
        if day_idx < 4:
            window_start_day = 4
        elif day_idx > num_days - 4:
            window_start_day = num_days - 8
        else:
            window_start_day = day_idx - 4
        
        window_start = window_start_day * steps_per_day
        window_end = (window_start_day + 8) * steps_per_day
        
        if valid_mask[window_start:window_end].sum() >= MIN_HALFHOURS_PER_8DAY:
            window_mask = valid_mask[window_start:window_end]
            et_valid = evapotranspiration[window_start:window_end][window_mask][:, np.newaxis]
            gxv_valid = gpp_times_vpd_sqrt[window_start:window_end][window_mask][:, np.newaxis]
            slope, _, _, _ = np.linalg.lstsq(et_valid, gxv_valid, rcond=None)
            actual_wue_8day[day_idx] = slope[0]
    
    # Calculate daily ET sum
    et_daily_sum = evapotranspiration[hourly_mask].reshape(-1, steps_per_day).sum(axis=1)
    
    # Calculate transpiration estimates
    transpiration_ratio_daily = actual_wue_daily / potential_wue
    daily_transpiration = et_daily_sum * transpiration_ratio_daily
    
    transpiration_ratio_8day = actual_wue_8day / potential_wue
    transpiration_8day = et_daily_sum * transpiration_ratio_8day
    
    return potential_wue, daily_transpiration, transpiration_8day
