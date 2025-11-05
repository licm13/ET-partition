# -*- coding: utf-8 -*-
# @Author: Gemini AI
# @Date: 2025-07-23
# @Description: ET partitioning functions with MCMC timeout protection

import pandas as pd
import numpy as np
import emcee
import time

def calculate_chi_o(data, col_photos, col_vpd, col_tair, c_coef, z):
    df = data.copy()
    df.rename(columns={col_photos: 'Photos', col_vpd: 'VPD', col_tair: 'Tair'}, inplace=True)
    df['VPD'] = df['VPD'] / 10.0
    growth_threshold = df['Photos'].quantile(0.85)
    tmp = df[df['Photos'] > growth_threshold]
    tair_g = tmp['Tair'].mean(skipna=True)
    vpd_g = tmp['VPD'].mean(skipna=True)
    logistic_chi_o = 0.0545 * (tair_g - 25) - 0.58 * np.log(vpd_g) - 0.0815 * z + c_coef
    chi_o = np.exp(logistic_chi_o) / (1 + np.exp(logistic_chi_o))
    return chi_o

def calculate_WUE_o(data, col_photos, col_vpd, col_tair, c_coef, z):
    df = data.copy()
    df.rename(columns={col_photos: 'Photos', col_vpd: 'VPD', col_tair: 'Tair'}, inplace=True)
    df['VPD'] = df['VPD'] / 10.0
    growth_threshold = df['Photos'].quantile(0.85)
    tmp = df[df['Photos'] > growth_threshold]
    tair_g = tmp['Tair'].mean(skipna=True)
    vpd_g = tmp['VPD'].mean(skipna=True)
    chi_o = calculate_chi_o(data, col_photos, col_vpd, col_tair, c_coef, z)
    wue_o = (390 * (1 - chi_o) * 96) / (1.6 * vpd_g) * 0.001
    return wue_o

def gc_model(par, Q, VPD, Tair, gcmax):
    a1, D0, Topt = par[0], par[1], par[2]
    FQ = Q / (Q + a1 + 1e-6)
    Fd = np.exp(-D0 * VPD)
    Tl, Th = 0, 50
    b4 = (Th - Topt) / (Th - Tl)
    b3 = 1 / ((Topt - Tl) * (Th - Topt)**b4)
    temp_term = np.clip(Th - Tair, a_min=0, a_max=None)
    Ftemp = b3 * (Tair - Tl) * temp_term**b4
    Ftemp = np.clip(Ftemp, a_min=0, a_max=None)
    sensitivity_function = FQ * Fd * Ftemp
    max_sens = np.nanmax(sensitivity_function)
    sensitivity_function_scaled = sensitivity_function / (max_sens + 1e-6)
    return gcmax * sensitivity_function_scaled

def get_1d_array(dataframe, column_name):
    """
    Extract 1D array from DataFrame column, handling edge cases.
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe
    column_name : str
        Name of column to extract
        
    Returns
    -------
    np.ndarray
        1D numpy array
    """
    values = dataframe[column_name]
    if isinstance(values, pd.DataFrame):
        print(f"[Warning] Column '{column_name}' returned 2D array, using first column")
        return values.iloc[:, 0].values
    return values.values


def _prepare_data_for_models(parameters, data, chi_optimal):
    """
    Prepare and calculate intermediate variables for physiological models.
    
    Parameters
    ----------
    parameters : array-like
        Model parameters [a1, D0, Topt, beta]
    data : pd.DataFrame
        Input data with required meteorological columns
    chi_optimal : float
        Optimal chi parameter under ideal conditions
        
    Returns
    -------
    dict
        Dictionary containing calculated conductances and environmental variables
    """
    processed_data = data.copy()
    processed_data['VPD'] /= 10.0  # Convert to kPa
    
    # Fill missing radiation with alternative source
    radiation_missing = processed_data['Q'].isna()
    if 'Q_in' in processed_data.columns:
        processed_data.loc[radiation_missing, 'Q'] = processed_data.loc[radiation_missing, 'Q_in'] * 2
    processed_data['Q'] = processed_data['Q'].fillna(0)
    
    # Extract variables as 1D arrays
    photosynthesis = get_1d_array(processed_data, 'Photos')
    sensible_heat = get_1d_array(processed_data, 'H')
    vpd = get_1d_array(processed_data, 'VPD')
    air_temp = get_1d_array(processed_data, 'Tair')
    air_pressure = get_1d_array(processed_data, 'Pair')
    radiation = get_1d_array(processed_data, 'Q')
    co2_concentration = get_1d_array(processed_data, 'Ca')
    wind_speed = get_1d_array(processed_data, 'WS')
    friction_velocity = get_1d_array(processed_data, 'Ustar')

    # Physical constants
    SPECIFIC_HEAT_AIR = 1003.5  # J/(kg·K)
    GAS_CONSTANT_DRY_AIR = 287.058  # J/(kg·K)
    MOLAR_MASS_AIR = 0.0289644  # kg/mol
    
    # Calculate air density and molar density
    air_density = (air_pressure * 1000) / (GAS_CONSTANT_DRY_AIR * (air_temp + 273.15) + 1e-6)
    molar_density = air_density / MOLAR_MASS_AIR

    # Calculate aerodynamic resistances
    beta = parameters[3]
    resistance_momentum = wind_speed / (friction_velocity**2 + 1e-6)
    resistance_boundary = 6.2 * (friction_velocity + 1e-6)**-0.67
    resistance_total = resistance_momentum + resistance_boundary
    resistance_water = resistance_momentum + 2 * (1.05 / 0.71 / 1.57)**(2/3) * resistance_boundary
    resistance_co2 = resistance_momentum + 2 * (1.05 / 0.71)**(2/3) * resistance_boundary
    
    # Calculate plant temperature and VPD at leaf level
    plant_temp = (sensible_heat * resistance_total / (SPECIFIC_HEAT_AIR * air_density + 1e-6)) + air_temp
    saturation_vp_plant = 0.61078 * np.exp((17.269 * plant_temp) / (237.3 + plant_temp))
    saturation_vp_air = 0.61078 * np.exp((17.269 * air_temp) / (237.3 + air_temp))
    actual_vapor_pressure = saturation_vp_air - vpd
    vpd_plant = np.clip(saturation_vp_plant - actual_vapor_pressure, a_min=0, a_max=None)

    # Calculate maximum conductance under optimal conditions
    photosynthesis_max = np.nanquantile(photosynthesis, 0.90)
    vpd_at_max = processed_data['VPD'][processed_data['Photos'] > photosynthesis_max].mean(skipna=True)
    chi_max = chi_optimal * (1 / (1 + beta * (vpd_at_max**0.5 if vpd_at_max > 0 else 0) + 1e-6))
    
    high_productivity_mask = processed_data['Photos'] > photosynthesis_max
    conductance_max_value = np.nanmedian(
        photosynthesis_max / (
            molar_density[high_productivity_mask] * 
            co2_concentration[high_productivity_mask] * 
            (1 - chi_max) + 1e-6
        )
    )
    conductance_max_value = conductance_max_value if np.isfinite(conductance_max_value) else 0.1

    # Calculate conductances
    conductance_co2_modeled = gc_model(parameters[:3], radiation, vpd, air_temp, gcmax=conductance_max_value)
    conductance_water_modeled = 1.6 * conductance_co2_modeled
    conductance_co2_bulk = molar_density / (1 / (conductance_co2_modeled + 1e-6) + resistance_co2)
    conductance_water_bulk = molar_density / (1 / (conductance_water_modeled + 1e-6) + resistance_water)
    chi = chi_optimal * (1 / (1 + beta * (vpd_plant**0.5 if (vpd_plant > 0).any() else 0) + 1e-6))

    return {
        "gc_bulk": conductance_co2_bulk,
        "gw_bulk": conductance_water_bulk,
        "Chi": chi,
        "Ca": co2_concentration,
        "VPD_plant": vpd_plant,
        "Pair": air_pressure
    }

def photos_model(par, data, Chi_o):
    data_renamed = data.rename(columns={'GPP_NT_VUT_MEAN': 'Photos', 'NEE_VUT_USTAR50_JOINTUNC': 'Photos_unc',
                                        'H_F_MDS': 'H', 'VPD_F': 'VPD', 'TA_F': 'Tair', 'PA_F': 'Pair',
                                        'PPFD_IN': 'Q', 'SW_IN_F': 'Q_in', 'CO2_F_MDS': 'Ca',
                                        'USTAR': 'Ustar', 'WS_F': 'WS'})
    prepared = _prepare_data_for_models(par, data_renamed, Chi_o)
    return prepared['gc_bulk'] * prepared['Ca'] * (1 - prepared['Chi'])

def transpiration_model(par, data, Chi_o):
    data_renamed = data.rename(columns={'GPP_NT_VUT_MEAN': 'Photos', 'NEE_VUT_USTAR50_JOINTUNC': 'Photos_unc',
                                        'H_F_MDS': 'H', 'VPD_F': 'VPD', 'TA_F': 'Tair', 'PA_F': 'Pair',
                                        'PPFD_IN': 'Q', 'SW_IN_F': 'Q_in', 'CO2_F_MDS': 'Ca',
                                        'USTAR': 'Ustar', 'WS_F': 'WS'})
    prepared = _prepare_data_for_models(par, data_renamed, Chi_o)
    return prepared['gw_bulk'] * prepared['VPD_plant'] / (prepared['Pair'] + 1e-6) * 1000

def log_prob_function(par, data, Chi_o, WUE_o, par_lower, par_upper):
    if not all(par_lower[i] <= par[i] <= par_upper[i] for i in range(len(par))):
        return -np.inf
    df = data[(data['Photos'] > 0)].dropna(subset=['Photos', 'Q', 'VPD', 'Tair'])
    if len(df) < 10:
        return -np.inf
    Photos = df['Photos'].values
    Photos_unc = df['Photos_unc'].values
    Photos_mod = photos_model(par, df, Chi_o)
    transpiration_mod = transpiration_model(par, df, Chi_o)
    if not (np.all(np.isfinite(Photos_mod)) and np.all(np.isfinite(transpiration_mod))):
        return -np.inf
    WaterCost_i = np.nansum(transpiration_mod) / (np.nansum(Photos_mod) + 1e-6)
    Phi = WaterCost_i * WUE_o
    Photos_unc_threshold = np.maximum(Photos * 0.1, Photos_unc)
    Photos_unc_threshold[Photos_unc_threshold == 0] = 1.0
    FO = np.nansum(((Photos_mod - Photos) / Photos_unc_threshold)**2) / len(Photos_mod)
    return -0.5 * (FO + Phi)

def optimal_parameters(par_lower, par_upper, data, Chi_o, WUE_o):
    print("开始 MCMC 参数优化 ...")
    start_time = time.time()
    max_duration = 30  # 超时秒数
    data_renamed = data.rename(columns={'GPP_NT_VUT_MEAN': 'Photos', 'NEE_VUT_USTAR50_JOINTUNC': 'Photos_unc',
                                        'H_F_MDS': 'H', 'VPD_F': 'VPD', 'TA_F': 'Tair', 'PA_F': 'Pair',
                                        'PPFD_IN': 'Q', 'SW_IN_F': 'Q_in', 'CO2_F_MDS': 'Ca',
                                        'USTAR': 'Ustar', 'WS_F': 'WS'})
    ndim = len(par_lower)
    nwalkers = 10
    nsteps = 100
    nburn = 30
    pos = np.random.rand(nwalkers, ndim) * (np.array(par_upper) - np.array(par_lower)) + np.array(par_lower)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_function,
                                    args=[data_renamed, Chi_o, WUE_o, par_lower, par_upper])

    try:
        for i, _ in enumerate(sampler.sample(pos, iterations=nsteps, progress=False)):
            if time.time() - start_time > max_duration:
                raise TimeoutError("MCMC 优化超时")
        print("MCMC 完成，提取参数...")
        samples = sampler.get_chain(discard=nburn, thin=5, flat=True)
        if samples.shape[0] == 0:
            raise ValueError("无有效样本")
        return np.median(samples, axis=0)
    except Exception as e:
        print(f"[警告] MCMC 失败，使用默认参数：{e}")
        return np.array([50, 0.1, 20, 0.5])
