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

def get_1d_array(df, col):
    values = df[col]
    if isinstance(values, pd.DataFrame):
        print(f"[警告] '{col}' 返回二维数组，尝试使用第一列")
        return values.iloc[:, 0].values
    return values.values

def _prepare_data_for_models(par, data, Chi_o):
    df = data.copy()
    df['VPD'] /= 10.0
    q_mask = df['Q'].isna()
    if 'Q_in' in df.columns:
        df.loc[q_mask, 'Q'] = df.loc[q_mask, 'Q_in'] * 2
    df['Q'] = df['Q'].fillna(0)
    
    Photos = get_1d_array(df, 'Photos')
    H = get_1d_array(df, 'H')
    VPD = get_1d_array(df, 'VPD')
    Tair = get_1d_array(df, 'Tair')
    Pair = get_1d_array(df, 'Pair')
    Q = get_1d_array(df, 'Q')
    Ca = get_1d_array(df, 'Ca')
    WS = get_1d_array(df, 'WS')
    Ustar = get_1d_array(df, 'Ustar')

    Cp = 1003.5
    R_gas_constant = 287.058
    M = 0.0289644
    dens = (Pair * 1000) / (R_gas_constant * (Tair + 273.15) + 1e-6)
    Mden = dens / M

    beta = par[3]
    ra_m = WS / (Ustar**2 + 1e-6)
    ra_b = 6.2 * (Ustar + 1e-6)**-0.67
    ra = ra_m + ra_b
    ra_w = ra_m + 2 * (1.05 / 0.71 / 1.57)**(2/3) * ra_b
    ra_c = ra_m + 2 * (1.05 / 0.71)**(2/3) * ra_b
    Tplant = (H * ra / (Cp * dens + 1e-6)) + Tair
    es_plant = 0.61078 * np.exp((17.269 * Tplant) / (237.3 + Tplant))
    es_air = 0.61078 * np.exp((17.269 * Tair) / (237.3 + Tair))
    ea = es_air - VPD
    VPD_plant = np.clip(es_plant - ea, a_min=0, a_max=None)

    Photos_max = np.nanquantile(Photos, 0.90)
    Dmax = df['VPD'][df['Photos'] > Photos_max].mean(skipna=True)
    Chimax = Chi_o * (1 / (1 + beta * (Dmax**0.5 if Dmax > 0 else 0) + 1e-6))
    gcmax_val = np.nanmedian(Photos_max / (Mden[df['Photos'] > Photos_max] * Ca[df['Photos'] > Photos_max] * (1 - Chimax) + 1e-6))
    gcmax_val = gcmax_val if np.isfinite(gcmax_val) else 0.1

    gc_mod = gc_model(par[:3], Q, VPD, Tair, gcmax=gcmax_val)
    gw_mod = 1.6 * gc_mod
    gc_bulk = Mden / (1 / (gc_mod + 1e-6) + ra_c)
    gw_bulk = Mden / (1 / (gw_mod + 1e-6) + ra_w)
    Chi = Chi_o * (1 / (1 + beta * (VPD_plant**0.5 if (VPD_plant > 0).any() else 0) + 1e-6))

    return {
        "gc_bulk": gc_bulk, "gw_bulk": gw_bulk, "Chi": Chi,
        "Ca": Ca, "VPD_plant": VPD_plant, "Pair": Pair
    }
    print(f"[DEBUG] Q shape: {Q.shape}, Ca shape: {Ca.shape}, Chi shape: {Chi.shape}")

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
