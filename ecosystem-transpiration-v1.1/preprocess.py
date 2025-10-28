import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

with open('BerkeleyConversion.json') as f:
    BerkeleyConversion = json.load(f)
with open('Units.json') as f:
    Units = json.load(f)
with open('LongNames.json') as f:
    LongNames = json.load(f)

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        return float(s)
    except ValueError:
        return s

def get_metadata(site):
   """ Extracts the site metadata from http://sites.fluxdata.org/{site}/."""
   from bs4 import BeautifulSoup
   import requests
   page = requests.get("http://sites.fluxdata.org/{site}/".format(site = site))
   soup = BeautifulSoup(page.content, 'html.parser')
   metadata = {}
   maininfo = soup.findAll("table", {"class": "maininfo"})[0]
   table    = maininfo.find_all('tr')
   rows     = [r for r in table if len(r)>1]
   metadata = {}
   for r in rows:
       for i in r.children:
           if i.attrs['class'][0]=='label':
               label = i.get_text().rstrip(':')
           elif i.attrs['class'][0]=='value':
               value = i.get_text()
           else:
               raise RuntimeError('unknown line from web info table: '+"http://sites.fluxdata.org/{site}/".format(site = site))
       metadata[label] = is_number_tryexcept(value.strip())
   metadata['contact_name']  = metadata['Tower Team'].split('\n')[0][4:].split(' <')[0]
   metadata['contact_email'] = metadata['Tower Team'].split('\n')[0][4:].split(' <')[1].split('>')[0]
   metadata['dataset']  = 'FLUXNET2015'
   return(metadata)

def build_dataset(_file):
    """build_dataset(_file)
    
    Builds an xarray object from a standard FLUXNET2015 half-houlry .csv file,
    including metadata. Only a subset of data is used.
    
    Parameters
    ----------
    _file : string
        location of the FLUXNET2015 .csv file

    Returns
    -------
    ds : xarray.Dataset
        xarray dataset of the data
    """
    ds = pd.read_csv(_file,
                     sep             = ',',
                     keep_default_na = False,
                     usecols         = ['TIMESTAMP_START','TIMESTAMP_END']+list(BerkeleyConversion.keys()))
    site = _file.split('_')[1]
    ds['TIMESTAMP_START'] = pd.to_datetime(ds['TIMESTAMP_START'], format="%Y%m%d%H%M" )
    ds['TIMESTAMP_END'] = pd.to_datetime(ds['TIMESTAMP_END'], format="%Y%m%d%H%M" )
    ds['time'] = np.stack([ds['TIMESTAMP_START'].values.astype(np.float64),ds['TIMESTAMP_END'].values.astype(np.float64)]).mean(axis=0).astype('datetime64[ns]').astype('datetime64[m]').astype('datetime64[ns]')
    ds = ds.set_index('time').to_xarray()
    for var in BerkeleyConversion.keys():
        if var not in ds.variables:
            ds[var] = ds.TA_F_MDS * np.nan
    ds = ds[list(BerkeleyConversion.keys())].rename(BerkeleyConversion)
    ds['time'] = ds.time.values.astype('datetime64[m]').astype('datetime64[ns]')
    try:
        metadata = get_metadata(site)
        ds = ds.assign_attrs(**metadata)
    except Exception as e:
        warnings.warn('Metadata could not be loaded and will not be included in the dataset')

    if np.isnan(ds.H.values[::2]).all():
        ds.attrs['agg_code'] = 'HR'
    else:
        ds.attrs['agg_code'] = 'HH'
    for _var in Units.keys():
        if _var not in ds.variables:
            ds[_var] = ds['SW_IN_POT']*np.nan
        ds[_var] = ds[_var].assign_attrs(units=Units[_var],long_name=LongNames[_var])
    ds = ds.where(ds!=-9999)
    
    ds.attrs['build_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return(ds)

def build_dataset_modified(_file):
    """build_dataset(_file)
    
    Builds an xarray object from a standard FLUXNET2015 half-houlry .csv file,
    including metadata. Only a subset of data is used.
    
    Parameters
    ----------
    _file : string
        location of the FLUXNET2015 .csv file

    Returns
    -------
    ds : xarray.Dataset
        xarray dataset of the data
    """
    # 這裡的 read_csv 會根據精簡後的 BerkeleyConversion.json 只讀取必要的列
    ds = pd.read_csv(_file,
                     sep             = ',',
                     keep_default_na = False,
                     usecols         = ['TIMESTAMP_START','TIMESTAMP_END']+list(BerkeleyConversion.keys()))
    site = _file.split('_')[1]
    ds['TIMESTAMP_START'] = pd.to_datetime(ds['TIMESTAMP_START'], format="%Y%m%d%H%M" )
    ds['TIMESTAMP_END'] = pd.to_datetime(ds['TIMESTAMP_END'], format="%Y%m%d%H%M" )
    ds['time'] = np.stack([ds['TIMESTAMP_START'].values.astype(np.float64),ds['TIMESTAMP_END'].values.astype(np.float64)]).mean(axis=0).astype('datetime64[ns]').astype('datetime64[m]').astype('datetime64[ns]')
    ds = ds.set_index('time').to_xarray()
    
    # --- 刪除 ---
    # 下面這個循環是多餘的，因為 read_csv 已經保證了所有列都被讀取。
    # 如果列不存在，read_csv 會直接報錯，所以這個檢查沒有意義。
    # for var in BerkeleyConversion.keys():
    #     if var not in ds.variables:
    #         ds[var] = ds.TA_F_MDS * np.nan
    
    ds = ds[list(BerkeleyConversion.keys())].rename(BerkeleyConversion)
    ds['time'] = ds.time.values.astype('datetime64[m]').astype('datetime64[ns]')
    
    try:
        metadata = get_metadata(site)
        ds = ds.assign_attrs(**metadata)
    except Exception as e:
        warnings.warn('Metadata could not be loaded and will not be included in the dataset')

    if np.isnan(ds.H.values[::2]).all():
        ds.attrs['agg_code'] = 'HR'
    else:
        ds.attrs['agg_code'] = 'HH'
    
    # --- 修改 ---
    # 原來的代碼會遍歷 Units.json 中的所有變量，
    # 並嘗試用一個不再加載的變量(SW_IN_POT)來創建缺失的列，這會導致 KeyError。
    # 新的代碼只會遍歷數據集(ds)中實際存在的變量，並為它們分配單位。
    for _var in ds.variables:
        if _var in Units: # 檢查該變量是否存在於單位和長名稱的定義文件中
            ds[_var] = ds[_var].assign_attrs(units=Units[_var],long_name=LongNames[_var])

    ds = ds.where(ds!=-9999)
    
    ds.attrs['build_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return(ds)
