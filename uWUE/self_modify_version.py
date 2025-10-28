import os
import re
from time import time

# --- 核心库 ---
import xarray as xr
import numpy as np
import pandas as pd
from preprocess import build_dataset_modified
import bigleaf
import zhou

# --- 配置区 ---
# 设置你的数据源根目录 (存放 FLX...csv 文件的位置)
base_path = 'Z:\\LCM\\TEA\\Test' 
# 设置统一的输出目录
output_path = 'Z:\\LCM\\TEA\\Test'
# --- 配置结束 ---

# --- 主程序开始 ---
print(f"开始扫描并处理目录: {base_path}")
print(f"输出结果将保存至: {output_path}")

# 确保输出目录存在，如果不存在则创建
os.makedirs(output_path, exist_ok=True)

# 用于匹配文件夹名称的正则表达式
# 格式为: AMF_SITENAME_FLUXNET_FULLSET_YYYY-YYYY_VERSION
# 注释部分为测试站点FLX_FI-Hyy_FLUXNET2015_FULLSET_HH_2008-2010_1-3.csv

folder_pattern = re.compile(r'^AMF_.*_FLUXNET_FULLSET_\d{4}-\d{4}_\d+-\d+$')
# folder_pattern = re.compile(r'^FLX_.*_FLUXNET2015_FULLSET_\d{4}-\d{4}_\d+-\d+$') 


# 检查根目录是否存在
if not os.path.exists(base_path):
    print(f"错误：找不到数据源路径 '{base_path}'。请确保路径正确。")
else:
    # 列出目录下的所有文件和文件夹
    all_entries = os.listdir(base_path)

    # 遍历根目录下的所有条目
    for folder_name in all_entries:
        folder_path = os.path.join(base_path, folder_name)

        # 检查当前条目是否是文件夹，并且名称是否符合指定格式
        if os.path.isdir(folder_path) and folder_pattern.match(folder_name):
            print(f"\n[处理文件夹]: {folder_name}")

            # 1. 根据文件夹名构建CSV文件的名称
            # 格式为: AMF_SITENAME_FLUXNET_HH_... .csv
            csv_filename = folder_name.replace('_FLUXNET_FULLSET_', '_FLUXNET_FULLSET_HH_') + '.csv'
            # csv_filename = folder_name.replace('_FLUXNET2015_FULLSET_', '_FLUXNET2015_FULLSET_HH_') + '.csv' 单纯测试用的
            csv_filepath = os.path.join(folder_path, csv_filename)

            # 2. 检查对应的CSV文件是否存在
            if not os.path.exists(csv_filepath):
                print(f"  -> 未找到CSV文件: {csv_filename}，跳过此文件夹。")
                continue

            print(f"  -> 正在读取文件: {csv_filename}")

            # 从文件名提取站点名称 (e.g., 'AMF_FI-Hyy_...' -> 'FI-Hyy')
            try:
                sitename = csv_filename.split('_')[1]
            except IndexError:
                print(f"\n[跳过文件]: 无法从文件名 '{csv_filename}' 中解析站点名称。")
                continue
            
            # 构建完整的文件路径和输出路径
            output_filename = f"{sitename}_uWUE_output.csv"
            output_filepath = os.path.join(output_path, output_filename)

            start_time = time()
            
            # <<< 修改开始 >>>
            # 1. 使用 build_dataset_modified 加载数据, 并添加错误处理
            try:
                ec = build_dataset_modified(csv_filepath)
                print("  -> 数据加载完成。")
            except Exception as e:
                print(f"  -> 错误: 加载文件 '{csv_filename}' 时出错。")
                print(f"  -> 详细信息: {e}")
                print(f"  -> 跳过此文件。")
                continue # 跳过当前循环，处理下一个文件夹
            # <<< 修改结束 >>>

            # 2. 设置时间步长参数 (固定为半小时)
            hourlyMask = xr.DataArray(np.ones(ec.time.shape).astype(bool), coords=[ec.time], dims=['time'])
            nStepsPerDay = 48

            # 3. 计算 ET
            ec['ET'] = (bigleaf.LE_to_ET(ec.LE, ec.TA) * 60 * 60 * (24 / nStepsPerDay))
            ec['ET'] = ec['ET'].assign_attrs(long_name='evapotranspiration', units='mm per timestep')

            # 4. 填充缺失的 NETRAD
            ec['NETRAD'][np.isnan(ec['NETRAD'])] = ec['LE'][np.isnan(ec['NETRAD'])] + ec['H'][np.isnan(ec['NETRAD'])] + ec['G'][np.isnan(ec['NETRAD'])]
            print("  -> ET 计算和 NETRAD 填充完成。")

            # 5. 计算 PET (潜在蒸散)
            PET, _ = bigleaf.PET(ec.TA, ec.PA, ec.NETRAD, G=ec.G, S=None, alpha=1.26,
                                 missing_G_as_NA=False, missing_S_as_NA=False)
            ec['PET'] = PET * 60 * 60 * (24 / nStepsPerDay)
            print("  -> PET 计算完成。")

            # 6. 计算 Zhou 分解所需的掩码
            uWUEa_Mask, uWUEp_Mask = zhou.zhouFlags(ec, nStepsPerDay, hourlyMask, GPPvariant='GPP_NT')
            print("  -> Zhou 掩码计算完成。")

            # 7. 准备用于存储每日结果的 Dataset
            ds_zhou = ec[['ET']].resample(time='D').sum(skipna=False)
            ds_zhou['ET'] = ds_zhou['ET'].assign_attrs(long_name='evapotranspiration', units='mm d-1')
            ds_zhou['zhou_T'] = ds_zhou['ET'] * np.nan
            ds_zhou['zhou_T'] = ds_zhou['zhou_T'].assign_attrs(
                long_name='uWUE estimated transpiration with uWUEa calculate for each day, using GPP_NT',
                units='mm d-1')
            ds_zhou['zhou_T_8day'] = ds_zhou['ET'] * np.nan
            ds_zhou['zhou_T_8day'] = ds_zhou['zhou_T_8day'].assign_attrs(
                long_name='uWUE estimated transpiration with uWUEa calculate for an 8 day moving window (centered), using GPP_NT',
                units='mm d-1')

            # 8. 按年份循环，执行 Zhou 分解
            print("  -> 开始按年份执行 Zhou 分解...")
            ET_vals = ec.ET.values
            GxV_vals = (ec.GPP_NT * np.sqrt(ec.VPD)).values

            for year in np.unique(ec['time.year']):
                yearMask = (ec['time.year'] == year).values
                uWUEp, zhou_T, zhou_T_8day = zhou.zhou_part(ET_vals[yearMask], GxV_vals[yearMask],
                                                           uWUEa_Mask[yearMask], uWUEp_Mask[yearMask],
                                                           nStepsPerDay, hourlyMask[yearMask],
                                                           rho=95 / 100)
                ds_zhou['zhou_T'][ds_zhou['time.year'] == year] = zhou_T
                ds_zhou['zhou_T_8day'][ds_zhou['time.year'] == year] = zhou_T_8day
                print(f"    - {year} 年完成, uWUEp = {uWUEp:.4f}")

            # 9. 保存结果到 CSV 文件
            ds_zhou.to_dataframe().to_csv(output_filepath)
            
            total_time = time() - start_time
            print(f"  -> ✅ 成功! 结果已保存至: {output_filename}")
            print(f"  -> 总耗时: {total_time:.2f} 秒。")

    print("\n所有符合条件的文件处理完毕。")