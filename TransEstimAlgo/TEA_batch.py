# -*- coding: utf-8 -*-
"""
@Project:   AmeriFLUX Data Batch Processing for TEA Model
@File:      run_tea_batch.py
@Author:    Changming Li & Gemini (AI assistant)
@Date:      2025-07-12
@Version:   1.1

Description:
This script automates the process of running the Transpiration-Evaporation-WUE (TEA) 
partitioning model on a collection of AmeriFLUX datasets.

It performs the following steps:
1. Scans a specified base directory for AmeriFLUX site folders that match a specific naming pattern.
2. For each site folder, it locates the corresponding half-hourly (HH) fullset CSV data file.
3. It reads the data, validates the presence of required columns, and performs necessary pre-processing,
   including column renaming and unit conversion.
4. It calls the `simplePartition` function from the custom 'TEA' library to calculate
   Transpiration (T), Evaporation (E), and Water Use Efficiency (WUE).
5. It saves the results for each site into a separate CSV file in a designated output directory.
"""

import os
import pandas as pd
import numpy as np
import re
from time import time
# 导入自定义的TEA库，包含了数据预处理和核心分区算法
from TEA.PreProc import build_dataset, preprocess
from TEA.TEA import partition, simplePartition

# --- 1. 配置区 (Configuration Section) ---
# 在此设置所有路径和常量，方便未来修改
# ------------------------------------------------

# 设置你的数据源根目录，存放所有AMF_SITENAME...格式的文件夹
BASE_PATH = r'Z:\Observation\AmeriFLUX' 
# BASE_PATH = r'Z:\Observation\FLUXNET.4.0\FLUXNET2015-Tier2'
# 设置统一的输出目录，用于存放所有站点的计算结果
OUTPUT_PATH = r'Z:\LCM\ET_T_Partition\TransEstimAlgo\AmeriFLUX_TEA_Output'
# OUTPUT_PATH = r'Z:\LCM\ET_T_Partition\TransEstimAlgo\FLUXNET_TEA_Output'

# 用于匹配文件夹名称的正则表达式
# 格式为: AMF_SITENAME_FLUXNET_FULLSET_YYYY-YYYY_X-X
# re.compile预编译正则表达式，可以提高循环中的匹配效率
FOLDER_PATTERN = re.compile(r'^AMF_.*_FLUXNET_FULLSET_\d{4}-\d{4}_\d+-\d+$')
# FOLDER_PATTERN = re.compile(r'^FLX_.*_FLUXNET2015_FULLSET_\d{4}-\d{4}_\d+-\d+$')


def process_site_folder(folder_path, output_path):
    """
    处理单个站点文件夹，执行数据读取、预处理、TEA计算和结果保存。

    Args:
        folder_path (str): 单个站点文件夹的完整路径。
        output_path (str): 保存结果的输出目录路径。
    """
    folder_name = os.path.basename(folder_path)
    print(f"\n[处理文件夹]: {folder_name}")

    # 3.1. 根据文件夹名构建CSV文件的名称
    # AmeriFLUX标准文件名通常将 'FULLSET' 替换为 'FULLSET_HH' 代表半小时数据
    # csv_filename = folder_name.replace('_FLUXNET2015_FULLSET_', '_FLUXNET2015_FULLSET_HH_') + '.csv'

    csv_filename = folder_name.replace('_FLUXNET_FULLSET_', '_FLUXNET_FULLSET_HH_') + '.csv'
    csv_filepath = os.path.join(folder_path, csv_filename)

    # 3.2. 检查对应的CSV文件是否存在，确保数据源可用
    if not os.path.exists(csv_filepath):
        print(f"  -> 未找到CSV文件: {csv_filename}，跳过此文件夹。")
        return  # 退出当前函数，处理下一个文件夹

    print(f"  -> 正在读取文件: {csv_filename}")
    
    try:
        # 3.3. 使用pandas读取CSV文件。对于可能存在的编码或解析问题，可在此处添加error_handling
        df = pd.read_csv(csv_filepath, on_bad_lines='skip') # 'on_bad_lines'可以跳过格式错误的行
    except Exception as e:
        print(f"  -> 读取CSV文件时发生错误: {e}，跳过此文件夹。")
        return

    # 3.4. 定义需要提取和重命名的列的映射关系
    # Key: 原始文件中列名 (Source)
    # Value: TEA模型需要的列名 (Destination)
    column_mapping = {
        'LE_F_MDS': 'ET',            # 潜热通量 (W m-2) -> 蒸散
        'GPP_NT_VUT_REF': 'GPP',     # 总初级生产力 (umol CO2 m-2 s-1)
        'TA_F_MDS': 'Tair',          # 空气温度 (°C)
        'RH': 'RH',                  # 相对湿度 (%)
        'VPD_F_MDS': 'VPD',          # 饱和水汽压差 (hPa)
        'P_ERA': 'precip',           # 降水 (mm)
        'SW_IN_F': 'Rg',             # 向下短波辐射 (W m-2)
        'WS': 'u',                   # 风速 (m s-1)
        'SW_IN_POT': 'Rg_pot'        # 潜在向下短波辐射 (W m-2)
    }

    # 检查所需列是否都存在于文件中
    original_columns = list(column_mapping.keys())
    missing_cols = [col for col in original_columns if col not in df.columns]
    if missing_cols:
        print(f"  -> 警告: 文件中缺少必需列 {missing_cols}，无法进行TEA计算，将跳过此文件。")
        return

    # 3.5. 数据预处理
    # a. 仅提取指定的列，并使用 .copy() 避免SettingWithCopyWarning
    processed_df = df[original_columns].copy()
    # b. 重命名列以匹配TEA函数输入
    processed_df.rename(columns=column_mapping, inplace=True)
    
    # c. 对 ET 列进行单位转换
    # 原始单位: W m-2 (瓦/平方米, 能量通量)
    # 目标单位: mm/30min (毫米/半小时, 水量)
    # 转换因子 ≈ 1 / (λ * ρ_w) * 1800 s/30min
    # 其中 λ 是水的汽化潜热 (~2.45 MJ/kg), ρ_w 是水的密度 (~1000 kg/m^3)
    # 0.0007348 是一个近似转换系数，将 W/m^2 直接转换为 mm/30min
    processed_df['ET'] = processed_df['ET'] * 0.0007348
    
    # d. 生成新的 timestamp 列, 并移动到第一列
    # AmeriFLUX 半小时数据，创建一个从0开始，步长为30分钟的序列作为时间戳
    num_rows = len(processed_df) 
    processed_df['timestamp'] = range(0, num_rows * 30, 30)
    # 调整列顺序，将 'timestamp' 放在第一位
    processed_df = processed_df[['timestamp'] + [col for col in processed_df.columns if col != 'timestamp']]

    print("  -> 数据预处理完成，准备调用 TEA 函数...")

    # 3.6. 准备 TEA 函数的输入变量
    # 将DataFrame的列转换为NumPy数组，这是许多科学计算函数的标准输入格式
    timestamp = processed_df['timestamp'].values
    ET = processed_df['ET'].values
    GPP = processed_df['GPP'].values
    RH = processed_df['RH'].values
    Rg = processed_df['Rg'].values
    Rg_pot = processed_df['Rg_pot'].values
    Tair = processed_df['Tair'].values
    VPD = processed_df['VPD'].values
    precip = processed_df['precip'].values
    u = processed_df['u'].values

    # 3.7. *** 调用核心算法 ***
    # 使用 `simplePartition` 函数进行蒸散分区计算
    TEA_T, TEA_E, TEA_WUE = simplePartition(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u)
    
    # 3.8. 保存结果
    # a. 从文件夹名称中提取 SITENAME (例如 'AMF_AR-CCg_...' -> 'AR-CCg')
    sitename = folder_name.split('_')[1]
    
    # b. 构建新的输出文件名
    output_filename = f"{sitename}_TEA_results.csv"
    output_filepath = os.path.join(output_path, output_filename)

    # c. 将计算结果整合到新的DataFrame中
    results_df = pd.DataFrame({
        'timestamp': timestamp, # 保留时间戳以便与输入对齐
        'TEA_T': TEA_T,         # 计算出的蒸腾 (Transpiration)
        'TEA_E': TEA_E,         # 计算出的蒸发 (Evaporation)
        'TEA_WUE': TEA_WUE      # 计算出的水分利用效率 (Water Use Efficiency)
    })
    
    # d. 将结果写入CSV文件，不包含pandas的行索引
    results_df.to_csv(output_filepath, index=False)
    print(f"  -> 结果已成功保存至: {output_filepath}")


def main():
    """
    主函数，执行整个批处理流程。
    """
    print("--- TEA 模型批量处理程序启动 ---")
    print(f"开始扫描并处理目录: {BASE_PATH}")
    print(f"输出结果将保存至: {OUTPUT_PATH}")
    print(f"将使用 'TEA.TEA.simplePartition' 函数计算 T, E, 和 WUE。")

    # --- 2. 准备工作 (Setup) ---
    # --------------------------------

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 检查根目录是否存在
    if not os.path.exists(BASE_PATH):
        print(f"错误：找不到数据源路径 '{BASE_PATH}'。请检查路径配置或磁盘连接。")
        return # 严重错误，直接退出程序

    # --- 3. 主循环 (Main Loop) ---
    # -------------------------------
    all_entries = os.listdir(BASE_PATH)
    
    # 遍历根目录下的所有条目
    for entry_name in all_entries:
        folder_path = os.path.join(BASE_PATH, entry_name)

        # 检查当前条目是否是文件夹，并且名称是否符合指定格式
        if os.path.isdir(folder_path) and FOLDER_PATTERN.match(entry_name):
            # 调用函数处理该文件夹
            process_site_folder(folder_path, OUTPUT_PATH)

    print("\n--- 所有符合条件的文件夹处理完毕 ---")


# --- 程序入口 ---
# 只有当该脚本被直接执行时，才运行main()函数
# 如果该脚本被其他模块导入，则不会执行
if __name__ == "__main__":
    main()