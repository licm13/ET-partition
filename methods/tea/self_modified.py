import os
import pandas as pd
import numpy as np
import re
from time import time
from TEA.PreProc import build_dataset
from TEA.PreProc import preprocess
from TEA.TEA import partition
from TEA.TEA import simplePartition
from time import time

# --- 配置区 ---
# 设置你的数据源根目录
base_path = 'Z:\Observation\AmeriFLUX' 
# 设置统一的输出目录
output_path = 'Z:\Observation\AmeriFLUX\TEA_Output'
# --- 配置结束 ---


print(f"开始扫描并处理目录: {base_path}")
print(f"输出结果将保存至: {output_path}")
print(f"将使用 'TEA.TEA.simplePartition' 函数计算 T, E, 和 WUE。")

# 确保输出目录存在，如果不存在则创建
os.makedirs(output_path, exist_ok=True)

# 用于匹配文件夹名称的正则表达式
# 格式为: AMF_SITENAME_FLUXNET_FULLSET_YYYY-YYYY_X-X
folder_pattern = re.compile(r'^AMF_.*_FLUXNET_FULLSET_\d{4}-\d{4}_\d+-\d+$')

# 检查根目录是否存在
if not os.path.exists(base_path):
    print(f"错误：找不到路径 '{base_path}'。请确保该磁盘或路径存在。")
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
            csv_filename = folder_name.replace('_FLUXNET_FULLSET_', '_FLUXNET_FULLSET_HH_') + '.csv'
            csv_filepath = os.path.join(folder_path, csv_filename)

            # 2. 检查对应的CSV文件是否存在
            if not os.path.exists(csv_filepath):
                print(f"  -> 未找到CSV文件: {csv_filename}，跳过此文件夹。")
                continue

            print(f"  -> 正在读取文件: {csv_filename}")
            
            # 3. 使用pandas读取CSV文件
            df = pd.read_csv(csv_filepath)

            # 4. 定义需要提取和重命名的列的映射关系
            column_mapping = {
                'LE_F_MDS': 'ET',
                'GPP_NT_VUT_REF': 'GPP',
                'TA_F_MDS': 'Tair',
                'RH': 'RH',
                'VPD_F_MDS': 'VPD',
                'P_ERA': 'precip',
                'SW_IN_F': 'Rg',
                'WS': 'u',  
                'SW_IN_POT': 'Rg_pot' 
            }

            # 获取所有需要用到的原始列名
            original_columns = list(column_mapping.keys())

            # 检查所需列是否都存在于文件中
            missing_cols = [col for col in original_columns if col not in df.columns]
            if missing_cols:
                print(f"  -> 警告: 文件中缺少必需列 {missing_cols}，无法进行TEA计算，将跳过此文件。")
                continue
            
            # 5. 提取指定的列，并重命名
            processed_df = df[original_columns].copy()
            processed_df.rename(columns=column_mapping, inplace=True)

            # 6. 对 ET 列进行单位转换
            processed_df['ET'] = processed_df['ET'] * 0.0007348

            # 7. 生成新的 timestamp 列, 并移动到第一列
            num_rows = len(processed_df) 
            processed_df['timestamp'] = range(0, num_rows * 30, 30)
            processed_df = processed_df[['timestamp'] + [col for col in processed_df.columns if col != 'timestamp']]

            print("  -> 数据预处理完成，准备调用 TEA 函数...")

            # 8. 准备 TEA 函数的输入变量
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

            TEA_T,TEA_E,TEA_WUE = simplePartition(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u)
            
            # ds = build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u)
            # ds = preprocess(ds)
            # start = time()

            # RFmod_vars=['Rg','Tair','RH','u','Rg_pot_daily',
            #         'Rgpotgrad','year','GPPgrad','DWCI','C_Rg_ET','CSWI']

            # RandomForestRegressor_kwargs={
            #     'n_estimators'  : 100,
            #     'oob_score'     : True,
            #     'max_features'  : "n/3",
            #     'verbose'       : 0,
            #     'warm_start'    : False,
            #     'n_jobs'        : 3}

            # ds = partition(ds,
            #     percs = np.linspace(50,100,11),
            #     CSWIlims = np.array([-0.5]),
            #     RFmod_vars = RFmod_vars,
            #     RandomForestRegressor_kwargs = RandomForestRegressor_kwargs       
            #     )
            # print('{0:0.3} minutes to process'.format((time()-start)/60))
            
            # 9. 将输入数据和计算结果合并并保存 (*** 此处已按要求修改 ***)
            # 从文件夹名称中提取 SITENAME (例如 'AMF_AR-CCg_...' -> 'AR-CCg')
            sitename = folder_name.split('_')[1]
            
            # 构建新的输出文件名
            output_filename = f"{sitename}_TEA_results.csv"
            
            # 构建完整的输出文件路径
            output_filepath = os.path.join(output_path, output_filename)

            results_df = pd.DataFrame({
                    'timestamp': timestamp,
                    'TEA_T': TEA_T,
                    'TEA_E': TEA_E,
                    'TEA_WUE': TEA_WUE
                })
            
            results_df.to_csv(output_filepath, index=False)

    print("\n所有文件夹处理完毕。")