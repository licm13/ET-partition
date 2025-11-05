"""Batch processing toolkit for the uWUE evapotranspiration partition method."""

from __future__ import annotations

import argparse
import logging
import sys
import re
from datetime import datetime
from pathlib import Path
from time import time
from typing import Iterable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter

from . import bigleaf, zhou
from .preprocess import build_dataset_modified

# 设置绘图样式
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

DEFAULT_PATTERN = re.compile(
    r"^(?:AMF|FLX)_.*_FLUXNET(?:2015)?_FULLSET_\d{4}-\d{4}_\d+-\d+$"
)

class uWUEBatchProcessor:
    """
    uWUE 批处理器类
    
    主要功能:
    1. 批量处理 FLUXNET 数据文件
    2. 执行 Zhou uWUE 分解
    3. 生成可视化结果
    4. 导出处理结果
    """
    
    def __init__(
        self,
        base_path: Path,
        output_path: Path,
        create_plots: bool = True,
        folder_pattern: re.Pattern[str] = DEFAULT_PATTERN,
    ) -> None:
        """
        初始化处理器
        
        参数:
        - base_path: 数据源根目录
        - output_path: 输出目录
        - create_plots: 是否生成可视化图表
        """
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.create_plots = create_plots
        self.folder_pattern = folder_pattern

        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 处理结果统计
        self.processing_stats = {
            'total_folders': 0,
            'processed_successfully': 0,
            'failed_processing': 0,
            'processing_times': [],
            'sites_processed': []
        }
    
    def _setup_logging(self):
        """设置日志记录"""
        log_file = self.output_path / f'uwue_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def print_header(self):
        """打印程序头部信息"""
        header = """
╔════════════════════════════════════════════════════════════════════════════════════╗
║                            uWUE 批处理分析工具                                    ║
║                                                                                    ║
║  作者: LCM                                                                         ║
║  日期: 2025-07-11                                                                  ║
║  协助: Gemini & Claude AI                                                          ║
║                                                                                    ║
║  功能: 批量处理 FLUXNET 数据，执行 Zhou uWUE 分解方法                             ║
║        计算蒸散发中的蒸腾部分，并生成可视化结果                                   ║
╚════════════════════════════════════════════════════════════════════════════════════╝
        """
        print(header)
        self.logger.info("uWUE 批处理程序启动")
        self.logger.info(f"数据源路径: {self.base_path}")
        self.logger.info(f"输出路径: {self.output_path}")
    
    def scan_directories(self):
        """扫描并识别符合条件的数据文件夹"""
        if not self.base_path.exists():
            self.logger.error(f"数据源路径不存在: {self.base_path}")
            return []
        
        all_entries = list(self.base_path.iterdir())
        valid_folders = []
        
        for folder_path in all_entries:
            if folder_path.is_dir() and self.folder_pattern.match(folder_path.name):
                valid_folders.append(folder_path)
        
        self.processing_stats['total_folders'] = len(valid_folders)
        self.logger.info(f"发现 {len(valid_folders)} 个符合条件的文件夹")
        
        return valid_folders
    
    def process_single_site(self, folder_path):
        """
        处理单个站点的数据
        
        参数:
        - folder_path: 站点文件夹路径
        
        返回:
        - success: 是否处理成功
        - sitename: 站点名称
        - processing_time: 处理时间
        - results: 处理结果 (xarray Dataset)
        """
        folder_name = folder_path.name
        start_time = time()
        
        try:
            # 1. 构建 CSV 文件名
            # csv_filename = folder_name.replace('_FLUXNET_FULLSET_', '_FLUXNET_FULLSET_HH_') + '.csv' # AmeriFLUX的命名规则
            csv_filename = folder_name.replace('_FLUXNET2015_FULLSET_', '_FLUXNET2015_FULLSET_HH_') + '.csv'  # FLUXNET/测试集的命名规则
            csv_filepath = folder_path / csv_filename
            
            if not csv_filepath.exists():
                self.logger.warning(f"CSV 文件不存在: {csv_filename}")
                return False, None, 0, None
            
            # 2. 提取站点名称
            try:
                sitename = csv_filename.split('_')[1]
            except IndexError:
                self.logger.error(f"无法从文件名解析站点名称: {csv_filename}")
                return False, None, 0, None
            
            self.logger.info(f"🔄 开始处理站点: {sitename}")
            
            # 3. 加载数据
            try:
                ec = build_dataset_modified(str(csv_filepath))
                self.logger.info(f"  ✅ 数据加载完成，数据点数: {len(ec.time)}")
            except Exception as e:
                self.logger.error(f"  ❌ 数据加载失败: {e}")
                return False, sitename, time() - start_time, None
            
            # 4. 数据预处理和计算
            results = self._perform_uwue_analysis(ec, sitename)
            
            # 5. 保存结果
            self._save_results(results, sitename)
            
            # 6. 生成可视化图表
            if self.create_plots:
                self._create_visualization(results, sitename)
            
            processing_time = time() - start_time
            self.processing_stats['processed_successfully'] += 1
            self.processing_stats['processing_times'].append(processing_time)
            self.processing_stats['sites_processed'].append(sitename)
            
            self.logger.info(f"  ✅ 站点 {sitename} 处理完成，耗时: {processing_time:.2f} 秒")
            
            return True, sitename, processing_time, results
            
        except Exception as e:
            self.processing_stats['failed_processing'] += 1
            processing_time = time() - start_time
            self.logger.error(f"  ❌ 处理站点失败: {e}")
            return False, folder_name, processing_time, None
    
    def _perform_uwue_analysis(self, ec, sitename):
        """执行 uWUE 分析"""
        self.logger.info(f"  🔬 开始 uWUE 分析...")
        
        # 设置时间步长参数 (半小时数据)
        hourlyMask = xr.DataArray(
            np.ones(ec.time.shape).astype(bool), 
            coords=[ec.time], 
            dims=['time']
        )
        nStepsPerDay = 48  # 每天48个半小时
        
        # 计算蒸散发 (ET)
        ec['ET'] = (bigleaf.LE_to_ET(ec.LE, ec.TA) * 60 * 60 * (24 / nStepsPerDay))
        ec['ET'] = ec['ET'].assign_attrs(
            long_name='evapotranspiration', 
            units='mm per timestep'
        )
        
        # 填充缺失的净辐射数据
        missing_netrad = np.isnan(ec['NETRAD'])
        ec['NETRAD'][missing_netrad] = (
            ec['LE'][missing_netrad] + 
            ec['H'][missing_netrad] + 
            ec['G'][missing_netrad]
        )
        
        self.logger.info(f"  📊 填充了 {missing_netrad.sum().values} 个缺失的净辐射数据点")
        
        # 计算潜在蒸散发 (PET)
        PET, _ = bigleaf.PET(
            ec.TA, ec.PA, ec.NETRAD, 
            G=ec.G, S=None, alpha=1.26,
            missing_G_as_NA=False, 
            missing_S_as_NA=False
        )
        ec['PET'] = PET * 60 * 60 * (24 / nStepsPerDay)
        
        # Calculate masks needed for Zhou partitioning
        actual_wue_mask, potential_wue_mask = zhou.build_zhou_masks(
            ec, nStepsPerDay, hourlyMask, gpp_variable='GPP_NT'
        )
        
        self.logger.info(f"  🎯 Valid data masks: actual={actual_wue_mask.sum()}, potential={potential_wue_mask.sum()}")
        
        # Prepare daily aggregated dataset
        ds_zhou = ec[['ET']].resample(time='D').sum(skipna=False)
        ds_zhou['ET'] = ds_zhou['ET'].assign_attrs(
            long_name='evapotranspiration', 
            units='mm d-1'
        )
        
        # Initialize transpiration variables
        ds_zhou['zhou_T'] = ds_zhou['ET'] * np.nan
        ds_zhou['zhou_T'] = ds_zhou['zhou_T'].assign_attrs(
            long_name='uWUE estimated transpiration (daily uWUEa)',
            units='mm d-1'
        )
        
        ds_zhou['zhou_T_8day'] = ds_zhou['ET'] * np.nan
        ds_zhou['zhou_T_8day'] = ds_zhou['zhou_T_8day'].assign_attrs(
            long_name='uWUE estimated transpiration (8-day moving window)',
            units='mm d-1'
        )
        
        # Execute Zhou partitioning by year
        self.logger.info(f"  🔄 Starting Zhou partitioning by year...")
        et_values = ec.ET.values
        gpp_vpd_values = (ec.GPP_NT * np.sqrt(ec.VPD)).values
        
        years = np.unique(ec['time.year'])
        uwue_values = {}
        
        for year in years:
            year_mask = (ec['time.year'] == year).values
            potential_wue, daily_transp, transp_8day = zhou.zhou_part(
                et_values[year_mask], gpp_vpd_values[year_mask],
                actual_wue_mask[year_mask], potential_wue_mask[year_mask],
                nStepsPerDay, hourlyMask[year_mask],
                percentile=0.95
            )
            
            ds_zhou['zhou_T'][ds_zhou['time.year'] == year] = daily_transp
            ds_zhou['zhou_T_8day'][ds_zhou['time.year'] == year] = transp_8day
            uwue_values[year] = potential_wue
            
            self.logger.info(f"    - {year} 年: uWUEp = {uWUEp:.4f}")
        
        # 添加站点和处理信息
        ds_zhou.attrs['sitename'] = sitename
        ds_zhou.attrs['processing_date'] = datetime.now().isoformat()
        ds_zhou.attrs['uwue_values'] = str(uwue_values)
        
        return ds_zhou
    
    def _save_results(self, results, sitename):
        """保存处理结果"""
        output_filename = f"{sitename}_uWUE_output.csv"
        output_filepath = self.output_path / output_filename
        
        # 保存为 CSV
        results.to_dataframe().to_csv(output_filepath)
        
        # 保存为 NetCDF (更适合科学数据)
        netcdf_filepath = self.output_path / f"{sitename}_uWUE_output.nc"
        results.to_netcdf(netcdf_filepath)
        
        self.logger.info(f"  💾 结果已保存: {output_filename}")
    
    def _create_visualization(self, results, sitename):
        """创建可视化图表"""
        try:
            # 创建图表目录
            plots_dir = self.output_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # 转换为 DataFrame 以便绘图
            df = results.to_dataframe().reset_index()
            df = df.dropna()  # 移除 NaN 值
            
            if len(df) == 0:
                self.logger.warning(f"  ⚠️ 站点 {sitename} 无有效数据，跳过绘图")
                return
            
            # 创建多子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'uWUE 分析结果 - {sitename}', fontsize=16, fontweight='bold')
            
            # 1. 蒸散发时间序列
            ax1 = axes[0, 0]
            ax1.plot(df['time'], df['ET'], 'b-', alpha=0.7, label='总蒸散发 (ET)')
            ax1.plot(df['time'], df['zhou_T'], 'r-', alpha=0.8, label='蒸腾 (T)')
            ax1.set_title('蒸散发与蒸腾时间序列')
            ax1.set_ylabel('蒸散发 (mm d⁻¹)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 蒸腾比例
            ax2 = axes[0, 1]
            df['T_ET_ratio'] = df['zhou_T'] / df['ET']
            ax2.plot(df['time'], df['T_ET_ratio'], 'g-', alpha=0.7)
            ax2.set_title('蒸腾比例 (T/ET)')
            ax2.set_ylabel('蒸腾比例')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            
            # 3. 散点图: ET vs T
            ax3 = axes[1, 0]
            scatter = ax3.scatter(df['ET'], df['zhou_T'], c=df.index, cmap='viridis', alpha=0.6)
            ax3.plot([0, df['ET'].max()], [0, df['ET'].max()], 'k--', alpha=0.5, label='1:1 线')
            ax3.set_xlabel('总蒸散发 (mm d⁻¹)')
            ax3.set_ylabel('蒸腾 (mm d⁻¹)')
            ax3.set_title('蒸散发 vs 蒸腾')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 月平均值
            ax4 = axes[1, 1]
            df['month'] = pd.to_datetime(df['time']).dt.month
            monthly_mean = df.groupby('month')[['ET', 'zhou_T']].mean()
            
            months = monthly_mean.index
            ax4.plot(months, monthly_mean['ET'], 'b-o', label='ET')
            ax4.plot(months, monthly_mean['zhou_T'], 'r-o', label='蒸腾')
            ax4.set_title('月平均蒸散发')
            ax4.set_xlabel('月份')
            ax4.set_ylabel('蒸散发 (mm d⁻¹)')
            ax4.set_xticks(range(1, 13))
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 格式化时间轴
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(DateFormatter('%Y'))
                ax.xaxis.set_major_locator(mdates.YearLocator())
            
            plt.tight_layout()
            
            # 保存图表
            plot_filename = plots_dir / f"{sitename}_uWUE_analysis.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"  📈 可视化图表已保存: {plot_filename.name}")
            
        except Exception as e:
            self.logger.error(f"  ❌ 创建可视化图表失败: {e}")
    
    def generate_summary_report(self):
        """生成处理总结报告"""
        self.logger.info("📋 生成处理总结报告...")
        
        # 计算统计信息
        total_time = sum(self.processing_stats['processing_times'])
        avg_time = np.mean(self.processing_stats['processing_times']) if self.processing_stats['processing_times'] else 0
        
        # 创建总结报告
        report = f"""
╔════════════════════════════════════════════════════════════════════════════════════╗
║                                处理总结报告                                        ║
╠════════════════════════════════════════════════════════════════════════════════════╣
║ 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                            ║
║ 总文件夹数: {self.processing_stats['total_folders']}                               ║
║ 成功处理: {self.processing_stats['processed_successfully']}                        ║
║ 处理失败: {self.processing_stats['failed_processing']}                             ║
║ 成功率: {self.processing_stats['processed_successfully']/max(self.processing_stats['total_folders'],1)*100:.1f}%  ║
║ 总耗时: {total_time:.2f} 秒                                                        ║
║ 平均耗时: {avg_time:.2f} 秒/站点                                                   ║
╚════════════════════════════════════════════════════════════════════════════════════╝

成功处理的站点:
{chr(10).join([f"  • {site}" for site in self.processing_stats['sites_processed']])}
        """
        
        print(report)
        
        # 保存报告到文件
        report_file = self.output_path / 'processing_summary.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"📄 总结报告已保存: {report_file}")
    
    def run(self, parallel: bool = False, workers: Optional[int] = None):
        """
        运行批处理程序

        参数:
        - parallel: 是否使用并行处理
        - workers: 并行工作进程数 (None = 使用CPU核心数)
        """
        self.print_header()

        # 扫描目录
        valid_folders = self.scan_directories()

        if not valid_folders:
            self.logger.warning("未找到符合条件的文件夹，程序结束")
            return

        if parallel:
            self._run_parallel(valid_folders, workers)
        else:
            self._run_serial(valid_folders)

        # 生成总结报告
        self.generate_summary_report()

        self.logger.info("\n🎉 所有处理完成!")

    def _run_serial(self, valid_folders):
        """串行处理所有站点"""
        # 处理每个站点
        for i, folder_path in enumerate(valid_folders, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"处理进度: {i}/{len(valid_folders)} ({i/len(valid_folders)*100:.1f}%)")
            self.logger.info(f"当前文件夹: {folder_path.name}")

            success, sitename, proc_time, results = self.process_single_site(folder_path)

            if success:
                self.logger.info(f"✅ 成功处理站点 {sitename}")
            else:
                self.logger.error(f"❌ 处理失败: {folder_path.name}")

    def _run_parallel(self, valid_folders, workers: Optional[int] = None):
        """并行处理所有站点"""
        if workers is None:
            workers = multiprocessing.cpu_count()

        self.logger.info(f"\n🚀 使用并行处理模式，工作进程数: {workers}")
        self.logger.info(f"   Parallel processing with {workers} workers")

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_folder = {
                executor.submit(self.process_single_site, folder): folder
                for folder in valid_folders
            }

            # Process completed tasks
            completed = 0
            total = len(valid_folders)

            for future in as_completed(future_to_folder):
                folder_path = future_to_folder[future]
                completed += 1

                try:
                    success, sitename, proc_time, results = future.result()

                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"处理进度: {completed}/{total} ({completed/total*100:.1f}%)")

                    if success:
                        self.logger.info(f"✅ 成功处理站点 {sitename} (耗时: {proc_time:.2f}秒)")
                    else:
                        self.logger.error(f"❌ 处理失败: {folder_path.name}")

                except Exception as e:
                    self.logger.error(f"❌ 处理站点时发生异常 {folder_path.name}: {e}")
                    self.processing_stats['failed_processing'] += 1


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Command-line interface for the uWUE batch workflow."""

    repo_root = Path(__file__).resolve().parents[2]
    default_base = repo_root / "data" / "test_site"
    default_output = repo_root / "outputs" / "uwue"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-path",
        type=Path,
        default=default_base,
        help="Directory containing Fluxnet/AmeriFlux style site folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output,
        help="Directory where uWUE results will be stored.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of diagnostic plots.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_PATTERN.pattern,
        help="Regular expression used to match site folder names.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing of sites.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores).",
    )

    args = parser.parse_args(args=list(argv) if argv is not None else None)

    processor = uWUEBatchProcessor(
        base_path=args.base_path,
        output_path=args.output_path,
        create_plots=not args.no_plots,
        folder_pattern=re.compile(args.pattern),
    )
    processor.run(parallel=args.parallel, workers=args.workers)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()