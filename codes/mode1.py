import os
import matplotlib
# Use non-interactive backend to avoid display issues
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from scipy import stats
import cdsapi
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple, Optional, Dict, Any, Union, List
import numpy.typing as npt
from matplotlib.figure import Figure
from numpy import floating, ndarray

# Use built-in matplotlib fonts only
plt.rcParams["font.family"] = "DejaVu Sans, sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # Ensure minus signs display correctly

# 创建数据和结果目录
data_dir = Path("data")
results_dir = Path("results")
figures_dir = Path("figures")

data_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# 定义研究区域和时间范围
LAT_MIN, LAT_MAX = 15, 50
LON_MIN, LON_MAX = 70, 130
START_YEAR, END_YEAR = 1950, 2023

# 定义季节
seasons = {
    "spring": [3, 4, 5],  # 春季：3-5月
    "summer": [6, 7, 8],  # 夏季：6-8月
    "autumn": [9, 10, 11],  # 秋季：9-11月
    "winter": [12, 1, 2]  # 冬季：12-2月（跨年）
}

import time

# 下载ERA5温度数据
def download_era5_data(year, max_retries=3, delay=60):
    """下载指定年份的ERA5逐日2米温度数据，并带有重试机制"""
    output_file = data_dir / f"era5_t2m_{year}.nc"
    
    if output_file.exists():
        print(f"{year}年数据已存在，跳过下载")
        return output_file
    
    print(f"正在下载{year}年数据...")
    c = cdsapi.Client()
    
    for attempt in range(max_retries):
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ['2m_temperature'],
                    'year': str(year),
                    'month': ["01", "02", "03", "04", "05", "06", 
                              "07", "08", "09", "10", "11", "12"],
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': ["00:00", "03:00", "06:00", "09:00",
                            "12:00", "15:00", "18:00", "21:00"],  # 每天8个时间步
                    'area': [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # 北、西、南、东
                    'format': 'netcdf',
                },
                str(output_file)
            )
            print(f"{year}年数据下载完成")
            return output_file
        except Exception as e:
            print(f"下载{year}年数据时出错 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"将在 {delay} 秒后重试...")
                time.sleep(delay)
            else:
                print(f"已达到最大重试次数，下载{year}年数据失败")
                return None

# 计算积温和热浪事件
def calculate_temperature_metrics(
    temperatures: Union[xr.DataArray, npt.NDArray[floating], List[float]],
    base_temp: float = 10.0,
    heatwave_threshold: float = 35.0
) -> Tuple[float, float, float, int, int]:
    """
    Calculate active accumulated temperature, effective accumulated temperature, 
    mean temperature, and heatwave events
    
    Parameters:
    temperatures: Temperature sequence (K)
    base_temp: Biological base temperature (°C)
    heatwave_threshold: Heatwave threshold (°C)
    
    Returns:
    Tuple[active accumulated temperature, effective accumulated temperature, 
          mean temperature, heatwave days, heatwave events] 
    """
    # Convert to numpy array of float64
    if isinstance(temperatures, xr.DataArray):
        temp_values = temperatures.values.astype(float)
    else:
        temp_values = np.array(temperatures, dtype=float)
    
    temperatures_c = temp_values - 273.15  # K -> °C
    
    # Ensure base_temp is float
    base_temp_float = float(base_temp)
    heatwave_threshold_float = float(heatwave_threshold)
    
    # Calculate active accumulated temperature
    active_mask = temperatures_c >= base_temp_float
    active_accumulated = float(np.nansum(temperatures_c[active_mask]))
    
    # Calculate effective accumulated temperature
    effective_accumulated = float(np.nansum(temperatures_c[active_mask] - base_temp_float))
    
    # Calculate mean temperature
    mean_temp = float(np.nanmean(temperatures_c))
    
    # Detect heatwave events (consecutive days above threshold)
    heatwave_mask = temperatures_c > heatwave_threshold_float
    heatwave_days = 0
    heatwave_events = 0
    current_streak = 0
    
    for day in heatwave_mask:
        if day:
            current_streak += 1
        else:
            if current_streak >= 3:
                heatwave_events += 1
                heatwave_days += current_streak
            current_streak = 0
    
    # Check last streak
    if current_streak >= 3:
        heatwave_events += 1
        heatwave_days += current_streak
    
    return active_accumulated, effective_accumulated, mean_temp, heatwave_days, heatwave_events

# 计算趋势率
def calculate_trend(data: Union[xr.DataArray, npt.NDArray[np.float64]]) -> Tuple[float, float]:
    """
    计算时间序列的线性趋势率
    
    参数:
    data: 时间序列数据 (xarray.DataArray 或 numpy数组)
    
    返回:
    Tuple[slope: 趋势率, p_value: 显著性水平]
    """
    try:
        # 确保输入为float64 numpy数组
        if isinstance(data, xr.DataArray):
            data_values = data.values.astype(np.float64)
        else:
            data_values = np.array(data, dtype=np.float64)
        
        # 创建时间序列
        years = np.arange(len(data_values), dtype=np.float64)
        valid_mask = ~np.isnan(data_values)
        
        if np.sum(valid_mask) < 10:  # 至少需要10个有效数据点
            return np.nan, np.nan
        
        # 提取有效数据
        x = years[valid_mask]
        y = data_values[valid_mask]
        
        # 执行线性回归
        result = stats.linregress(x, y)
        
        # 确保结果是浮点数
        if isinstance(result, tuple):
            # 旧版本scipy返回普通元组
            slope = float(result[0]) * 10.0  # 转换为10年变化率
            p_value = float(result[3])
        else:
            # 新版本scipy返回命名元组
            slope = float(result.slope) * 10.0  # 转换为10年变化率
            p_value = float(result.pvalue)
        
        return slope, p_value
    except Exception:
        return np.nan, np.nan

# 处理单个年份的数据
def process_year(year):
    """处理单个年份的数据，计算年尺度积温"""
    file_path = data_dir / f"era5_t2m_{year}.nc"
    
    if not file_path.exists():
        print(f"{year}年数据文件不存在")
        return None
    
    try:
        # 打开数据文件
        with xr.open_dataset(file_path) as ds:
            # 计算日平均温度 (使用valid_time作为时间维度)
            t2m_daily = ds.t2m.resample(valid_time='1D').mean()
            
            # 计算年尺度温度指标
            active_annual, effective_annual, mean_annual, heatwave_days_annual, heatwave_events_annual = xr.apply_ufunc(
                calculate_temperature_metrics,
                t2m_daily,
                input_core_dims=[['valid_time']],
                output_core_dims=[[], [], [], [], []],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float, float, float, int, int]
            )
            active_annual = xr.DataArray(active_annual)
            effective_annual = xr.DataArray(effective_annual)
            mean_annual = xr.DataArray(mean_annual)
            heatwave_days_annual = xr.DataArray(heatwave_days_annual)
            heatwave_events_annual = xr.DataArray(heatwave_events_annual)
            
            # 添加年份坐标
            active_annual = active_annual.assign_coords(year=year)
            effective_annual = effective_annual.assign_coords(year=year)
            mean_annual = mean_annual.assign_coords(year=year)
            
            # 计算季节尺度积温
            seasonal_results = {}
            for season_name, months in seasons.items():
                # 提取季节数据
                if season_name == "winter":
                    # 冬季跨年度，需要特殊处理
                    if year > START_YEAR:
                        prev_year_file = data_dir / f"era5_t2m_{year-1}.nc"
                        if not prev_year_file.exists():
                            continue
                            
                        with xr.open_dataset(prev_year_file) as prev_ds:
                            # 提取前一年12月数据
                            prev_dec = prev_ds.t2m.sel(valid_time=prev_ds.valid_time.dt.month == 12)
                            
                        # 提取当年1-2月数据
                        current_jan_feb = t2m_daily.sel(valid_time=t2m_daily.valid_time.dt.month.isin([1, 2]))
                        
                        # 合并数据
                        winter_data = xr.concat([prev_dec, current_jan_feb], dim='time')
                        
                        # 计算冬季温度指标
                        active_season, effective_season, mean_season, heatwave_days_season, heatwave_events_season = xr.apply_ufunc(
                            calculate_temperature_metrics,
                            winter_data,
                            input_core_dims=[['valid_time']],
                            output_core_dims=[[], [], [], [], []],
                            vectorize=True,
                            dask='parallelized',
                            output_dtypes=[float, float, float, int, int]
                        )
                        active_season = xr.DataArray(active_season)
                        effective_season = xr.DataArray(effective_season)
                        mean_season = xr.DataArray(mean_season)
                        heatwave_days_season = xr.DataArray(heatwave_days_season)
                        heatwave_events_season = xr.DataArray(heatwave_events_season)
                        
                        active_season = active_season.assign_coords(year=year)
                        effective_season = effective_season.assign_coords(year=year)
                        mean_season = mean_season.assign_coords(year=year)
                        
                        seasonal_results[season_name] = {
                            'active': active_season,
                            'effective': effective_season,
                            'mean': mean_season
                        }
                else:
                    # 其他季节
                    season_data = t2m_daily.sel(valid_time=t2m_daily.valid_time.dt.month.isin(months))
                    
                    if len(season_data.valid_time) > 0:
                        active_season, effective_season, mean_season, heatwave_days_season, heatwave_events_season = xr.apply_ufunc(
                            calculate_temperature_metrics,
                            season_data,
                            input_core_dims=[['valid_time']],
                            output_core_dims=[[], [], [], [], []],
                            vectorize=True,
                            dask='parallelized',
                            output_dtypes=[float, float, float, int, int]
                        )
                        active_season = xr.DataArray(active_season)
                        effective_season = xr.DataArray(effective_season)
                        mean_season = xr.DataArray(mean_season)
                        heatwave_days_season = xr.DataArray(heatwave_days_season)
                        heatwave_events_season = xr.DataArray(heatwave_events_season)
                        
                        active_season = active_season.assign_coords(year=year)
                        effective_season = effective_season.assign_coords(year=year)
                        mean_season = mean_season.assign_coords(year=year)
                        
                        seasonal_results[season_name] = {
                            'active': active_season,
                            'effective': effective_season,
                            'mean': mean_season
                        }
            
            # 计算月尺度积温
            monthly_results = {}
            for month in range(1, 13):
                month_data = t2m_daily.sel(valid_time=t2m_daily.valid_time.dt.month == month)
                
                if len(month_data.valid_time) > 0:
                    active_month, effective_month, mean_month, heatwave_days_month, heatwave_events_month = xr.apply_ufunc(
                        calculate_temperature_metrics,
                        month_data,
                        input_core_dims=[['valid_time']],
                        output_core_dims=[[], [], [], [], []],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=[float, float, float, int, int]
                    )
                    active_month = xr.DataArray(active_month)
                    effective_month = xr.DataArray(effective_month)
                    mean_month = xr.DataArray(mean_month)
                    heatwave_days_month = xr.DataArray(heatwave_days_month)
                    heatwave_events_month = xr.DataArray(heatwave_events_month)
                    
                    active_month = active_month.assign_coords(year=year, month=month)
                    effective_month = effective_month.assign_coords(year=year, month=month)
                    mean_month = mean_month.assign_coords(year=year, month=month)
                    
                    monthly_results[month] = {
                        'active': active_month,
                        'effective': effective_month,
                        'mean': mean_month
                    }
            
            return {
                'annual': {
                    'active': active_annual,
                    'effective': effective_annual,
                    'mean': mean_annual,
                    'heatwave_days': heatwave_days_annual,
                    'heatwave_events': heatwave_events_annual
                },
                'seasonal': seasonal_results,
                'monthly': monthly_results
            }
    except Exception as e:
        print(f"处理{year}年数据时出错: {e}")
        return None

# 计算趋势
def calculate_all_trends(annual_data, seasonal_data, monthly_data):
    """计算所有时间尺度的趋势"""
    # 年尺度趋势
    annual_trends = {}
    for var in ['active', 'effective', 'mean']:
        data = xr.concat([annual_data[year]['annual'][var] for year in annual_data if annual_data[year] is not None], dim='year')
        trend, p_value = xr.apply_ufunc(
            calculate_trend,
            data,
            input_core_dims=[['year']],
            output_core_dims=[[], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float, float]
        )
        trend = xr.DataArray(trend)
        p_value = xr.DataArray(p_value)
        annual_trends[var] = {'trend': trend, 'p_value': p_value}
    
    # 季节尺度趋势
    seasonal_trends = {}
    for season in seasons:
        seasonal_trends[season] = {}
        for var in ['active', 'effective', 'mean']:
            data_list = []
            for year in seasonal_data:
                if seasonal_data[year] is not None and season in seasonal_data[year]['seasonal']:
                    data_list.append(seasonal_data[year]['seasonal'][season][var])
            
            if data_list:
                data = xr.concat(data_list, dim='year')
                trend, p_value = xr.apply_ufunc(
                    calculate_trend,
                    data,
                    input_core_dims=[['year']],
                    output_core_dims=[[], []],
                    vectorize=True,
                    dask='parallelized',
                    output_dtypes=[float, float]
                )
                trend = xr.DataArray(trend)
                p_value = xr.DataArray(p_value)
                seasonal_trends[season][var] = {'trend': trend, 'p_value': p_value}
    
    # 月尺度趋势
    monthly_trends = {}
    for month in range(1, 13):
        monthly_trends[month] = {}
        for var in ['active', 'effective', 'mean']:
            data_list = []
            for year in monthly_data:
                if monthly_data[year] is not None and month in monthly_data[year]['monthly']:
                    data_list.append(monthly_data[year]['monthly'][month][var])
            
            if data_list:
                data = xr.concat(data_list, dim='year')
                trend, p_value = xr.apply_ufunc(
                    calculate_trend,
                    data,
                    input_core_dims=[['year']],
                    output_core_dims=[[], []],
                    vectorize=True,
                    dask='parallelized',
                    output_dtypes=[float, float]
                )
                trend = xr.DataArray(trend)
                p_value = xr.DataArray(p_value)
                monthly_trends[month][var] = {'trend': trend, 'p_value': p_value}
    
    return annual_trends, seasonal_trends, monthly_trends

# 绘制空间分布图
def plot_spatial_distribution(
    data: xr.DataArray,
    title: str,
    units: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    绘制空间分布图
    
    参数:
    data: 要绘制的数据 (xarray.DataArray)
    title: 图表标题
    units: 数据单位
    vmin: 颜色条最小值 (可选)
    vmax: 颜色条最大值 (可选)
    cmap: 颜色映射 (默认'viridis')
    save_path: 保存路径 (可选)
    
    返回:
    matplotlib.figure.Figure 对象
    """
    # 创建图形和坐标轴
    fig = plt.figure(figsize=(12, 8))
    # 创建坐标轴并设置投影
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 添加地理特征 (使用类型忽略来避免Pylance错误)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)  # type: ignore
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1)  # type: ignore
    ax.add_feature(cfeature.LAND.with_scale('50m'), color='lightgray', alpha=0.3)  # type: ignore
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), color='lightblue', alpha=0.3)  # type: ignore
    
    # 设置经纬度范围
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())  # type: ignore
    
    # 添加并配置网格线
    # Note: Pylance incorrectly flags gridlines() as needing 'self' parameter
    # This is a known issue with cartopy's type hints
    gl = ax.gridlines(  # type: ignore[no-untyped-call, misc]
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False  # type: ignore[attr-defined, no-any-attr]
    gl.right_labels = False  # type: ignore[attr-defined, no-any-attr]
    # 获取坐标（更健壮的实现）
    try:
        if 'longitude' in data.coords:
            lon = data.longitude
        elif 'lon' in data.coords:
            lon = data.lon
        else:
            print(f"错误: 数据集缺少经度坐标. 可用坐标: {list(data.coords)}")
            raise ValueError("数据集缺少经度坐标")
            
        if 'latitude' in data.coords:
            lat = data.latitude
        elif 'lat' in data.coords:
            lat = data.lat
        else:
            print(f"错误: 数据集缺少纬度坐标. 可用坐标: {list(data.coords)}")
            raise ValueError("数据集缺少纬度坐标")
    except Exception as e:
        print(f"获取坐标时出错: {e}")
        # 尝试使用默认坐标名称
        try:
            lon = data.lon
            lat = data.lat
            print("使用默认坐标名称 'lon' 和 'lat' 成功")
        except:
            print("尝试使用默认坐标名称失败")
            raise
    
    # 确保数据是2D的
    if len(data.shape) == 2:
        plot_data = data
    else:
        # 对于更高维数据，取第一个元素
        plot_data = data
        while len(plot_data.shape) > 2:
            plot_data = plot_data.isel({plot_data.dims[0]: 0})
        
        # 如果取出的数据仍然不是2D，取第一个元素
        if len(plot_data.shape) > 2:
            plot_data = plot_data[0]
    
    # Handle NaN values by setting them to a masked array
    masked_data = np.ma.masked_invalid(plot_data)
    
    # 绘制数据
    im = ax.pcolormesh(
        lon, lat, masked_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar with ASCII-safe label
    safe_units = units.encode('ascii', 'ignore').decode()
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label(safe_units, fontsize=12)
    
    # Use ASCII-safe title
    safe_title = title.encode('ascii', 'ignore').decode()
    ax.set_title(safe_title, fontsize=14)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)  # 关闭图形以节省内存
    return fig

# Plot trend scatter
def plot_trend_scatter(
    active_trend: xr.DataArray,
    effective_trend: xr.DataArray,
    mean_trend: xr.DataArray,
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Plot scatter of mean warming vs accumulated temperature changes
    
    Parameters:
    active_trend: Active accumulated temperature trend data
    effective_trend: Effective accumulated temperature trend data
    mean_trend: Mean temperature trend data
    save_path: Save path (optional)
    
    Returns:
    matplotlib.figure.Figure object
    """
    def safe_plot(ax, x_data, y_data, x_label, y_label, title):
        """Safe scatter plot with regression"""
        # Filter NaN values
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_valid = x_data[mask]
        y_valid = y_data[mask]
        
        if len(x_valid) < 2:
            ax.text(0.5, 0.5, 'Insufficient data points', fontsize=14, ha='center', va='center')
            ax.set_title(f'{title} (Insufficient Data)', fontsize=14)
            return
        
        # Plot scatter
        ax.scatter(x_valid, y_valid, alpha=0.3, s=10)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Calculate regression
        try:
            # Handle different versions of scipy
            if hasattr(stats.linregress, 'slope'):
                # Newer scipy version with named attributes
                result = stats.linregress(x_valid, y_valid)
                slope = result.slope
                intercept = result.intercept
                r_value = result.rvalue
                p_value = result.pvalue
            else:
                # Older scipy version returns tuple
                slope, intercept, r_value, p_value, _ = stats.linregress(x_valid, y_valid)
            
            if not np.isnan(slope):
                x_range = np.linspace(np.min(x_valid), np.max(x_valid), 100)
                ax.plot(x_range, slope * x_range + intercept, 'r-', 
                        label=f'Regression: y = {slope:.2f}x + {intercept:.2f}\nr = {r_value:.2f}, p = {p_value:.2e}')
                ax.legend(fontsize=10)
        except Exception as e:
            print(f"Regression calculation failed: {e}")
    
    # Convert to numpy arrays and flatten
    active_flat = active_trend.values.flatten()
    effective_flat = effective_trend.values.flatten()
    mean_flat = mean_trend.values.flatten()
    
    # Ensure float64 type
    active_flat = active_flat.astype(np.float64)
    effective_flat = effective_flat.astype(np.float64)
    mean_flat = mean_flat.astype(np.float64)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Relationship between active accumulated temperature trend and mean temperature trend
    safe_plot(
        ax1, 
        active_flat, 
        mean_flat,
        'Active Accumulated Temperature Trend (°C/decade)',
        'Mean Temperature Trend (°C/decade)',
        'Relationship between Active Accumulated Temperature and Mean Temperature Trends'
    )
    
    # Relationship between effective accumulated temperature trend and mean temperature trend
    safe_plot(
        ax2, 
        effective_flat, 
        mean_flat,
        'Effective Accumulated Temperature Trend (°C/decade)',
        'Mean Temperature Trend (°C/decade)',
        'Relationship between Effective Accumulated Temperature and Mean Temperature Trends'
    )
    
    plt.tight_layout()
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)  # Close figure to save memory
    return fig

# 主函数
def load_all_datasets():
    """加载并处理所有年份的数据集"""
    years = list(range(START_YEAR, END_YEAR + 1))
    annual_data = {}
    seasonal_data = {}
    monthly_data = {}
    
    # 下载所有年份数据
    print("开始下载所有年份数据...")
    for year in tqdm(years, desc="下载年份数据"):
        download_era5_data(year, max_retries=5, delay=120)
    
    # 处理所有年份数据
    print("开始处理所有年份数据...")
    for year in tqdm(years, desc="处理年份数据"):
        result = process_year(year)
        if result:
            annual_data[year] = result
            seasonal_data[year] = result
            monthly_data[year] = result
    
    return annual_data, seasonal_data, monthly_data

def main():
    # 检查xarray后端依赖
    try:
        import netCDF4 as netcdf4
        import h5netcdf
        print("xarray后端依赖已安装")
    except ImportError:
        print("警告: xarray后端依赖未安装 - 请运行: pip install netcdf4 h5netcdf")
    
    # 加载所有数据集
    annual_data, seasonal_data, monthly_data = load_all_datasets()
    
    # 添加数据验证函数
    def validate_data(data, var_name, threshold=0.1):
        """检查数据有效性，确保有足够非NaN值"""
        if data is None:
            return False
        non_nan_count = np.count_nonzero(~np.isnan(data.values))
        total_count = data.size
        valid_ratio = non_nan_count / total_count if total_count > 0 else 0
        if valid_ratio < threshold:
            print(f"警告: {var_name} 数据有效性不足 ({valid_ratio*100:.1f}%)")
            return False
        return True
    
    # 计算多年平均值
    print("计算多年平均值...")
    # 年尺度
    annual_mean = {}
    for var in ['active', 'effective', 'mean']:
        data_list = []
        for year in annual_data:
            if annual_data[year] is not None:
                data_list.append(annual_data[year]['annual'][var])
        
        if data_list:
            # 确保我们得到一个2D数组（纬度，经度）
            da = xr.concat(data_list, dim='year').mean(dim='year')
            # 如果还有年份维度，移除它
            if 'year' in da.dims:
                da = da.squeeze('year', drop=True)
            annual_mean[var] = da
    
    # Plot multi-year mean annual accumulated temperature (Fig 1)
    print("Plotting multi-year mean annual accumulated temperature...")
    if 'active' in annual_mean:
        plot_spatial_distribution(
            annual_mean['active'], 
            'Multi-year Mean Active Accumulated Temperature', 
            '°C',
            vmin=0, 
            vmax=8000, 
            cmap='YlOrRd', 
            save_path=figures_dir / 'annual_active_mean.png'
        )
    else:
        print("警告: 无法绘制多年平均活动积温分布 - 数据缺失")
    
    if 'effective' in annual_mean:
        plot_spatial_distribution(
            annual_mean['effective'],
            'Multi-year Mean Effective Accumulated Temperature',
            '°C',
            vmin=0,
            vmax=6000,
            cmap='YlOrRd',
            save_path=figures_dir / 'annual_effective_mean.png'
        )
    else:
        print("警告: 无法绘制多年平均有效积温分布 - 数据缺失")
    
    if 'mean' in annual_mean:
        plot_spatial_distribution(
            annual_mean['mean'],
            'Multi-year Mean Temperature',
            '°C',
            vmin=-10,
            vmax=30,
            cmap='RdBu_r',
            save_path=figures_dir / 'annual_mean_temp.png'
        )
    else:
        print("警告: 无法绘制多年平均温度分布 - 数据缺失")
    
    # 计算趋势
    print("计算趋势率...")
    annual_trends, seasonal_trends, monthly_trends = calculate_all_trends(
        annual_data, seasonal_data, monthly_data
    )
    
    # Plot annual accumulated temperature trend distribution (Fig 2)
    print("Plotting annual accumulated temperature trend distribution...")
    if validate_data(annual_trends['active']['trend'], 'Active accumulated temperature trend'):
        plot_spatial_distribution(
            annual_trends['active']['trend'],
            'Active Accumulated Temperature Trend',
            '°C/decade',
            vmin=-2,
            vmax=2,
            cmap='RdBu_r',
            save_path=figures_dir / 'annual_active_trend.png'
        )
    else:
        print("警告: 活动积温趋势数据无效，跳过绘图")
    
    if validate_data(annual_trends['effective']['trend'], '有效积温趋势'):
        plot_spatial_distribution(
            annual_trends['effective']['trend'],
            'Effective Accumulated Temperature Trend',
            '°C/decade',
            vmin=-2,
            vmax=2,
            cmap='RdBu_r',
            save_path=figures_dir / 'annual_effective_trend.png'
        )
    else:
        print("警告: 有效积温趋势数据无效，跳过绘图")
    
    if validate_data(annual_trends['mean']['trend'], '平均温度趋势'):
        plot_spatial_distribution(
            annual_trends['mean']['trend'],
            'Mean Temperature Trend',
            '°C/decade',
            vmin=-2,
            vmax=2,
            cmap='RdBu_r',
            save_path=figures_dir / 'annual_mean_trend.png'
        )
    else:
        print("警告: 平均温度趋势数据无效，跳过绘图")
    
    # Plot seasonal accumulated temperature trend distribution
    print("Plotting seasonal accumulated temperature trend distribution...")
    for season in seasons:
        if season in seasonal_trends:
            for var in ['active', 'effective', 'mean']:
                if var in seasonal_trends[season]:
                    if validate_data(seasonal_trends[season][var]['trend'], f'{season} season {var} accumulated temperature trend'):
                        title_map = {
                            'active': 'Active Accumulated Temperature',
                            'effective': 'Effective Accumulated Temperature',
                            'mean': 'Mean Temperature'
                        }
                        title = f'{season.capitalize()} {title_map[var]} Trend'
                        plot_spatial_distribution(
                            seasonal_trends[season][var]['trend'],
                            title,
                            '°C/10年',
                            vmin=-3,
                            vmax=3,
                            cmap='RdBu_r',
                            save_path=figures_dir / f'seasonal_{season}_{var}_trend.png'
                        )
                    else:
                        print(f"警告: {season}季节{var}积温趋势数据无效，跳过绘图")
    
    # Plot monthly accumulated temperature trend distribution
    print("Plotting monthly accumulated temperature trend distribution...")
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    for month in range(1, 13):
        if month in monthly_trends:
            for var in ['active', 'effective', 'mean']:
                if var in monthly_trends[month]:
                    if validate_data(monthly_trends[month][var]['trend'], f'Month {month} {var} accumulated temperature trend'):
                        title_map = {
                            'active': 'Active Accumulated Temperature',
                            'effective': 'Effective Accumulated Temperature',
                            'mean': 'Mean Temperature'
                        }
                        title = f'{month_names[month-1]} {title_map[var]} Trend'
                        plot_spatial_distribution(
                            monthly_trends[month][var]['trend'],
                            title,
                            '°C/10年',
                            vmin=-3,
                            vmax=3,
                            cmap='RdBu_r',
                            save_path=figures_dir / f'monthly_{month:02d}_{var}_trend.png'
                        )
                    else:
                        print(f"警告: {month}月{var}积温趋势数据无效，跳过绘图")
    
    # Analyze relationship between mean warming and accumulated temperature changes
    print("Plotting relationship between mean warming and accumulated temperature changes...")
    # 验证数据有效性
    valid_active = validate_data(annual_trends['active']['trend'], '活动积温趋势')
    valid_effective = validate_data(annual_trends['effective']['trend'], '有效积温趋势')
    valid_mean = validate_data(annual_trends['mean']['trend'], '平均温度趋势')

    if valid_active and valid_effective and valid_mean:
        plot_trend_scatter(
            active_trend=annual_trends['active']['trend'], 
            effective_trend=annual_trends['effective']['trend'], 
            mean_trend=annual_trends['mean']['trend'],
            save_path=figures_dir / 'temperature_accumulated_correlation.png'
        )
    else:
        print("警告: 积温趋势数据无效，跳过绘制散点图")
    
    # Analyze heatwave events
    print("Analyzing heatwave event trends...")
    heatwave_days = []
    heatwave_events = []
    years_list = []
    
    # 收集热浪数据
    for year in annual_data:
        if annual_data[year] is not None:
            try:
                days = annual_data[year]['annual']['heatwave_days'].values.mean()
                events = annual_data[year]['annual']['heatwave_events'].values.mean()
                heatwave_days.append(days)
                heatwave_events.append(events)
                years_list.append(year)
            except KeyError:
                print(f"警告: {year}年热浪数据缺失")
    
    # 绘制热浪事件趋势图
    if len(heatwave_days) > 10:  # 至少需要10年数据
        plt.figure(figsize=(14, 6))
        
        # Heatwave days trend
        plt.subplot(1, 2, 1)
        plt.plot(years_list, heatwave_days, 'o-', label='Observed')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Heatwave Days', fontsize=12)
        plt.title('Annual Heatwave Days Trend', fontsize=14)
        
        # Add trend line
        if len(heatwave_days) > 1:
            z_days = np.polyfit(years_list, heatwave_days, 1)
            p_days = np.poly1d(z_days)
            plt.plot(years_list, p_days(years_list), 'r--', 
                    label=f'Trend: {z_days[0]*10:.2f} days/decade')
        plt.legend(fontsize=10)
        plt.grid(True)
        
        # Heatwave events trend
        plt.subplot(1, 2, 2)
        plt.plot(years_list, heatwave_events, 's-', color='orange', label='Observed')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Heatwave Events', fontsize=12)
        plt.title('Annual Heatwave Events Trend', fontsize=14)
        
        # Add trend line
        if len(heatwave_events) > 1:
            z_events = np.polyfit(years_list, heatwave_events, 1)
            p_events = np.poly1d(z_events)
            plt.plot(years_list, p_events(years_list), 'r--', 
                    label=f'Trend: {z_events[0]*10:.2f} events/decade')
        plt.legend(fontsize=10)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'heatwave_trends.png', dpi=300)
        plt.close()
        print("热浪趋势图已保存")
    else:
        print(f"警告: 热浪数据不足({len(heatwave_days)}年)，跳过绘制热浪趋势图")
    
    # Plot regional mean temperature time series
    print("Plotting regional mean temperature time series...")
    mean_temps = []
    years_list = []
    
    for year in annual_data:
        if annual_data[year] is not None:
            try:
                # 计算区域平均温度
                mean_temp = annual_data[year]['annual']['mean'].values.mean()
                mean_temps.append(mean_temp)
                years_list.append(year)
            except KeyError:
                print(f"警告: {year}年平均温度数据缺失")
    
    if len(mean_temps) > 10:
        plt.figure(figsize=(12, 7))
        plt.plot(years_list, mean_temps, 'o-', color='red', label='Observed')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mean Temperature (°C)', fontsize=12)
        plt.title(f'Regional Mean Temperature Change ({START_YEAR}-{END_YEAR})', fontsize=14)
        
        # Add trend line
        z = np.polyfit(years_list, mean_temps, 1)
        p = np.poly1d(z)
        plt.plot(years_list, p(years_list), 'r--', 
                label=f'Trend: {z[0]*10:.2f}°C/decade')
        
        # Add statistics
        plt.text(0.05, 0.95, 
                f'Mean Temperature: {np.mean(mean_temps):.2f}°C\nTrend: {z[0]*10:.2f}°C/decade',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.savefig(figures_dir / 'regional_mean_temperature_trend.png', dpi=300)
        plt.close()
        print("区域平均温度时间序列图已保存")
    else:
        print(f"警告: 平均温度数据不足({len(mean_temps)}年)，跳过绘制时间序列图")
    
    print("Analysis completed! All results have been saved to the respective directories.")

if __name__ == "__main__":
    main()
