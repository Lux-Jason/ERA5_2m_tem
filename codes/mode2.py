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
results_dir = Path("t3_results")
figures_dir = Path("t3_figures")

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

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU only")

def calculate_accumulated_temperature_grid(temperatures, base_temp=10.0):
    """
    计算活动积温和有效积温 (GPU加速向量化版本)
    
    参数:
    temperatures: 3D温度数组 (时间, 纬度, 经度) in K
    base_temp: 生物学下限温度 (°C)
    
    返回:
    Tuple[活动积温, 有效积温, 平均温度] (单位均为°C)
    """
    # 使用GPU或CPU
    xp = cp if GPU_AVAILABLE else np
    
    # 转换为摄氏度
    temp_c = temperatures - 273.15
    
    # 确保数据是3D的 (时间, 纬度, 经度)
    if temp_c.ndim != 3:
        # 尝试修复维度问题
        if temp_c.ndim == 4:
            # 去除单一维度
            temp_c = xp.squeeze(temp_c)
        else:
            raise ValueError(f"温度数据维度错误: 期望3维，实际得到{temp_c.ndim}维")
    
    # 创建温度 >= 基础温度的掩码
    mask = temp_c >= base_temp
    
    # 计算活动积温 (满足条件的温度值总和)
    active_accumulated = xp.sum(temp_c * mask, axis=0)
    
    # 计算有效积温 (满足条件的 (温度 - 基础温度) 总和)
    effective_accumulated = xp.sum((temp_c - base_temp) * mask, axis=0)
    
    # 计算平均温度
    mean_temp = xp.mean(temp_c, axis=0)
    
    # 确保输出是2D数组 (纬度, 经度)
    def ensure_2d(arr):
        """确保输出为2D数组，自动处理多余维度（兼容GPU）"""
        if arr.ndim == 2:
            return arr
        
        # 尝试去除单一维度
        arr_squeezed = xp.squeeze(arr)
        if arr_squeezed.ndim == 2:
            return arr_squeezed
        
        # 处理3D数组 - 取时间维度平均值
        if arr.ndim == 3:
            return xp.mean(arr, axis=0)
        
        # 特殊处理：当数据维度>3时取首元素
        if arr.ndim > 3:
            return arr[0]
        
        # 最终尝试重塑为二维
        try:
            # 计算二维形状 (尽可能接近正方形)
            size = arr.size
            dim1 = int(xp.sqrt(size))
            dim2 = size // dim1
            return xp.reshape(arr, (dim1, dim2))
        except:
            return arr
    
    active_accumulated = ensure_2d(active_accumulated)
    effective_accumulated = ensure_2d(effective_accumulated)
    mean_temp = ensure_2d(mean_temp)
    
    return active_accumulated, effective_accumulated, mean_temp

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
        
        if np.sum(valid_mask) < 2:  # 降低最小数据点要求
            return np.nan, np.nan
        
        # 提取有效数据
        x = years[valid_mask]
        y = data_values[valid_mask]
        
        # 使用命名元组解包并添加类型忽略
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)  # type: ignore
        slope = slope * 10.0  # type: ignore  # 转换为10年变化率
        
        return float(slope), float(p_value)  # type: ignore
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
            
            # 打印维度信息用于调试
            print(f"处理 {year} 年数据:")
            print(f"原始数据集维度: {ds.dims}")
            print(f"日平均温度维度: {t2m_daily.dims}")
            print(f"日平均温度值形状: {t2m_daily.values.shape}")
            
            # 获取数据数组 (时间, 纬度, 经度)
            data_array = t2m_daily.values
            
            # 使用GPU加速计算积温
            active_annual, effective_annual, mean_annual = calculate_accumulated_temperature_grid(data_array, base_temp=10.0)
            
            # 打印计算后的形状
            print(f"活动积温形状: {active_annual.shape}")
            print(f"有效积温形状: {effective_annual.shape}")
            print(f"平均温度形状: {mean_annual.shape}")
            
            # 转换回numpy数组（如果是GPU）
            if GPU_AVAILABLE:
                active_annual = cp.asnumpy(active_annual)
                effective_annual = cp.asnumpy(effective_annual)
                mean_annual = cp.asnumpy(mean_annual)
            
            # 创建DataArray - 更健壮地处理维度
            def create_data_array(data, dims, coords):
                """创建DataArray，自动处理维度匹配问题"""
                # 首先尝试去除所有单一维度
                data = data.squeeze()
                
                # 记录调试信息
                print(f"创建DataArray: 数据维度={data.ndim}, 预期维度={len(dims)}, 数据形状={data.shape}")
                
                try:
                    # 尝试创建DataArray
                    return xr.DataArray(data, dims=dims, coords=coords)
                except ValueError as e:
                    # 如果维度不匹配，尝试自动对齐维度
                    print(f"维度不匹配警告: {e} - 尝试自动对齐维度")
                    
                    # 确保数据是2D的
                    if data.ndim == 2:
                        return xr.DataArray(data, dims=dims, coords=coords)
                    
                    # 尝试使用坐标中的维度名称
                    if len(coords) == 2:
                        coord_dims = list(coords.keys())
                        return xr.DataArray(data, dims=coord_dims, coords=coords)
                    
                    # 创建通用维度名称作为后备
                    new_dims = [f'dim_{i}' for i in range(data.ndim)]
                    return xr.DataArray(data, dims=new_dims)
            
            
            # 创建DataArray
            active_annual = create_data_array(
                active_annual,
                dims=('latitude', 'longitude'),
                coords={'latitude': t2m_daily.latitude, 'longitude': t2m_daily.longitude}
            )
            effective_annual = create_data_array(
                effective_annual,
                dims=('latitude', 'longitude'),
                coords={'latitude': t2m_daily.latitude, 'longitude': t2m_daily.longitude}
            )
            mean_annual = create_data_array(
                mean_annual,
                dims=('latitude', 'longitude'),
                coords={'latitude': t2m_daily.latitude, 'longitude': t2m_daily.longitude}
            )
            
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
                        
                        # 获取数据数组 (时间, 纬度, 经度)
                        winter_array = winter_data.values
                        
                        # 使用GPU加速计算积温
                        active_season, effective_season, mean_season = calculate_accumulated_temperature_grid(winter_array, base_temp=10.0)
                        
                        # 转换回numpy数组（如果是GPU）
                        if GPU_AVAILABLE:
                            active_season = cp.asnumpy(active_season)
                            effective_season = cp.asnumpy(effective_season)
                            mean_season = cp.asnumpy(mean_season)
                        
                        # 创建DataArray
                        # 创建DataArray - 使用健壮的创建函数
                        active_season = create_data_array(
                            active_season,
                            dims=('latitude', 'longitude'),
                            coords={'latitude': winter_data.latitude, 'longitude': winter_data.longitude}
                        )
                        effective_season = create_data_array(
                            effective_season,
                            dims=('latitude', 'longitude'),
                            coords={'latitude': winter_data.latitude, 'longitude': winter_data.longitude}
                        )
                        mean_season = create_data_array(
                            mean_season,
                            dims=('latitude', 'longitude'),
                            coords={'latitude': winter_data.latitude, 'longitude': winter_data.longitude}
                        )
                        
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
                        # 获取数据数组 (时间, 纬度, 经度)
                        season_array = season_data.values
                        
                        # 使用GPU加速计算积温
                        active_season, effective_season, mean_season = calculate_accumulated_temperature_grid(season_array, base_temp=10.0)
                        
                        # 转换回numpy数组（如果是GPU）
                        if GPU_AVAILABLE:
                            active_season = cp.asnumpy(active_season)
                            effective_season = cp.asnumpy(effective_season)
                            mean_season = cp.asnumpy(mean_season)
                        
                        # 创建DataArray - 使用健壮的创建函数
                        active_season = create_data_array(
                            active_season,
                            dims=('latitude', 'longitude'),
                            coords={'latitude': season_data.latitude, 'longitude': season_data.longitude}
                        )
                        effective_season = create_data_array(
                            effective_season,
                            dims=('latitude', 'longitude'),
                            coords={'latitude': season_data.latitude, 'longitude': season_data.longitude}
                        )
                        mean_season = create_data_array(
                            mean_season,
                            dims=('latitude', 'longitude'),
                            coords={'latitude': season_data.latitude, 'longitude': season_data.longitude}
                        )
                        active_season = xr.DataArray(active_season)
                        effective_season = xr.DataArray(effective_season)
                        mean_season = xr.DataArray(mean_season)
                        
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
                    # 获取数据数组 (时间, 纬度, 经度)
                    month_array = month_data.values
                    
                    # 使用GPU加速计算积温
                    active_month, effective_month, mean_month = calculate_accumulated_temperature_grid(month_array, base_temp=10.0)
                    
                    # 转换回numpy数组（如果是GPU）
                    if GPU_AVAILABLE:
                        active_month = cp.asnumpy(active_month)
                        effective_month = cp.asnumpy(effective_month)
                        mean_month = cp.asnumpy(mean_month)
                    
                    # 创建DataArray - 使用健壮的创建函数
                    active_month = create_data_array(
                        active_month,
                        dims=('latitude', 'longitude'),
                        coords={'latitude': month_data.latitude, 'longitude': month_data.longitude}
                    )
                    effective_month = create_data_array(
                        effective_month,
                        dims=('latitude', 'longitude'),
                        coords={'latitude': month_data.latitude, 'longitude': month_data.longitude}
                    )
                    mean_month = create_data_array(
                        mean_month,
                        dims=('latitude', 'longitude'),
                        coords={'latitude': month_data.latitude, 'longitude': month_data.longitude}
                    )
                    active_month = xr.DataArray(active_month)
                    effective_month = xr.DataArray(effective_month)
                    mean_month = xr.DataArray(mean_month)
                    
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
                    'mean': mean_annual
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

# 保存趋势数据集
def save_trend_datasets(trends, prefix):
    """保存趋势数据到NetCDF文件"""
    for key, data_dict in trends.items():
        if isinstance(data_dict, dict):
            ds = xr.Dataset()
            for var_name, trend_data in data_dict.items():
                if 'trend' in trend_data:
                    ds[f'{var_name}_trend'] = trend_data['trend']
                    ds[f'{var_name}_trend'].attrs['units'] = '°C/10a'
                if 'p_value' in trend_data:
                    ds[f'{var_name}_p_value'] = trend_data['p_value']
            
            if ds:
                ds.to_netcdf(results_dir / f'{prefix}_{key}_trends.nc')
        else:
            print(f"警告: 无法保存{prefix}_{key}的趋势数据 - 数据结构无效")

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

# 绘制趋势散点图
def plot_trend_scatter(
    active_trend: xr.DataArray,
    effective_trend: xr.DataArray,
    mean_trend: xr.DataArray,
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    绘制平均增温与积温变化的关系散点图
    
    参数:
    active_trend: 活动积温趋势数据
    effective_trend: 有效积温趋势数据
    mean_trend: 平均温度趋势数据
    save_path: 保存路径 (可选)
    
    返回:
    matplotlib.figure.Figure 对象
    """
    def safe_plot(ax, x_data, y_data, x_label, y_label, title):
        """安全绘制散点图并计算回归"""
        # 过滤NaN值
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_valid = x_data[mask]
        y_valid = y_data[mask]
        
        if len(x_valid) < 2:
            ax.text(0.5, 0.5, 'Insufficient valid data points', fontsize=14, ha='center', va='center')
            ax.set_title(f'{title} (Insufficient data)', fontsize=14)
            return
        
        # 绘制散点图
        ax.scatter(x_valid, y_valid, alpha=0.3, s=10)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # 计算回归
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
            if not np.isnan(slope):
                x_range = np.linspace(np.min(x_valid), np.max(x_valid), 100)
                ax.plot(x_range, slope * x_range + intercept, 'r-', 
                        label=f'Regression: y = {slope:.2f}x + {intercept:.2f}\nr = {r_value:.2f}, p = {p_value:.2e}')
                ax.legend(fontsize=10)
        except Exception as e:
            print(f"Regression calculation failed: {e}")
    
    # 转换为numpy数组并展平
    active_flat = active_trend.values.flatten().astype(np.float64)
    effective_flat = effective_trend.values.flatten().astype(np.float64)
    mean_flat = mean_trend.values.flatten().astype(np.float64)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 活动积温与平均温度趋势关系
    safe_plot(
        ax1, 
        active_flat, 
        mean_flat,
        'Active accumulated temperature trend (°C/10a)',
        'Mean temperature trend (°C/10a)',
        'Relationship between active accumulated temperature trend and mean temperature trend'
    )
    
    # 有效积温与平均温度趋势关系
    safe_plot(
        ax2, 
        effective_flat, 
        mean_flat,
        'Effective accumulated temperature trend (°C/10a)',
        'Mean temperature trend (°C/10a)',
        'Relationship between effective accumulated temperature trend and mean temperature trend'
    )
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)  # 关闭图形以节省内存
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
    def validate_data(data, var_name, min_valid_points=10):
        """检查数据有效性，确保有足够非NaN值"""
        if data is None:
            return False
        non_nan_count = np.count_nonzero(~np.isnan(data.values))
        if non_nan_count < min_valid_points:
            print(f"警告: {var_name} 有效数据点不足 ({non_nan_count}/{data.size})")
            return False
        return True
    
    # 计算多年平均值并保存结果
    print("计算多年平均值并保存结果...")
    # 年尺度
    annual_mean = {}
    annual_data_list = {var: [] for var in ['active', 'effective', 'mean']}
    
    # 收集所有年份数据
    for year in annual_data:
        if annual_data[year] is not None:
            for var in ['active', 'effective', 'mean']:
                annual_data_list[var].append(annual_data[year]['annual'][var])
    
    # 计算平均值并保存
    for var in ['active', 'effective', 'mean']:
        if annual_data_list[var]:
            # 创建包含所有年份的数据集
            data_all_years = xr.concat(annual_data_list[var], dim='year')
            data_all_years.name = var
            data_all_years.attrs['units'] = '°C'
            
            # 保存所有年份的原始数据
            data_all_years.to_netcdf(results_dir / f'annual_{var}_all_years.nc')
            
            # 计算多年平均值
            da = data_all_years.mean(dim='year')
            da.name = f'{var}_mean'
            da.attrs['units'] = '°C'
            
            # 保存多年平均值
            da.to_netcdf(results_dir / f'annual_{var}_mean.nc')
            annual_mean[var] = da
    
    # 绘制多年平均年积温分布（图1）
    print("绘制多年平均年积温分布图...")
    if 'active' in annual_mean:
        plot_spatial_distribution(
            annual_mean['active'], 
            '多年平均活动积温分布', 
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
            '多年平均有效积温分布',
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
            '多年平均温度分布',
            '°C',
            vmin=-10,
            vmax=30,
            cmap='RdBu_r',
            save_path=figures_dir / 'annual_mean_temp.png'
        )
    else:
        print("警告: 无法绘制多年平均温度分布 - 数据缺失")
    
    # 计算趋势并保存结果
    print("计算趋势率并保存结果...")
    annual_trends, seasonal_trends, monthly_trends = calculate_all_trends(
        annual_data, seasonal_data, monthly_data
    )
    
    # 保存趋势数据集
    save_trend_datasets(annual_trends, 'annual')
    save_trend_datasets(seasonal_trends, 'seasonal')
    save_trend_datasets(monthly_trends, 'monthly')
    
    # 绘制年积温趋势率分布（图2）
    print("绘制年积温趋势率分布图...")
    if validate_data(annual_trends['active']['trend'], '活动积温趋势'):
        plot_spatial_distribution(
            annual_trends['active']['trend'],
            '活动积温趋势率分布',
            '°C/10年',
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
            '有效积温趋势率分布',
            '°C/10年',
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
            '平均温度趋势率分布',
            '°C/10年',
            vmin=-2,
            vmax=2,
            cmap='RdBu_r',
            save_path=figures_dir / 'annual_mean_trend.png'
        )
    else:
        print("警告: 平均温度趋势数据无效，跳过绘图")
    
    # 绘制季节积温趋势率分布
    print("绘制季节积温趋势率分布图...")
    for season in seasons:
        if season in seasonal_trends:
            for var in ['active', 'effective', 'mean']:
                if var in seasonal_trends[season]:
                    if validate_data(seasonal_trends[season][var]['trend'], f'{season}季节{var}积温趋势'):
                        title = f'{season.capitalize()} {var}积温趋势率分布'
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
    
    # 绘制月尺度积温趋势率分布
    print("绘制月尺度积温趋势率分布图...")
    month_names = ['一月', '二月', '三月', '四月', '五月', '六月', 
                  '七月', '八月', '九月', '十月', '十一月', '十二月']
    
    for month in range(1, 13):
        if month in monthly_trends:
            for var in ['active', 'effective', 'mean']:
                if var in monthly_trends[month]:
                    if validate_data(monthly_trends[month][var]['trend'], f'{month}月{var}积温趋势'):
                        title = f'{month_names[month-1]} {var}积温趋势率分布'
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
    
    # 建立平均增温与积温变化的关系
    print("绘制平均增温与积温变化的关系图...")
    # 验证数据有效性 - 现在只需要有效点数量大于2即可
    if validate_data(annual_trends['active']['trend'], '活动积温趋势', min_valid_points=2) and \
       validate_data(annual_trends['effective']['trend'], '有效积温趋势', min_valid_points=2) and \
       validate_data(annual_trends['mean']['trend'], '平均温度趋势', min_valid_points=2):
        plot_trend_scatter(
            active_trend=annual_trends['active']['trend'], 
            effective_trend=annual_trends['effective']['trend'], 
            mean_trend=annual_trends['mean']['trend'],
            save_path=figures_dir / 'temperature_accumulated_correlation.png'
        )
    else:
        print("警告: 积温趋势数据无效，跳过绘制散点图")
    
    print("分析完成！所有结果已保存到相应目录。")

if __name__ == "__main__":
    main()
