import xarray as xr
import os

def inspect_file(file_path):
    try:
        ds = xr.open_dataset(file_path)
        print(f"文件: {os.path.basename(file_path)}")
        print(f"维度: {list(ds.dims)}")
        print(f"坐标: {list(ds.coords)}")
        print(f"数据变量: {list(ds.data_vars)}")
        print(f"时间坐标示例: {ds.time.values[0] if 'time' in ds.coords else '无time坐标'}")
        print(f"valid_time坐标示例: {ds.valid_time.values[0] if 'valid_time' in ds.coords else '无valid_time坐标'}")
        print("-"*50)
    except Exception as e:
        print(f"检查{file_path}出错: {str(e)}")

if __name__ == "__main__":
    data_dir = "./data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
    for f in files[:2]:  # 只检查前两个文件
        inspect_file(os.path.join(data_dir, f))
