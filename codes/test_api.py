import cdsapi

def test_cds_api():
    try:
        print("尝试连接CDS API...")
        c = cdsapi.Client()
        print("CDS API连接成功！配置正确。")
        return True
    except Exception as e:
        print(f"CDS API连接失败: {str(e)}")
        print("请确保您已完成以下配置步骤:")
        print("1. 在CDS网站注册账号(https://cds.climate.copernicus.eu/)")
        print("2. 获取API密钥(https://cds.climate.copernicus.eu/api-how-to)")
        print("3. 在用户主目录创建.cdsapirc文件")
        print("4. 文件内容格式: url: [您的URL]\\nkey: [UID]:[API密钥]")
        print("5. 确保已安装cdsapi包: pip install cdsapi")
        return False

if __name__ == "__main__":
    test_cds_api()
