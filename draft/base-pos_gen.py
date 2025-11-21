import numpy as np

# 定义经纬度范围
longitude_range = (-96, -90)
latitude_range = (27, 31)

# 定义划分数目
num_longitudes = 6  # 经度划分为2个区间，取2个中间点
num_latitudes = 2  # 纬度划分为3个区间，取5个中间点

# 将经度和纬度分别划分为均匀的点
longitudes = np.linspace(longitude_range[0], longitude_range[1], num_longitudes + 1)[1:-1]  # 取去掉顶点的中间点
latitudes = np.linspace(latitude_range[0], latitude_range[1], num_latitudes + 1)[1:-1]  # 取去掉顶点的中间点

# 通过笛卡尔积将经度和纬度组合，生成所有可能的基站位置
base_positions = np.array(np.meshgrid(latitudes, longitudes)).T.reshape(-1, 2)

# 保存为npy文件
np.save('dblrp_5_10_base_positions.npy', base_positions)

# 打印结果，检查生成的坐标
print("Generated base positions (longitude, latitude):")
print(base_positions)