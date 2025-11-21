import pickle
import os
import numpy as np

# 加载问题实例
# 获取当前脚本所在的文件夹路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 构建 .pkl 文件的完整路径
file_path = os.path.join(current_directory, 'data', 'cflp', 'inst_f10_c10_r2.0_iss1_bt1_nsp10000_nse5000_sd7.pkl')



with open(file_path, 'rb') as file:
    cflp_instance = pickle.load(file)

# 提取数据
n_customers = cflp_instance['n_customers']
n_facilities = cflp_instance['n_facilities']
demands = cflp_instance['demands']
capacities = cflp_instance['capacities']
fixed_costs = cflp_instance['fixed_costs']
trans_costs = cflp_instance['trans_costs']
c_x = cflp_instance['c_x']
c_y = cflp_instance['c_y']
f_x = cflp_instance['f_x']
f_y = cflp_instance['f_y']

# 打印信息检查
print(f"Number of customers: {n_customers}")
print(f"Number of facilities: {n_facilities}")
print(f"Demands: {demands}")
print(f"Facility capacities: {capacities}")
print(f"Transportation costs: {trans_costs}")
print(f"Facility locations: {list(zip(f_x, f_y))}")
print(f"Customer locations: {list(zip(c_x, c_y))}")

# 你可以在这里进一步处理数据并用它们来构建优化模型
