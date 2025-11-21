import numpy as np


# def load_and_inspect_npz(file_path):
#     """
#     加载 .npz 文件并显示所有键及其对应的数据。
#     适合在 PyCharm 中添加断点进行调试和查看数据。
#
#     参数:
#     file_path (str): .npz 文件的路径
#     """
#     # 加载 .npz 文件
#     npz_data = np.load(file_path, allow_pickle=True)
#
#     # 打印所有的键和相应的内容
#     print("Keys in the .npz file:")
#     for key in npz_data.files:
#         print(f"Key: {key}")
#         print(f"Data shape: {npz_data[key].shape}")
#         let_me_see = npz_data[key]
#         print('let_me_see:', let_me_see)
#         print(f"Data: {npz_data[key]}")
#         print("-" * 50)
#
#     return npz_data  # 返回 npz 数据，供调试时查看
#
#
# # 调用该函数并检查数据
# file_path = '20240830_368278220_gminmax.npz'
# npz_data = load_and_inspect_npz(file_path)



import numpy as np
import os

# def process_vessel_data(npz_folder, output_x_hist_path, output_y_fut_path):
#     """
#     处理指定文件夹中的所有 .npz 文件，生成 x_hist 和 y_fut 文件。
#
#     参数:
#     npz_folder (str): 存放所有 .npz 文件的文件夹路径
#     output_x_hist_path (str): 输出的 x_hist 文件路径
#     output_y_fut_path (str): 输出的 y_fut 文件路径
#     """
#     x_hist_list = []
#     y_fut_list = []
#
#     # 遍历文件夹中的所有 .npz 文件
#     for npz_file in os.listdir(npz_folder):
#         if npz_file.endswith('.npz'):
#             # 加载 .npz 文件
#             npz_path = os.path.join(npz_folder, npz_file)
#             npz_data = np.load(npz_path, allow_pickle=True)
#
#             # 提取 'X' 和 'Y' 数据
#             X = npz_data['X']  # 形状: (145, 72, 5)
#             Y = npz_data['Y']  # 形状: (142, 72, 2)
#
#             # 获取第一次滑动取值（索引为0）
#             x_hist = X[0]  # 取第一个滑动值 (72, 5)
#             y_fut = Y[0]   # 取第一个滑动值 (72, 2)
#
#             # 将单艘船的数据添加到列表中
#             x_hist_list.append(x_hist)
#             y_fut_list.append(y_fut)
#
#     # 将所有船的 x_hist 和 y_fut 拼接成一个三维数组
#     x_hist_array = np.stack(x_hist_list, axis=0)  # 形状: (n_vessels, 72, 5)
#     y_fut_array = np.stack(y_fut_list, axis=0)    # 形状: (n_vessels, 72, 2)
#
#     # 保存为 .npy 文件
#     np.save(output_x_hist_path, x_hist_array)
#     np.save(output_y_fut_path, y_fut_array)
#
#     print(f"Saved x_hist to {output_x_hist_path}")
#     print(f"Saved y_fut to {output_y_fut_path}")
#
# # 使用示例
# npz_folder = 'gminmax_npz_10_vessels'  # 请替换为实际路径
# output_x_hist_path = 'x_hist.npy'  # 输出文件路径
# output_y_fut_path = 'y_fut.npy'    # 输出文件路径
#
# process_vessel_data(npz_folder, output_x_hist_path, output_y_fut_path)

x_hist = np.load('x_hist.npy')
y_fut = np.load('y_fut.npy')
print('x_hist shape:', x_hist.shape)  # 应该是 (n_vessels, 72, 5)