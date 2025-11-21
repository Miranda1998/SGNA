import numpy as np

# 假设原始数据是一个形状为 (10, 72, 2) 的数组
original_array = np.random.rand(10, 72, 2)  # 这里是一个示例，你可以替换为你的原数据

# 压缩后的时刻数为36，所以我们需要每隔一个时刻选择一次
compressed_array = original_array[:, ::2, :]  # 每隔一个时刻取一次，保留奇数索引

# 输出压缩后的数组的形状
print(compressed_array.shape)  # 应该是 (10, 36, 2)