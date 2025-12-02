import pickle
import os
import numpy as np
x_vect = [1, 2, 3]
demand = [4, 5, 6]
a = x_vect + demand

cons = "1_2"
# 你可以在这里进一步处理数据并用它们来构建优化模型
# print("a", a)
#
# demand.append(7)
# print("demand", demand)
# print("a", a)
if '1' in cons:
    print('yes')