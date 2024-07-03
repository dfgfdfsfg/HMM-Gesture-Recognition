import matplotlib.pyplot as plt
import os

import numpy as np

filepath_1 = './test_local/inf/inf+16.txt'
filepath_2 = './test_local/inf.kalman/inf+16.txt'

data = np.loadtxt(filepath_1, dtype=float)
data_kalman = np.loadtxt(filepath_2, dtype=float)

data = np.array(data)
data_kalman = np.array(data_kalman)
time = data[:, 0]  # 时间
time_kalman = data_kalman[:, 0]  # 时间

namelist = ["AX", "AY", "AZ", "WX", "WY", "WZ"]
time_normalized = time - time[0]
time_normalized_kalman = time_kalman - time_kalman[0]
# 绘制加速度数据
plt.figure(figsize=(15, 12))
for i in range(1, 13):
    plt.subplot(6, 2, i)
    if i % 2 == 1:
        plt.title(f"Original + {namelist[(i+1)// 2 - 1]}")
        plt.plot(time_normalized, data[:, (i+1) // 2])
    else:
        plt.title(f"kalman + {namelist[(i+1)// 2 - 1]}")
        plt.plot(time_normalized_kalman, data_kalman[:, i // 2])

# 显示图表
plt.tight_layout()
plt.show()
