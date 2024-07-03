import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import os
from LH_HMM import LH_HMM
from filterpy.kalman import KalmanFilter
from tqdm import tqdm
filename = "./test"
Gesturelist = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']
for name in Gesturelist:
    file_name = os.path.join(filename, name)
    writefile = os.path.join(filename, name + ".kalman")

    if not os.path.exists(writefile):
        os.makedirs(writefile)

    for txt_file in os.listdir(file_name):
        if txt_file.endswith('.txt'):
            openfile = os.path.join(file_name, txt_file)
            features = []
            with open(openfile, 'r', encoding='utf8') as f:
                for line in f:
                    data = line.split()
                    features.append([float(x) for x in data])

            write_file = os.path.join(writefile, txt_file)
            with open(write_file, 'w') as file:
                kf = KalmanFilter(dim_x=6, dim_z=6)  # 适当调整维度
                kf.F = np.eye(6)  # 状态转移矩阵
                kf.H = np.eye(6)  # 观测矩阵
                kf.R = np.eye(6) * 0.5  # 观测噪声协方差
                kf.Q = np.eye(6) * 0.1  # 过程噪声协方差
                kf.P *= 100  # 初始估计误差协方差
                for i in tqdm(features):
                    time, ax, ay, az, gx, gy, gz = i
                    z = np.array([ax, ay, az, gx, gy, gz])
                    kf.predict()
                    kf.update(z)
                    file.write(str(time) + ' ')
                    for j in kf.x:
                        file.write(str(j[0])+' ')
                    file.write('\n')
                # file.write(kf.x)
