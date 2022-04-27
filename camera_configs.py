# encoding=utf-8

import cv2
import numpy as np


# 左相机内参矩阵
K1 = np.array([[3213.08, 0.0, 600.64723],
               [0.0, 3060.75, 186.58058],
               [0.0, 0.0, 1.0]])

# 左相机畸变系数: [k1,k2,p1,p2,k3,(k4,k5,k6)]
distort1 = np.array([[-0.1426, -0.4962, -0.0150, -0.0744, 0.00000]])

# 右相机内参矩阵
K2 = np.array([[3096.37, 0.0, 400.00856],
               [0.0, 3090.52, 269.37140],
               [0.0, 0.0, 1.0]])

# 右相机畸变系数: [k1,k2,p1,p2,k3,(k4,k5,k6)]
distort2 = np.array([[-0.5450, -0.2870, -0.0110, -0.0329, 0.00000]])

om = np.array([0.01911, 0.03125, -0.00960])  # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R: 旋转矩阵与旋转向量之间的相互转换
T = np.array([-70.59612, -2.60704, 18.87635])  # 平移关系向量
print('R:\n', R)
print('T:\n', T, '\n')

# left_camera_matrix = np.array([[4213.08, 0., 251.64723],
#                                [0.,4060.75, 286.58058],
#                                [0., 0., 1.]])
# left_distortion = np.array([[-0.1426, -0.4962, -0.0150, -0.0744, 0.00000]])
#
# right_camera_matrix = np.array([[3096.37, 0., 217.00856],
#                                 [0., 3090.52, 269.37140],
#                                 [0., 0., 1.]])
# right_distortion = np.array([[-0.5450, -0.2870, -0.0110, -0.0329, 0.00000]])
# om = np.array([0.01911, 0.03125, -0.00960])  # 旋转关系向量
# R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# T = np.array([-70.59612, -2.60704, 18.87635])  # 平移关系向量

size = (512, 384)  # 图像尺寸

# ----- 进行立体更正
# P1: 第一台相机矫正后坐标系的投影矩阵
# P2: 第二台相机矫正后坐标系的投影矩阵
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1,
                                                                  distort1,
                                                                  K2,
                                                                  distort2,
                                                                  size,
                                                                  R,
                                                                  T)

# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, distort1, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, distort2, R2, P2, size, cv2.CV_16SC2)
print("[Info]: stereo rectifying done.")