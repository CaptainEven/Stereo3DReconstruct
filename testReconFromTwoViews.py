# encoding=utf-8

import cv2
import math
import numpy as np
import camera_configs
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


# # 读取原图
# frame1 = cv2.imread("./img/01.bmp")
# frame2 = cv2.imread("./img/02.bmp")

# # 畸变矫正
# left_undistort = cv2.undistort(frame1,
#                                camera_configs.left_camera_matrix,
#                                camera_configs.left_distortion)
#
# right_undistort = cv2.undistort(frame2,
#                                 camera_configs.right_camera_matrix,
#                                 camera_configs.right_distortion)
#
# cv2.imwrite('./left_undistort.jpg', left_undistort)
# print('left_undistort.jpg written')
# cv2.imwrite('./right_undistort.jpg', right_undistort)
# print('right_undistort.jpg written')

# img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
# img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
#
# cv2.imwrite('./left_rect.jpg', img1_rectified)
# print('left_rect.jpg written')
# cv2.imwrite('./right_rect.jpg', img2_rectified)
# print('right_rect.jpg written')


# 相机(投影)矩阵P的分解: QR分解
def decompose_P(P):
    """
    :param P:
    :return:
    """
    assert P.shape == (3, 4)
    M = P[:3, :3]

    # -----
    # q: the orthogonal/unitary matrix
    # r: the upper-triangular matrix.
    q, r = np.linalg.qr(M)
    R, K = q, r
    KT = P[:, 3]
    T = np.dot(np.linalg.inv(K), KT)

    return K, R, T


# 相机坐标系下归一化坐标, 畸变
def distort_pt2d(fx, fy, cx, cy, k1, k2, pt2d):
    """
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :param k1:
    :param k2:
    :param pt2d: 相机坐标系下归一化坐标
    :return:
    """
    pt2d = np.array(pt2d, dtype=np.float32).squeeze()
    assert pt2d.size == 2
    u, v = pt2d  # 相机坐标系下归一化坐标

    # compute r^2 and r^4
    r_square = (u * u) + (v * v)
    r_quadric = r_square * r_square

    # do radial distortion only
    uc = u * (1.0 + k1 * r_square + k2 * r_quadric)
    vc = v * (1.0 + k1 * r_square + k2 * r_quadric)

    # convert back to pixel coordinates
    # using nearest neighbor interpolation
    u_distorted = fx * uc + cx
    v_distorted = fy * vc + cy

    return np.array([u_distorted, v_distorted], dtype=np.float32)


# 相机坐标系下归一化坐标, 去畸变
def undistort_pt2d(fx, fy, cx, cy, k1, k2, pt2d):
    """
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :param pt2d: 相机坐标系下归一化坐标
    :return:
    """
    assert pt2d.size() == 2
    u, v = pt2d

    # compute r^2 and r^4
    r_square = (u * u) + (v * v)
    r_quadric = r_square * r_square

    # do radial un-distortion only
    uc = u / (1.0 + k1 * r_square + k2 * r_quadric)
    vc = v / (1.0 + k1 * r_square + k2 * r_quadric)

    # convert back to pixel coordinates
    # using nearest neighbor interpolation
    u_corrected = fx * uc + cx
    v_corrected = fy * vc + cy

    return [u_corrected, v_corrected]


## --------- 自定义三角测量: SVD解齐次线性方程组(最小二乘估计值)
#            X
# P1 -x1 0   λ1
# P2 0 -x2   λ1
#
def my_triangulate(P1, P2, p1, p2):
    """
    :param P1:  camera matrix 1
    :param P2:  camera matrix 2
    :param p1:  pts2d 1
    :param p2:  pts2d 2
    :return:
    """
    assert p1.shape == p2.shape
    pts3d, pts3d_homo = [], []

    M = np.zeros((6, 6), dtype=np.float32)
    M[:3, :4] = np.float32(P1)
    M[3:, :4] = np.float32(P2)
    for pt_i, (x1, x2) in enumerate(zip(p1, p2)):
        if x1.size == 2 and x2.size == 2:
            x1 = np.array([x1[0], x1[1], 1.0], dtype=np.float32).T
            x2 = np.array([x2[0], x2[1], 1.0], dtype=np.float32).T

        M[:3, 4] = -x1
        M[3:, 5] = -x2

        # ----- SVD matrix decomposition
        U, S, V = np.linalg.svd(M)
        # -----

        X = V[-1, :4]

        pt3d_homo = X / X[3]
        pts3d.append(np.array([pt3d_homo[0], pt3d_homo[1], pt3d_homo[2]], dtype=np.float32))
        pts3d_homo.append(np.array(pt3d_homo, dtype=np.float32))

    return np.array(pts3d, dtype=np.float32), np.array(pts3d_homo, dtype=np.float32)


def my_camera_matrix_P(x, X):
    """
    通过SVD解齐次线性方程组, 求解相机的投影矩阵P(3×4)
    :param x: 齐次坐标表示的2D坐标
    :param X: 齐次坐标表示的3D坐标
    :return:
    """
    n = x.shape[1]
    if X.shape[1] != n:
        print("[Err]: number of points don't match")
        return

    # 创建用于计算DLT解的系数矩阵
    M = np.zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = X[:, i]
        M[3 * i + 1, 4:8] = X[:, i]
        M[3 * i + 2, 8:12] = X[:, i]
        M[3 * i:3 * i + 3, i + 12] = -x[:, i]

    U, S, V = np.linalg.svd(M)

    return V[-1, :12].reshape((3, 4))


def reconstructFromBino(P1, P2, p1, p2):
    """
    :param P1:
    :param P2:
    :param p1:  image1 points 2d
    :param p2:  image2 points 2d
    :return:
    """
    # p1 p2 原本Nx2 转置为2*N
    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)  # 返回4×N

    pts3d = []
    # print(pts4d.shape)
    for i in range(pts4d.shape[1]):  # 列数表示计算出来空间点的个数 将三角化的结果进行处理得到“正常”的点坐标
        col = pts4d[:, i]
        col = col / float(col[3])
        pts3d.append([col[0], col[1], col[2]])

    return np.array(pts3d)


def reproj_err_my(P, pt3d, pt2d, distort_coefs=[]):
    """
    :param P:
    :param pt3d:
    :param pt2d:
    :param distort_coefs:
    :return:
    """
    assert P.shape == (3, 4)
    P = np.float32(P)

    if isinstance(pt3d, list) and len(pt3d) == 3:
        pt3d.extend([1.0])
    elif type(pt3d) is np.ndarray:
        if pt3d.size == 3:
            pt3d = pt3d.squeeze()
            pt3d = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
        elif pt3d.size == 4:
            pt3d = pt3d

    ## ---------- 获取重投影坐标λx = PX = K[R|T]X
    if distort_coefs == []:
        # 这里没有考虑畸变的影响
        pt_homo = P.dot(np.array(pt3d))
        pt2d_my = pt_homo / pt_homo[2]
        pt2d_my = pt2d_my[:-1]
    else:  # 重投影且考虑畸变系数的影响: 这里只考虑主要的切向畸变k1, k2
        K, R, T = decompose_P(P)  # 相机投影矩阵分解: 这里会有误差产生
        K = np.float32(K)
        R = np.float32(R)
        T = np.float32(T)
        k1, k2 = distort_coefs.squeeze()[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x3d = pt3d.squeeze()
        x3d = x3d[:-1].reshape(3, 1)

        # 世界坐标系 ——> 相机坐标系
        x_cam = np.dot(R, x3d) + T.reshape(3, 1)

        # 相机坐标系下归一化坐标
        x_norm = np.array([x_cam[0] / x_cam[2], x_cam[1] / x_cam[2]], dtype=np.float32)

        # 畸变操作
        pt2d_my = distort_pt2d(fx, fy, cx, cy, k1, k2, x_norm)
        pt2d_my = pt2d_my.squeeze()

    return np.mean(np.abs(pt2d_my - pt2d))


def my_reproj_err(K, R, T, pt3d, pt2d, distort_coefs=[]):
    """
    :param K:
    :param R:
    :param T:
    :param pt3d:
    :param pt2d:
    :param distort_coefs:
    :return:
    """
    # 构建相机矩阵
    K = np.float32(K)
    R = np.float32(R)
    T = np.float32(T)
    P = np.zeros((3, 4), dtype=np.float32)
    P[:3, :3] = R
    P[:, 3] = T.squeeze()
    P = np.dot(K, P)

    # 齐次坐标
    if isinstance(pt3d, list) and len(pt3d) == 3:
        pt3d.extend([1.0])
    elif type(pt3d) is np.ndarray:
        if pt3d.size == 3:
            pt3d = pt3d.squeeze()
            pt3d = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0], dtype=np.float32).reshape(4, 1)
        elif pt3d.size == 4:
            pt3d = pt3d.reshape(4, 1)

    # ---------- 获取重投影坐标λx = PX = K[R|T]X
    if len(distort_coefs) == 0:  # 如果不考虑畸变: 直接矩阵乘法
        pt_homo = np.dot(P, pt3d)
        pt2d_my = pt_homo / pt_homo[2]
        pt2d_my = pt2d_my[:-1]
    else:  # 考虑畸变的影响
        k1, k2 = np.array(distort_coefs, dtype=np.float32).squeeze()[:2]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x3d = pt3d.squeeze()
        x3d = x3d[:-1].reshape(3, 1)

        # 世界坐标系 ——> 相机坐标系
        x_cam = np.dot(R, x3d) + T.reshape(3, 1)

        # 相机坐标系下归一化坐标
        x_norm = np.array([x_cam[0] / x_cam[2], x_cam[1] / x_cam[2]], dtype=np.float32)

        # 畸变操作
        pt2d_my = distort_pt2d(fx, fy, cx, cy, k1, k2, x_norm)

        # 格式化
        pt2d_my = pt2d_my.squeeze()

    return np.mean(np.abs(pt2d_my - pt2d))


def reproj_err_cv(P, pt3d, pt2d, distort_coefs=[]):
    """
    :param P:
    :param pt3d:
    :param pt2d:
    :return:
    """
    K, R, T = decompose_P(P)
    return cv_reproj_err(K, R, T, pt3d, pt2d, distort_coefs)


def cv_reproj_err(K, R, T, pt3d, pt2d, distort_coefs=[]):
    """
    :param K:
    :param R:
    :param T:
    :param pt3d:
    :param pt2d:
    :return:
    """
    if distort_coefs == []:
        distort_coefs = np.array([], dtype=np.float64)
    if isinstance(pt3d, list):
        if len(pt3d) == 3:
            pt3d = np.array(pt3d, dtype=np.float64).reshape(1, 3)
        elif len(pt3d) == 4:
            pt3d = np.array(pt3d[:3], dtype=np.float64).reshape(1, 3)
    elif type(pt3d) is np.ndarray:
        if pt3d.size == 3:
            pt3d = pt3d.reshape(1, 3)
        elif pt3d.size == 4:
            pt3d = pt3d.reshape(1, -1)[0, :3].reshape(1, 3)

    r_vec, _ = cv2.Rodrigues(R)  # openCV的重投影函数需要旋转向量
    pt2d_my, Jacob = cv2.projectPoints(pt3d, r_vec, T, K, distort_coefs)  # 进行重投影的时候为什么要考虑畸变?
    pt2d_my = np.squeeze(pt2d_my)

    return np.mean(np.abs(pt2d_my - pt2d))  # 取绝对值


def reconFrom2Views(K1, K2, R1, T1, R2, T2, p1, p2):
    # 设置两个相机的投影矩阵[R T]，且转为float型数据 triangulatePoints函数只支持float
    P1 = np.zeros((3, 4), dtype=np.float32)
    P2 = np.zeros((3, 4), dtype=np.float32)

    P1[0:3, 0:3] = np.float32(R1)  # 将R1赋值给proj1的对应位置（前三行三列）
    P1[:, 3] = np.float32(T1.T)  # 将T1的转置赋值给proj1的对应位置（第四列）
    P2[0:3, 0:3] = np.float32(R2)
    P2[:, 3] = np.float32(T2.T)

    fk1 = np.float32(K1)
    fk2 = np.float32(K2)

    P1 = np.dot(fk1, P1)
    P2 = np.dot(fk2, P2)

    # p1 p2 原本Nx2 转置为2*N
    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)  # 4×N

    pts3d = []
    # print(pts4d.shape)
    for i in range(pts4d.shape[1]):  # 列数表示计算出来空间点的个数 将三角化的结果进行处理得到“正常”的点坐标
        col = pts4d[:, i]
        col = col / float(col[3])
        pts3d.append([col[0], col[1], col[2]])

    return np.array(pts3d)


def twoviews_recon():
    # 读取原图
    frame1 = cv2.imread("./img/01.bmp")
    frame2 = cv2.imread("./img/02.bmp")

    # 畸变矫正
    img1_undistort = cv2.undistort(frame1,
                                   camera_configs.K1,
                                   camera_configs.distort1)

    img2_undistort = cv2.undistort(frame2,
                                   camera_configs.K2,
                                   camera_configs.distort2)

    img1_undistort_gray = cv2.cvtColor(img1_undistort, cv2.COLOR_BGR2GRAY)
    img2_undistort_gray = cv2.cvtColor(img2_undistort, cv2.COLOR_BGR2GRAY)

    img1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 检测SIFT关键点
    # ---------- SIFT/SURF/AKAZE extractor
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.AKAZE_create()
    # kpts1, descriptors1 = sift.detectAndCompute(img1_undistort_gray, None)
    # kpts2, descriptors2 = sift.detectAndCompute(img2_undistort_gray, None)
    kpts1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    kpts2, descriptors2 = sift.detectAndCompute(img2_gray, None)
    # ----------

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 设置两距离比值小于0.7时为可用匹配  （Lowe's ratio test）
    valid_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            valid_matches.append(m)

    good_matches = []
    if len(valid_matches) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in valid_matches])
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in valid_matches])

        # RANSAC随机采样一致
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

        # 将mask变成一维数组
        mask = mask.ravel().tolist()

        ## ----- mask out invalid match
        for match_i, match in enumerate(valid_matches):
            if mask[match_i] > 0:
                good_matches.append(match)

    else:
        print("not have enough matches!")

    good_matches = np.array(good_matches)

    # # 可视化特征点匹配
    # good_matches = np.expand_dims(good_matches, 0)
    # img_feat_match = cv2.drawMatchesKnn(img1_undistort, kpts1, img2_undistort, kpts2, good_matches[:15], None, flags=2)
    # win_name = 'Feature matching'
    # cv2.imshow(win_name, img_feat_match)
    # cv2.waitKey()
    # cv2.destroyWindow('Feature matching')

    good_matches = np.squeeze(good_matches)
    p1 = np.asarray([kpts1[m.queryIdx].pt for m in good_matches])  # queryIndex 查询图像中描述符的索引
    p2 = np.asarray([kpts2[m.trainIdx].pt for m in good_matches])
    # # -----

    pts3d = reconFrom2Views(K1=camera_configs.K1,
                            K2=camera_configs.K2,
                            R1=np.eye(3, 3, dtype=np.float32),
                            T1=np.zeros((3, 1), dtype=np.float32),
                            R2=camera_configs.R,
                            T2=camera_configs.T,
                            p1=p1,
                            p2=p2)

    inds = np.where(pts3d[:, 2] > 0)
    good_3d_pts = pts3d[inds]
    good_3d_pts *= 0.001  # mm -> m
    print('Some good Pts3D(m):\n', good_3d_pts[:5], '\n Remain total {} good 3D points.'.format(len(good_3d_pts)))

    ## ---------- 验证投影3D ——> 2D
    K1 = camera_configs.K1
    K2 = camera_configs.K2
    R1 = np.eye(3, 3, dtype=np.float32)
    T1 = np.zeros((3, 1), dtype=np.float32)
    R2 = camera_configs.R
    T2 = camera_configs.T
    # pts3d = pts3d.tolist()
    err_sum_my, err_sum_cv = 0.0, 0.0
    for pt2d_1, pt2d_2, pt3d in zip(p1, p2, pts3d):
        ## ----- left frame's re-projection error
        err_my_1 = my_reproj_err(K1, R1, T1, pt3d, pt2d_1, camera_configs.distort1)
        err_cv_1 = cv_reproj_err(K1, R1, T1, pt3d, pt2d_1)

        ## ----- right frame's re-projection error
        err_my_2 = my_reproj_err(K2, R2, T2, pt3d, pt2d_2, camera_configs.distort2)
        err_cv_2 = cv_reproj_err(K2, R2, T2, pt3d, pt2d_2)

        err_mean_my = np.mean(np.array([err_my_1, err_my_2]))
        err_mean_cv = np.mean(np.array([err_cv_1, err_cv_2]))

        err_sum_my += err_mean_my
        err_sum_cv += err_mean_cv

        # print('Re-projection error(my): {:.3f} pixel'.format(err_mean_my))
        # print('Re-projection error(cv): {:.3f} pixel\n'.format(err_mean_cv))

    print('Mean re-projection error(my): {:6.4} pixel'.format(err_sum_my / float(len(pts3d))))
    print('Mean re-projection error(cv): {:6.4} pixel\n'.format(err_sum_cv / float(len(pts3d))))

    pts3d = np.array(pts3d)[:, :-1]

    return pts3d, p1, p2


def my_recon_twoviews():
    # 读取原图
    frame1 = cv2.imread("./img/01.bmp")
    frame2 = cv2.imread("./img/02.bmp")

    # 畸变矫正
    img1_undistort = cv2.undistort(frame1,
                                   camera_configs.K1,
                                   camera_configs.distort1)

    img2_undistort = cv2.undistort(frame2,
                                   camera_configs.K2,
                                   camera_configs.distort2)

    # # 选取左右视图关键点
    # p1 = np.array([[199, 74], [335, 74], [337, 257], [200, 259]], dtype=np.float32)
    # p2 = np.array([[148, 62], [295, 64], [294, 248], [148, 249]], dtype=np.float32)

    img1_undistort_gray = cv2.cvtColor(img1_undistort, cv2.COLOR_BGR2GRAY)
    img2_undistort_gray = cv2.cvtColor(img2_undistort, cv2.COLOR_BGR2GRAY)

    # 检测SIFT关键点
    # ---------- SIFT/SURF/AKAZE extractor
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.AKAZE_create()
    kpts1, descriptors1 = sift.detectAndCompute(img1_undistort_gray, None)
    kpts2, descriptors2 = sift.detectAndCompute(img2_undistort_gray, None)
    # ----------

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 设置两距离比值小于0.7时为可用匹配  （Lowe's ratio test）
    valid_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            valid_matches.append(m)

    good_matches = []
    if len(valid_matches) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in valid_matches])
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in valid_matches])

        # RANSAC随机采样一致
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)

        # 将mask变成一维数组
        mask = mask.ravel().tolist()

        ## ----- mask out invalid match
        for match_i, match in enumerate(valid_matches):
            if mask[match_i] > 0:
                good_matches.append(match)

    else:
        print("not have enough matches!")

    good_matches = np.array(good_matches)

    # # 可视化特征点匹配
    # good_matches = np.expand_dims(good_matches, 0)
    # img_feat_match = cv2.drawMatchesKnn(img1_undistort, kpts1, img2_undistort, kpts2, good_matches[:15], None, flags=2)
    # win_name = 'Feature matching'
    # cv2.imshow(win_name, img_feat_match)
    # cv2.waitKey()
    # cv2.destroyWindow('Feature matching')

    good_matches = np.squeeze(good_matches)
    p1 = np.asarray([kpts1[m.queryIdx].pt for m in good_matches])  # queryIndex 查询图像中描述符的索引
    p2 = np.asarray([kpts2[m.trainIdx].pt for m in good_matches])
    # # -----

    # pts3d = reconFrom2Views(K1=camera_configs.left_camera_matrix,
    #                         K2=camera_configs.right_camera_matrix,
    #                         R1=np.eye(3, 3, dtype=np.float32),
    #                         T1=np.zeros((3, 1), dtype=np.float32),
    #                         R2=camera_configs.R,
    #                         T2=camera_configs.T,
    #                         p1=p1,
    #                         p2=p2)

    K1 = camera_configs.K1
    K2 = camera_configs.K2
    R1 = np.eye(3, 3, dtype=np.float32)
    T1 = np.zeros((3, 1), dtype=np.float32)
    R2 = camera_configs.R
    T2 = camera_configs.T

    P1 = np.zeros((3, 4), dtype=np.float32)
    P2 = np.zeros((3, 4), dtype=np.float32)

    P1[0:3, 0:3] = np.float32(R1)  # 将R1赋值给proj1的对应位置（前三行三列）
    P1[:, 3] = np.float32(T1.T)  # 将T1的转置赋值给proj1的对应位置（第四列）
    P2[0:3, 0:3] = np.float32(R2)
    P2[:, 3] = np.float32(T2.T)

    fk1 = np.float32(K1)
    fk2 = np.float32(K2)

    P1 = np.dot(fk1, P1)
    P2 = np.dot(fk2, P2)

    # ----- my implementation of triangulation
    pts3d, _ = my_triangulate(P1, P2, p1, p2)
    # -----

    print('Total {} 3D points reconstructed.'.format(pts3d.shape[0]))
    # print('Some Pts3D:\n', pts3d[:5])

    inds = np.where(pts3d[:, 2] > 0)
    good_3d_pts = pts3d[inds]
    good_3d_pts *= 0.001  # mm -> m
    print('Some good Pts3D(m):\n', good_3d_pts[:5], '\n Remain total {} good 3D points.'.format(len(good_3d_pts)))

    return good_3d_pts, p1, p2


def my_recon_bino():
    # 读取原图
    frame1 = cv2.imread("./img/01.bmp")
    frame2 = cv2.imread("./img/02.bmp")

    # 双目矫正
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 灰度图
    img1_rectified_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    img2_rectified_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 检测SIFT关键点
    # ---------- SIFT/SURF/AKAZE extractor
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.AKAZE_create()
    kpts1, descriptors1 = sift.detectAndCompute(img1_rectified_gray, None)
    kpts2, descriptors2 = sift.detectAndCompute(img2_rectified_gray, None)
    # ----------

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)  # NORM_L2, NORM_HAMMING, NORM_HAMMING2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 设置两距离比值小于0.7时为可用匹配  （Lowe's ratio test）
    valid_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            valid_matches.append(m)
    print('Total {:d} valid matches passed ratio test.'.format(len(valid_matches)))

    good_matches = []
    if len(valid_matches) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in valid_matches])
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in valid_matches])

        # RANSAC随机采样一致
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

        # 将mask变成一维数组
        mask = mask.ravel().tolist()

        ## ----- mask out invalid match
        for match_i, match in enumerate(valid_matches):
            if mask[match_i] > 0:
                good_matches.append(match)

    else:
        print("Not enough valid matches!")
        return
    # print('Total {:d} good matches.'.format(len(valid_matches)))
    print('Total {:d} good matches.'.format(len(good_matches)))

    # good_matches = np.array(valid_matches)
    good_matches = np.array(good_matches)

    # ## ---------- 可视化特征点匹配
    # good_matches = np.expand_dims(good_matches, 0)
    # img_feat_match = cv2.drawMatchesKnn(img1_rectified, kpts1, img2_rectified, kpts2, good_matches[:15], None, flags=2)
    # win_name = 'Feature matching'
    # cv2.imshow(win_name, img_feat_match)
    # cv2.waitKey()
    # cv2.destroyWindow('Feature matching')

    ## ---------- 根据匹配结果, 获取匹配到的2D特征点
    good_matches = np.squeeze(good_matches)
    p1 = np.asarray([kpts1[m.queryIdx].pt for m in good_matches])  # queryIndex 查询图像中描述符的索引
    p2 = np.asarray([kpts2[m.trainIdx].pt for m in good_matches])
    # # -----

    # pts3d = reconstructFromBino(P1=camera_configs.P1, P2=camera_configs.P2, p1=p1, p2=p2)
    pts3d, _ = my_triangulate(P1=camera_configs.P1, P2=camera_configs.P2, p1=p1, p2=p2)

    print('Total {} 3D points reconstructed.'.format(pts3d.shape[0]))
    # print('Some Pts3D:\n', pts3d[:5])

    inds = np.where(pts3d[:, 2] > 0)
    good_3d_pts = pts3d[inds]
    good_3d_pts *= 0.001  # mm -> m
    print('Some good Pts3D(m):\n', good_3d_pts[:5], '\n Remain total {} good 3D points.'.format(len(good_3d_pts)))

    ## ---------- 验证投影3D ——> 2D误差
    # pts3d = pts3d.tolist()
    err_sum_my, err_sum_cv = 0.0, 0.0
    for pt2d_1, pt2d_2, pt3d in zip(p1, p2, pts3d):
        ## ----- left frame's re-projection error
        err_my_1 = reproj_err_my(camera_configs.P1, pt3d, pt2d_1)
        err_cv_1 = reproj_err_cv(camera_configs.P1, pt3d, pt2d_1)

        ## ----- right frame's re-projection error
        err_my_2 = reproj_err_my(camera_configs.P2, pt3d, pt2d_2)
        err_cv_2 = reproj_err_cv(camera_configs.P2, pt3d, pt2d_2)

        err_mean_my = np.mean(np.array([err_my_1, err_my_2]))
        err_mean_cv = np.mean(np.array([err_cv_1, err_cv_2]))
        err_sum_my += err_mean_my
        err_sum_cv += err_mean_cv
        # print('Re-projection error(my): {:.3f} pixel'.format(err_mean_my))

    print('Mean re-projection error(my): {:6.4} pixel'.format(err_sum_my / float(len(pts3d))))
    print('Mean re-projection error(cv): {:6.4} pixel\n'.format(err_sum_cv / float(len(pts3d))))

    pts3d = np.array(pts3d)[:, :-1]

    return pts3d, p1, p2


def bino_recon():
    # 读取原图
    frame1 = cv2.imread("./img/01.bmp")
    frame2 = cv2.imread("./img/02.bmp")

    # 双目矫正
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 灰度图
    img1_rectified_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    img2_rectified_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 检测SIFT关键点
    # ---------- SIFT/SURF/AKAZE extractor
    # sift = cv2.xfeatures2d.SURF_create()
    sift = cv2.AKAZE_create()
    kpts1, descriptors1 = sift.detectAndCompute(img1_rectified_gray, None)
    kpts2, descriptors2 = sift.detectAndCompute(img2_rectified_gray, None)
    # ----------

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)  # NORM_HAMMING, NORM_HAMMING2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # 设置两距离比值小于0.7时为可用匹配  （Lowe's ratio test）
    valid_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            valid_matches.append(m)
    print('Total {:d} valid matches passed ratio test.'.format(len(valid_matches)))

    good_matches = []
    if len(valid_matches) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in valid_matches])
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in valid_matches])

        # RANSAC随机采样一致
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

        # 将mask变成一维数组
        mask = mask.ravel().tolist()

        ## ----- mask out invalid match
        for match_i, match in enumerate(valid_matches):
            if mask[match_i] > 0:
                good_matches.append(match)

    else:
        print("Not enough valid matches!")
        return
    # print('Total {:d} good matches.'.format(len(valid_matches)))
    print('Total {:d} good matches.'.format(len(good_matches)))

    # good_matches = np.array(valid_matches)
    good_matches = np.array(good_matches)

    # ## ---------- 可视化特征点匹配
    # good_matches = np.expand_dims(good_matches, 0)
    # img_feat_match = cv2.drawMatchesKnn(img1_rectified, kpts1, img2_rectified, kpts2, good_matches[:15], None, flags=2)
    # win_name = 'Feature matching'
    # cv2.imshow(win_name, img_feat_match)
    # cv2.waitKey()
    # cv2.destroyWindow('Feature matching')

    ## ---------- 根据匹配结果, 获取匹配到的2D特征点
    good_matches = np.squeeze(good_matches)
    p1 = np.asarray([kpts1[m.queryIdx].pt for m in good_matches])  # queryIndex 查询图像中描述符的索引
    p2 = np.asarray([kpts2[m.trainIdx].pt for m in good_matches])
    # # -----

    pts3d = reconstructFromBino(P1=camera_configs.P1, P2=camera_configs.P2, p1=p1, p2=p2)
    print('Total {} 3D points reconstructed.'.format(pts3d.shape[0]))
    # print('Some Pts3D:\n', pts3d[:5])

    inds = np.where(pts3d[:, 2] > 0)
    good_3d_pts = pts3d[inds]
    good_3d_pts *= 0.001  # mm -> m
    print('Some good Pts3D(m):\n', good_3d_pts[:5], '\n Remain total {} good 3D points.'
          .format(len(good_3d_pts)))

    ## ---------- 验证投影3D ——> 2D误差
    # pts3d = pts3d.tolist()
    err_sum_my, err_sum_cv = 0.0, 0.0
    for pt2d_1, pt2d_2, pt3d in zip(p1, p2, pts3d):
        ## ----- left frame's re-projection error
        err_my_1 = reproj_err_my(camera_configs.P1, pt3d, pt2d_1)
        err_cv_1 = reproj_err_cv(camera_configs.P1, pt3d, pt2d_1)

        ## ----- right frame's re-projection error
        err_my_2 = reproj_err_my(camera_configs.P2, pt3d, pt2d_2)
        err_cv_2 = reproj_err_cv(camera_configs.P2, pt3d, pt2d_2)

        err_mean_my = np.mean(np.array([err_my_1, err_my_2]))
        err_mean_cv = np.mean(np.array([err_cv_1, err_cv_2]))
        err_sum_my += err_mean_my
        err_sum_cv += err_mean_cv

        # print('Re-projection error(my): {:.3f} pixel'.format(err_mean_my))
        # print('Re-projection error(cv): {:.3f} pixel'.format(err_mean_cv))

    print('Mean re-projection error(my): {:6.4} pixel'.format(err_sum_my / float(len(pts3d))))
    print('Mean re-projection error(cv): {:6.4} pixel\n'.format(err_sum_cv / float(len(pts3d))))

    pts3d = np.array(pts3d)[:, :-1]
    return pts3d, p1, p2


def test_verify_P1P2():
    """
    Guess wrong!!!
    :return:
    """
    P1_guess = np.zeros((3, 4), dtype=np.float32)
    P2_guess = np.zeros((3, 4), dtype=np.float32)

    # 将R1赋值给proj1的对应位置（前三行三列）
    # 将T1的转置赋值给proj1的对应位置（第四列）
    P1_guess[0:3, 0:3] = np.float32(camera_configs.R1)  # R1
    P1_guess[:, 3] = np.zeros((3, 1), dtype=np.float32).T  # T1
    P2_guess[0:3, 0:3] = np.float32(camera_configs.R2)  # R2
    P2_guess[:, 3] = np.float32(camera_configs.T).T  # T2

    fk1 = np.float32(camera_configs.K1)
    fk2 = np.float32(camera_configs.K2)

    P1_guess = np.dot(fk1, P1_guess)
    P2_guess = np.dot(fk2, P2_guess)

    print('P1_guess:\n', P1_guess)
    print('P1:\n', camera_configs.P1)
    print('P2_guess:\n', P2_guess)
    print('P2:\n', camera_configs.P2)


def compare_two_recon_methods():
    """
    :return:
    """
    pts3d_bino, p1_bino, p2_bino = bino_recon()
    pts3d_2views, p1_2views, p2_2views = twoviews_recon()
    pts3d_bino_my, p1_bino_my, p2_bino_my = my_recon_bino()
    pts3d_2views_my, p1_2views_my, p2_2views_my = test_pose_from_feature_matching_for_bino()  # my_recon_twoviews()
    # pts3d_2views = pts3d_2views_my

    assert pts3d_bino.shape[0] == p1_bino.shape[0] == p2_bino.shape[0]
    assert pts3d_2views.shape[0] == p1_2views.shape[0] == p2_2views.shape[0]

    if pts3d_bino.shape[0] <= pts3d_2views.shape[0]:
        # for i, (pt2d_1_bino, pt2d_2_bino) in enumerate(zip(p1_bino, p2_bino)):
        #     for j, (pt2d_1_2views, pt2d_2views) in enumerate(zip(p1_2views, p2_2views)):
        #         pass

        for i, pt3d_bino in enumerate(pts3d_bino):
            min_dist_3d = float('inf')
            min_dist_id = -1
            min_dist_pt3d = np.zeros((1, 3), dtype=np.float32)
            for j, pt3d_2views in enumerate(pts3d_2views):
                dist_3d = math.sqrt(
                    (pt3d_bino[0] - pt3d_2views[0]) * (pt3d_bino[0] - pt3d_2views[0])
                    + (pt3d_bino[1] - pt3d_2views[1]) * (pt3d_bino[1] - pt3d_2views[1])
                    + (pt3d_bino[2] - pt3d_2views[2]) * (pt3d_bino[2] - pt3d_2views[2])
                )
                if dist_3d < min_dist_3d:
                    min_dist_3d = dist_3d
                    min_dist_id = j
                    min_dist_pt3d = pt3d_2views

            # 3D重建点最小距离(m)
            print('Min distance 3D(m): {:6.3f} | [{:6.3f}, {:6.3f}, {:6.3f}], [{:6.3f}, {:6.3f}, {:6.3f}]'
                  .format(min_dist_3d,
                          pt3d_bino[0], pt3d_bino[1], pt3d_bino[2],
                          min_dist_pt3d[0], min_dist_pt3d[1], min_dist_pt3d[2]))

            # 计算2D特征点最小距离(pixel)
            pt2d_1_bino, pt2d_2_bino = p1_bino[i], p2_bino[i]
            pt2d_1_2views, pt2d_2_2views = p1_2views[min_dist_id], p2_2views[min_dist_id]

            min_dist_2d_1 = math.sqrt(
                (pt2d_1_bino[0] - pt2d_1_2views[0]) * (pt2d_1_bino[0] - pt2d_1_2views[0])
                + (pt2d_1_bino[1] - pt2d_1_2views[1]) * (pt2d_1_bino[1] - pt2d_1_2views[1])
            )
            min_dist_2d_2 = math.sqrt(
                (pt2d_2_bino[0] - pt2d_2_2views[0]) * (pt2d_2_bino[0] - pt2d_2_2views[0])
                + (pt2d_2_bino[1] - pt2d_2_2views[1]) * (pt2d_2_bino[1] - pt2d_2_2views[1])
            )
            print('min_dist_2d_1(pixel): {:.3f}'.format(min_dist_2d_1))
            print('min_dist_2d_2(pixel): {:.3f}'.format(min_dist_2d_2))
    else:
        pass


def skew(a):
    """
    反对称矩阵: 对于任意向量v, 有a×v=Av
    :param a:
    :return: A
    """
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def test_pose_from_feature_matching_for_bino():
    """
    :return:
    """
    # 读取原图
    frame1 = cv2.imread("./img/01.bmp")  # "./img/01.bmp"
    frame2 = cv2.imread("./img/02.bmp")  # "./img/02.bmp"

    # 畸变矫正
    img1_undistort = cv2.undistort(frame1,
                                   camera_configs.K1,
                                   camera_configs.distort1)

    img2_undistort = cv2.undistort(frame2,
                                   camera_configs.K2,
                                   camera_configs.distort2)

    # # 选取左右视图关键点
    # p1 = np.array([[199, 74], [335, 74], [337, 257], [200, 259]], dtype=np.float32)
    # p2 = np.array([[148, 62], [295, 64], [294, 248], [148, 249]], dtype=np.float32)

    img1_undistort_gray = cv2.cvtColor(img1_undistort, cv2.COLOR_BGR2GRAY)
    img2_undistort_gray = cv2.cvtColor(img2_undistort, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 检测SIFT/SURF/AKAZE关键点
    # ---------- SIFT/SURF/AKAZE extractor
    # feature_extractor = cv2.xfeatures2d.SURF_create()
    feature_extractor = cv2.AKAZE_create()
    # feature_extractor = cv2.SIFT_create()
    kpts1, descriptors1 = feature_extractor.detectAndCompute(img1_undistort_gray, None)
    kpts2, descriptors2 = feature_extractor.detectAndCompute(img2_undistort_gray, None)
    # kpts1, descriptors1 = feature_extractor.detectAndCompute(img1_gray, None)
    # kpts2, descriptors2 = feature_extractor.detectAndCompute(img2_gray, None)
    # ----------

    # 创建FLANN对象
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(descriptors1, descriptors2, k=2)  # 匹配描述子 返回匹配的两点
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)  # NORM_L2, NORM_HAMMING, NORM_HAMMING2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches_rcp = bf.knnMatch(descriptors2, descriptors1, k=2)

    # 统计两帧图特征匹配的最小距离，和最大距离

    # 比例滤波器
    def ratio_test(matches):
        """
        :param matches:
        :return:
        """
        valid_matches = []
        min_dist, max_dist = np.inf, -np.inf
        for first, second in matches:  # top1 and top2
            if first.distance < min_dist:
                min_dist = first.distance
            if second.distance > max_dist:
                max_dist = second.distance

            if first.distance < 0.7 * second.distance:
                valid_matches.append(first)
        return  valid_matches, min_dist, max_dist

    # ----- Ratio test filter: 设置两距离比值小于0.7时为可用匹配(Lowe's ratio test)
    valid_matches, min_dist_1, max_dist_1 = ratio_test(matches)
    valid_matches_rcp, min_dist_2, max_dist_2 = ratio_test(matches_rcp)
    min_dist = min(min_dist_1, min_dist_2)
    max_dist = max(max_dist_1, max_dist_2)
    print('Min distance: {:.3f}'.format(min_dist))
    print('Max distance: {:.3f}\n'.format(max_dist))

    # ----- Reciprocity filter: 互惠滤波器滤波
    matches_rcp_filter = []
    for match_rcp in valid_matches_rcp:
        found = False
        for match in valid_matches:
            if match_rcp.queryIdx == match.trainIdx \
                    and match_rcp.trainIdx == match.queryIdx:
                matches_rcp_filter.append(match)
                found = True
                break
        if found:
            continue
    valid_matches = matches_rcp_filter

    # ----- Distance filter: 特征距离约束
    tmp_matches = []
    for match in valid_matches:
        if match.distance < 12.0 * min_dist:
            tmp_matches.append(match)
    valid_matches = tmp_matches

    # ----- 应用对极几何约束滤波
    if len(valid_matches) >= 10:
        # 获取匹配出来的点
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in valid_matches])
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in valid_matches])

        # RANSAC随机采样一致
        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

        # ----- mask out invalid match
        mask = mask.ravel().tolist()  # 将mask变成一维数组
        good_matches = [match for i, match in enumerate(valid_matches) if mask[i]]
        good_matches = np.array(good_matches)
    else:
        print("not have enough matches!")
        return

    # 可视化特征点匹配
    good_matches = np.expand_dims(good_matches, 0)
    img_feat_match = cv2.drawMatchesKnn(img1_undistort, kpts1, img2_undistort, kpts2, good_matches[:15], None, flags=2)
    win_name = 'Feature matching'
    cv2.imshow(win_name, img_feat_match)
    cv2.waitKey()
    cv2.destroyWindow('Feature matching')

    good_matches = np.squeeze(good_matches)
    p1 = np.asarray([kpts1[m.queryIdx].pt for m in good_matches])  # queryIndex 查询图像中描述符的索引
    p2 = np.asarray([kpts2[m.trainIdx].pt for m in good_matches])
    print('Total {:d} good matches of 2D feature points.'.format(p1.shape[0]))
    # # -----

    # ---------- 将2D特征点从像素坐标系投影到相机坐标系
    # 相机内参
    K1 = camera_configs.K1
    K2 = camera_configs.K2

    # 构造相机坐标系下归一化齐次坐标
    x1 = np.concatenate((p1, np.ones((p1.shape[0], 1), dtype=np.float32)), axis=1)
    x2 = np.concatenate((p2, np.ones((p2.shape[0], 1), dtype=np.float32)), axis=1)

    # print(x1[0].reshape(3, 1))

    # 归一化, 到相机归一化坐标系: normalized camera coordinate frame
    x1_cam = np.array([np.linalg.inv(K1).dot(x.reshape(3, 1)) for x in x1])
    x2_cam = np.array([np.linalg.inv(K2).dot(x.reshape(3, 1)) for x in x2])

    # ---------- 计算基础矩阵E
    E, mask = cv2.findFundamentalMat(x1_cam, x2_cam, cv2.FM_RANSAC)
    # print('E:\n', E)

    # ---------- 基础矩阵E分解, 得到4个解
    # 保证本质矩阵E的秩为2
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    mean_eigenvalue = (S[0] + S[1]) * 0.5
    # print('Mean eigenvalue: {:.3f}'.format(mean_eigenvalue))
    # E = np.dot(U, np.dot(np.diag([1.0, 1.0, 0]), V))  # S = [1, 1, 0]
    E = np.dot(U, np.dot(np.diag([mean_eigenvalue, mean_eigenvalue, 0]), V))  # S = [1, 1, 0]
    # print('E:\n', E)

    # 创建矩阵(Hartley)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    # print('W:\n', W)
    Z = skew([0, 0, -1])
    # Z = np.array([[0, -1, 0],
    #               [1, 0, 0],
    #               [0, 0, 0]])
    # print('Z:\n', Z)

    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float32)  # [R1|T1]

    # tmp_1 = U[:, 2]
    # tmp_2 = np.dot(U, np.dot(W, V))
    # tmp_3 = np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2]))
    ## ----------
    # MVG 2nd Page258: 从R,T的计算公式中可以看到R,T都有两种情况，组合起来R,T有4种组合方式。
    # P1 = [I | 0], P2 = [UWV | +u2] or [UWV | −u2] or [UWTV | +u2] or [UWTV | −u2]
    P2 = [
        np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
        np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
        np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
        np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T
    ]
    # print(P2)

    poses = [
        (np.dot(U, np.dot(W, V)), U[:, 2]),  # pose 0
        (np.dot(U, np.dot(W, V)), -U[:, 2]),  # pose 1
        (np.dot(U, np.dot(W.T, V)), U[:, 2]),  # pose 2
        (np.dot(U, np.dot(W.T, V)), -U[:, 2])  # pose 3
    ]
    poses = [(np.array(pose[0], dtype=np.float32), np.array(pose[1], dtype=np.float32)) for pose in poses]

    # ---------- 从4个解中选取正确的解:只有1个解的3D点位于2个相机前面
    ind = 0
    max_res = 0.0
    for i in range(4):
        ## 三角测量获取空间3D点坐标: 3D points
        pts3d, pts3d_homo = my_triangulate(np.dot(K1, P1), np.dot(K2, P2[i]), x1, x2)

        ## 空间3D点投影到归一化相机平面获取深度
        depth_1 = np.dot(P1, pts3d_homo.T)[2]
        depth_2 = np.dot(P2[i], pts3d_homo.T)[2]

        # pts3d_homo = cv2.triangulatePoints(P1, P2[i], p1.T, p2.T)  # 返回4×N
        # depth_1 = np.dot(P1, pts3d_homo)[2]
        # depth_2 = np.dot(P2[i], pts3d_homo)[2]

        sum_res = sum(depth_1 > 0.0) + sum(depth_2 > 0.0)
        if sum_res > max_res:
            max_res = sum_res
            ind = i
            in_front_inds = (depth_1 > 0.0) & (depth_2 > 0.0)

    print('The id {:d} pose is the correct pose.'.format(ind))

    # 获取位姿
    R, T = poses[ind]
    R, T = np.float32(R), np.float32(T)

    # 验证位姿——E = t^R*scale
    E_estimate = np.dot(skew(T), R)
    print('Original E:\n', E)
    print('E from t^R:\n', E_estimate)
    print('Estimated R:\n', R)
    print('Estimated T:\n', T)

    # ----- 取符合条件的2d特征点: 对应3D点在两个相机前面(深度值>0)
    p1 = p1[in_front_inds, :]
    p2 = p2[in_front_inds, :]

    # ---------- 计算正确的位姿[R|T]下的三角剖分
    # 构建相机投影矩阵
    P2_correct = P2[ind]
    P1 = np.dot(K1, P1)
    P2 = np.dot(K2, P2_correct)

    # ----- 三角剖分
    # pts3d, pts3d_homo = my_triangulate(P1, P2, x1, x2)
    pts3d_homo = cv2.triangulatePoints(P1, P2, p1.T, p2.T)  # 返回4×N
    pts3d = []
    for i in range(pts3d_homo.shape[1]):  # 列数表示计算出来空间点的个数 将三角化的结果进行处理得到"正常"的点坐标
        col = pts3d_homo[:, i]
        col = col / float(col[3])
        pts3d.append([col[0], col[1], col[2]])

    # ----- 根据对极几何约束过滤掉不符合精度要求的点(2D-3D)
    # 构造相机坐标系下归一化齐次坐标, 归一化到相机归一化坐标系: normalized camera coordinate frame
    x1 = np.concatenate((p1, np.ones((p1.shape[0], 1), dtype=np.float32)), axis=1)
    x2 = np.concatenate((p2, np.ones((p2.shape[0], 1), dtype=np.float32)), axis=1)
    x1_cam = np.array([np.linalg.inv(K1).dot(x.reshape(3, 1)) for x in x1])
    x2_cam = np.array([np.linalg.inv(K2).dot(x.reshape(3, 1)) for x in x2])

    # Epipolar constraint filter: 验证位姿——对极约束: 计算对极约束残差
    good_flags = [False for i in range(p1.shape[0])]
    for i, (pt2d_1_homo, pt2d_2_homo) in enumerate(zip(x1_cam, x2_cam)):
        ep_res = np.dot(np.dot(np.dot(pt2d_2_homo.T, skew(T)), R), pt2d_1_homo)
        if ep_res < 1e-5:
            good_flags[i] = True
            # print('Epi-polar constraint residual error: {:.6f}'.format(np.abs(np.squeeze(ep_res))))
        else:
            # print('Epi-polar constraint residual error is great!')
            pass
    good_flags = np.array(good_flags, dtype=np.bool)

    # ----- 取符合条件的2d, 3d特征点
    pts3d = np.array(pts3d, dtype=np.float32)[good_flags]
    p1 = p1[good_flags]
    p2 = p2[good_flags]
    # pts3d = pts3d[in_front_inds, :]
    print('Total {:d} 3D points in front of both cameras.'.format(pts3d.shape[0]))
    print(pts3d[:5])

    # 验证三角化点与特征点的重投影关系: 误差太大则使用BA优化
    ## ---------- 验证投影3D ——> 2D
    R1 = np.eye(3, dtype=np.float32)
    T1 = np.zeros((3, 1), dtype=np.float32)
    R2, T2 = poses[ind]

    # pts3d = pts3d.tolist()

    # err_sum_my = 0.0
    err_sum_cv = 0.0
    for pt2d_1, pt2d_2, pt3d in zip(p1, p2, pts3d):
        ## ----- left frame's re-projection error
        # err_my_1 = my_reproj_err(K1, R1, T1, pt3d, pt2d_1)
        err_cv_1 = cv_reproj_err(K1, R1, T1, pt3d, pt2d_1)

        ## ----- right frame's re-projection error
        # err_my_2 = my_reproj_err(K2, R2, T2, pt3d, pt2d_2)
        err_cv_2 = cv_reproj_err(K2, R2, T2, pt3d, pt2d_2)

        # err_mean_my = np.mean(np.array([err_my_1, err_my_2]))
        err_mean_cv = np.mean(np.array([err_cv_1, err_cv_2]))

        # print('Re-projection error(my): {:6.4f} pixel'.format(err_mean_my))
        # print('Re-projection error(cv): {:6.4f} pixel\n'.format(err_mean_cv))

        # err_sum_my += err_mean_my
        err_sum_cv += err_mean_cv

    # print('Mean re-projection error(my): {:6.4} pixel'.format(err_sum_my / float(len(pts3d))))
    print('Mean re-projection error(cv): {:6.4} pixel\n'.format(err_sum_cv / float(len(pts3d))))

    def res_func(all_params, n_pts, pts2d_2views, K_2views):
        """
        :param all_params:
        :param n_pts:
        :param pts2d_2views:
        :param K_2views:
        :return:
        """
        rots = all_params[:2 * 3].reshape((2, 3))  # 读取2个旋转向量
        mots = all_params[2 * 3: 2 * 6].reshape((2, 3))  # 读取2个平移向量
        pts3d = all_params[2 * 6:].reshape((n_pts, 3))  # 读取n_pts个空间点

        errs = []
        for view_i in range(2):  # residual error for 2 views
            rot = rots[view_i]
            mot = mots[view_i]

            for pt_j in range(n_pts):
                obj_p = pts3d[pt_j]
                img_p = pts2d_2views[view_i][pt_j]

                ## ---------- 3D ——> 2D
                est_p, J = cv2.projectPoints(obj_p, rot, mot, K_2views[view_i], np.array([]))
                est_p = est_p.reshape(2)

                # 观测 - 预测
                err = img_p - est_p
                errs.append(abs(err[0]))
                errs.append(abs(err[1]))

        return np.array(errs)

    def bundle_adjustment_sparsity(n_pts):
        """
        :param n_pts: n_obs = n_pts * 2(2 views)
        :return:
        """
        n_obs = n_pts * 2  # number of 2D-3D mappings(observations): 2 views
        m = n_obs * 2  # rows number: n_observations(mappings: 2 views here) * 2(x, y)
        n = 2 * 6 + n_pts * 3  # cols number:
        A = lil_matrix((m, n), dtype=np.int)

        # observations from 2 views
        obs_inds = np.arange(n_obs)

        # 3d pt index for each observation(2d feature point mapping to 3d point)
        pt3d_inds = np.zeros(n_obs, dtype=np.int)
        pt3d_inds[:n_pts] = np.arange(n_pts)
        pt3d_inds[n_pts: n_pts*2] = np.arange(n_pts)

        # view id for each pt2d feature point
        pt2d_view_inds = np.zeros(n_obs, dtype=np.int)
        pt2d_view_inds[:n_pts] = 0
        pt2d_view_inds[n_pts: n_pts * 2] = 1
        for i in range(6):  # camera pose: rotations and translations
            A[2 * obs_inds, pt2d_view_inds * 6 + i] = 1         # x row
            A[2 * obs_inds + 1, pt2d_view_inds * 6 + i] = 1     # y row

        for i in range(3):  # 3D points
            A[2 * obs_inds, 2 * 6 + pt3d_inds * 3 + i] = 1       # x row
            A[2 * obs_inds + 1, 2 * 6 + pt3d_inds * 3 + i] = 1   # y row

        return A

    # ---------- 尝试BA优化: 只优化空间点和位姿
    # 构建参数集合
    rot_vects = np.zeros((2, 3), dtype=np.float32)
    mot_vects = np.zeros((2, 3), dtype=np.float32)
    r1, _ = cv2.Rodrigues(R1)
    r2, _ = cv2.Rodrigues(R2)
    r1, r2 = r1.squeeze(), r2.squeeze()
    rot_vects[0] = r1
    rot_vects[1] = r2
    mot_vects[0] = T1.squeeze()
    mot_vects[1] = T2.squeeze()
    all_params = np.hstack((rot_vects.ravel(), mot_vects.ravel(), pts3d.ravel()))

    # 构建Jacob稀疏矩阵
    n_pts = pts3d.shape[0]
    A = bundle_adjustment_sparsity(n_pts)

    # 最小二乘优化
    pts2d_2views = []
    pts2d_2views.append(p1)
    pts2d_2views.append(p2)
    K_2views = []
    K_2views.append(np.array(K1, dtype=np.float32))
    K_2views.append(np.array(K2, dtype=np.float32))
    res = least_squares(res_func,  # 残差函数
                        all_params,  # 估计参数的初始值1d数组
                        jac_sparsity=A,  # jacob稀疏矩阵
                        verbose=2,
                        ftol=1e-9, xtol=1e-9,
                        x_scale='jac', method='trf', loss='linear',
                        args=(n_pts, pts2d_2views, K_2views))  # 残差函数参数

    # 更新相机位姿和空间点坐标
    new_params = res.x
    rots_ = new_params[:2*3].reshape((2, 3))
    mots_ = new_params[2*3:2*6].reshape((2, 3))
    pts3d_ = new_params[2*6:].reshape((-1, 3))
    R1_, _ = cv2.Rodrigues(rots_[0])  # 旋转向量 ——> 旋转矩阵
    R2_, _ = cv2.Rodrigues(rots_[1])
    T1_ = mots_[0]
    T2_ = mots_[1]
    print('R1:\n', R1)
    print('R1_:\n', R1_)

    print('T1:\n', T1)
    print('T1_:\n', T1_)

    print('R2:\n', R2)
    print('R2_:\n', R2_)

    print('T2:\n', T2)
    print('T2_:\n', T2_)

    # ----------

    # 重新计算重投影误差
    # err_sum_my = 0.0
    err_sum_cv = 0.0
    for pt2d_1, pt2d_2, pt3d in zip(p1, p2, pts3d_):
        ## ----- left frame's re-projection error
        # err_my_1 = my_reproj_err(K1, R1_, T1_, pt3d, pt2d_1)
        err_cv_1 = cv_reproj_err(K1, R1_, T1_, pt3d, pt2d_1)

        ## ----- right frame's re-projection error
        # err_my_2 = my_reproj_err(K2, R2_, T2_, pt3d, pt2d_2)
        err_cv_2 = cv_reproj_err(K2, R2_, T2_, pt3d, pt2d_2)

        # err_mean_my = np.mean(np.array([err_my_1, err_my_2]))
        err_mean_cv = np.mean(np.array([err_cv_1, err_cv_2]))

        # print('Re-projection error(my): {:6.4f} pixel'.format(err_mean_my))
        # print('Re-projection error(cv): {:6.4f} pixel\n'.format(err_mean_cv))

        # err_sum_my += err_mean_my
        err_sum_cv += err_mean_cv

    # print('Mean re-projection error(my): {:6.4} pixel'.format(err_sum_my / float(len(pts3d))))
    print('Mean re-projection error(cv): {:6.4} pixel\n'.format(err_sum_cv / float(len(pts3d))))

    pts3d_ = np.array(pts3d_)
    pts3d_ *= 0.001
    print(pts3d_[:15])
    print('Remain {:d} 3D points.'.format(pts3d_.shape[0]))
    return pts3d_, p1, p2


if __name__ == '__main__':
    # bino_recon()
    # twoviews_recon()
    # test_verify_P1P2()
    compare_two_recon_methods()
    # test_pose_from_feature_matching_for_bino()
