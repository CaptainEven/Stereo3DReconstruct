import cv2
import numpy as np

import camera_configs

# cv2.namedWindow("left")
# cv2.namedWindow("right")
# cv2.namedWindow("depth")
# cv2.moveWindow("left", 0, 0)
# cv2.moveWindow("right", 600, 0)
# cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
# cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
#
#
# # 添加点击事件，打印当前点的距离
# def callbackFunc(e, x, y, f, p):
#     if e == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)
#         print(threeD[y][x])
#
#
# cv2.setMouseCallback("depth", callbackFunc, None)
#
# while True:
#     # ret1, frame1 = camera1.read()
#     # ret2, frame2 = camera2.read()
#     # if not ret1 or not ret2:
#     #     break
#
#     frame1 = cv2.imread("./img/left_10.png")  # "./img/01.bmp"
#     frame2 = cv2.imread("./img/right_10.png")  # "./img/02.bmp"
#     cv2.imshow("frame1", frame1)
#     cv2.imshow("frame2", frame2)
#     # cv2.waitKey(1000)
#     # cv2.destroyAllWindows()
#
#     # 根据更正map对图片进行重构
#     # img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
#     # img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
#     img1_rectified = frame1
#     img2_rectified = frame2
#     cv2.imshow("rect1", img1_rectified)
#     cv2.imshow("rect2", img2_rectified)
#     # cv2.waitKey(1000)
#     # cv2.destroyAllWindows()
#
#     # 将图片置为灰度图，为StereoBM作准备
#     imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
#     imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("gray1", imgL)
#     cv2.imshow("gray2", imgR)
#     # cv2.waitKey(1000)
#     # cv2.destroyAllWindows()
#     img_channels = 1
#
#     # 两个track bar用来调节不同的参数查看效果
#     num = cv2.getTrackbarPos("num", "depth")
#     num = num if num > 0 else 1
#     block_size = cv2.getTrackbarPos("blockSize", "depth")
#     if block_size % 2 == 0:
#         block_size += 1  # odd block size
#     if block_size < 5:
#         block_size = 5
#     # print('Num: ', num)
#     # print('Block size: ', block_size)
#
#     # 根据Block Matching方法生成视差图(opencv里也提供了SGBM/Semi-Global Block Matching算法)
#     num_disp = 16 * num
#     num_disp = ((imgL.shape[1] // 8) + 15) & -16;
#     # stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
#     stereo = cv2.StereoSGBM_create(numDisparities=num_disp,
#                                    blockSize=block_size,
#                                    disp12MaxDiff=-1,
#                                    preFilterCap=1,
#                                    P1=8 * img_channels * block_size * block_size,
#                                    P2=32 * img_channels * block_size * block_size,
#                                    uniquenessRatio=10,
#                                    speckleWindowSize=100,
#                                    speckleRange=100,
#                                    mode=cv2.STEREO_SGBM_MODE_HH)
#     disparity = stereo.compute(imgL, imgR)
#     disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     # cv2.validateDisparity()
#
#     # ----- 将图片扩展至3d空间中，其z方向的值则为当前的距离
#     # 生成3D点云
#     threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32), camera_configs.Q)
#     # print(threeD)
#     inds = np.where(threeD[:, :, 2] > 0)
#     good_threeD = threeD[inds]
#
#     # 单位转换：mm ——> m
#     pts3d = good_threeD * 0.001
#
#     # ----- 测距: 随机输出一些距离
#     num_pts = pts3d.shape[0]
#     rand_pt_inds = [np.random.randint(int(0.3 * num_pts), int(0.7 * num_pts)) for i in range(5)]
#     dists = pts3d[rand_pt_inds][:, 2]
#     print('Measured distance: {:.5f}m'.format(dists[np.random.randint(0, 5)]))
#
#     cv2.imshow("pic1", img1_rectified)
#     cv2.imshow("pic2", img2_rectified)
#     cv2.imshow("disparity", disp)
#
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break
#
#     elif key == ord("s"):
#         cv2.imwrite("./snapshot/BM_left.jpg", imgL)
#         cv2.imwrite("./snapshot/BM_right.jpg", imgR)
#         cv2.imwrite("./snapshot/BM_depth.jpg", disp)
#
# cv2.destroyAllWindows()


def points2pcd(points, PCD_FILE_PATH):
    """
    :param points:
    :param PCD_FILE_PATH:
    :return:
    """
    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部(重要)
    handle.write('# .PCD v0.7 - Point Cloud Data file format\n'
                 'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + \
                 str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)

    handle.close()


def points2ply(points, colors, ply_f_path):
    """
    :param points:
    :param colors:
    :param ply_f_path:
    :return:
    """
    # 读取三维点坐标和颜色信息
    points = np.hstack([points.reshape(-1, 3), colors.reshape(-1, 3)])
    # 必须先写入, 然后利用write()在头部插入ply header
    np.savetxt(ply_f_path, points, fmt='%f %f %f %d %d %d')

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    \n
    '''

    with open(ply_f_path, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(points)))
        f.write(old)


def disp2depth(b, f, disp):
    """
    :param b:
    :param f:
    :param disp:
    :return:
    """
    disp = disp.astype(np.float32)
    non_zero_inds = np.where(disp)

    depth = np.zeros_like(disp, dtype=np.float32)
    depth[non_zero_inds] = b * f / disp[non_zero_inds]

    return depth


def test_kitti():
    """
    :return:
    """
    left = cv2.imread("./img/left_200.jpg")  # "./img/01.bmp"
    right = cv2.imread("./img/right_200.jpg")  # "./img/02.bmp"

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray1", imgL)
    # cv2.imshow("gray2", imgR)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    img_channels = 1

    # 两个track bar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    num = num if num > 0 else 1
    block_size = cv2.getTrackbarPos("blockSize", "depth")
    if block_size % 2 == 0:
        block_size += 1  # odd block size
    if block_size < 5:
        block_size = 5
    # print('Num: ', num)
    # print('Block size: ', block_size)

    # 根据Block Matching方法生成视差图(opencv里也提供了SGBM/Semi-Global Block Matching算法)
    num_disp = 16 * num
    num_disp = ((imgL.shape[1] // 8) + 15) & -16;
    # stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    stereo = cv2.StereoSGBM_create(numDisparities=num_disp,
                                   blockSize=block_size,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   P1=8 * img_channels * block_size * block_size,
                                   P2=32 * img_channels * block_size * block_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    disparity = stereo.compute(imgL, imgR)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.validateDisparity()
    # cv2.waitKey()

    ## 超参数: 用于点云截取
    MAX_DEPTH = 80.0
    MAX_HEIGHT = 1.5

    # KITTI数据集参数
    b = 0.54  # m
    f = 718.335  # pixel
    cx = 609.5593  # pixel
    cy = 172.8540  # pixel

    # ## xiaomi参数
    # f = (998.72290039062500 + 1000.0239868164063) * 0.5  # 1000.0
    # cx = 671.15643310546875
    # cy = 384.32458496093750
    # b = 0.12  # m

    H, W = disparity.shape[:2]
    print('W×H: {:d}×{:d}'.format(W, H))
    cx, cy = W * 0.5, H * 0.5
    c, r = np.meshgrid(np.arange(W), np.arange(H))
    # print(c, '\n', r)
    # x, y = np.arange(W), np.arange(H)

    # ---------- 视差图(uint16)——>深度图(float32)
    depth = disp2depth(b, f, disparity)

    # ---------- 深度图滤波
    mask = depth > 0.0
    depth = depth * mask
    mask = depth < MAX_DEPTH
    depth = depth * mask
    print('Max depth: {:.3f}m.'.format(np.max(depth)))

    # --------- 深度图——>点云x, y, z
    points = np.zeros((H, W, 3), dtype=np.float32)
    colors = np.zeros((H, W, 3), dtype=np.uint8)
    points[r, c, 0] = (c - cx) * depth / f  # x
    points[r, c, 1] = (r - cy) * depth / f  # y
    points[r, c, 2] = depth  # z

    # bgr ——> rgb
    colors = left[:, :, ::-1]

    # ----- 过滤掉深度值<=0的点
    inds = np.where(points[:, :, 2] > 0.0)
    points = points[inds]
    colors = colors[inds]

    # ----- 过滤掉x, y, z全为0的点
    inds = np.where((points[:, 0] != 0.0) |
                    (points[:, 1] != 0.0) |
                    (points[:, 2] != 0.0))
    points = points[inds]
    colors = colors[inds]

    # ----- 过滤掉
    inds = np.where(
        (points[:, 1] < MAX_HEIGHT)
        & (points[:, 1] > -MAX_HEIGHT)
    )
    points = points[inds]
    colors = colors[inds]
    print('{:d} 3D points left.'.format(inds[0].size))

    # 保存pcd点云文件
    pc_path = './pc_200.pcd'
    points2pcd(points, pc_path)
    print('PCD poind cloud {:s} saved.'.format(pc_path))

    # 保存ply点云文件
    ply_path = './ply_200.ply'
    points2ply(points, colors, ply_path)
    print('Ply poind cloud {:s} saved.'.format(ply_path))


if __name__ == '__main__':
    test_kitti()