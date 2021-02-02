import numpy as np
import cv2
import camera_configs

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)


# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        print(threeD[y][x])


cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    # ret1, frame1 = camera1.read()
    # ret2, frame2 = camera2.read()
    # if not ret1 or not ret2:
    #     break

    frame1 = cv2.imread("./img/01.bmp")
    frame2 = cv2.imread("./img/02.bmp")
    cv2.imshow("frame1", frame1)
    cv2.imshow("frame2", frame2)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    cv2.imshow("rect1", img1_rectified)
    cv2.imshow("rect2", img2_rectified)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray1", imgL)
    cv2.imshow("gray2", imgR)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

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
    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
    # stereo = cv2.StereoSGBM_create(numDisparities=32 * num, blockSize=block_size)
    disparity = stereo.compute(imgL, imgR)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # ----- 将图片扩展至3d空间中，其z方向的值则为当前的距离
    # 生成3D点云
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32), camera_configs.Q)
    # print(threeD)
    inds = np.where(threeD[:, :, 2] > 0)
    good_threeD = threeD[inds]

    # 单位转换：mm ——> m
    pts3d = good_threeD * 0.001

    # ----- 测距: 随机输出一些距离
    num_pts = pts3d.shape[0]
    rand_pt_inds = [np.random.randint(int(0.3*num_pts), int(0.7*num_pts)) for i in range(5)]
    dists = pts3d[rand_pt_inds][:, 2]
    print('Measured distance: {:.5f}m'.format(dists[np.random.randint(0, 5)]))

    cv2.imshow("pic1", img1_rectified)
    cv2.imshow("pic2", img2_rectified)
    cv2.imshow("disparity", disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    elif key == ord("s"):
        cv2.imwrite("./snapshot/BM_left.jpg", imgL)
        cv2.imwrite("./snapshot/BM_right.jpg", imgR)
        cv2.imwrite("./snapshot/BM_depth.jpg", disp)

cv2.destroyAllWindows()
