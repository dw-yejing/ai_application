import cv2
import numpy as np

img = cv2.imread("aaa.jpg")

width, height = 375, 525  # 所需图像大小

# 找K
pts1 = np.float32(
    [[655, 217], [964, 280], [501, 537], [845, 609]]
)  # 所需图像部分四个顶点的像素点坐标
pts2 = np.float32(
    [[0, 0], [width, 0], [0, height], [width, height]]
)  # 定义对应的像素点坐标
matrix_K = cv2.getPerspectiveTransform(
    pts1, pts2
)  # 使用getPerspectiveTransform()得到转换矩阵
img_K = cv2.warpPerspective(
    img, matrix_K, (width, height)
)  # 使用warpPerspective()进行透视变换

matrix_K_inv = np.linalg.inv(matrix_K)
img_K_inv = cv2.warpPerspective(img_K, matrix_K_inv, (img.shape[1], img.shape[0]))

# # 找Q
# pts3 = np.float32([[63, 325], [340, 279], [89, 634], [403, 573]])
# pts4 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix_Q = cv2.getPerspectiveTransform(pts3, pts4)
# img_Q = cv2.warpPerspective(img, matrix_Q, (width, height))

# # 找J
# pts5 = np.float32([[777, 107], [1019, 84], [842, 359], [1117, 332]])
# pts6 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix_J = cv2.getPerspectiveTransform(pts5, pts6)
# img_J = cv2.warpPerspective(img, matrix_J, (width, height))

# cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Original Image", img)
cv2.imshow("img K", img_K)
cv2.imshow("img K inv", img_K_inv)
# cv2.imshow("img Q", img_Q)
# cv2.imshow("img J", img_J)

cv2.waitKey(0)
