import cv2
import numpy as np

img = cv2.imread("mask_images/mask.jpg")
# 将图片分割为三个通道
b_channel, g_channel, r_channel = cv2.split(img)
alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (3, 3), cv2.BORDER_DEFAULT)
canny_img = cv2.Canny(blur_img, 50, 150)
contours, hierarchy = cv2.findContours(
    canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
# 获取面积最大的轮廓
contour = max(contours, key=cv2.contourArea)
# 获取最大轮廓的凸包
hull = cv2.convexHull(contour)
# 初始化一个alpha通道
alpha_channel = cv2.drawContours(alpha_channel, [hull], -1, 255, -1)
png_img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
# 获取最大轮廓的最小外接矩形
rect = cv2.minAreaRect(contour)
# 获取矩形四个顶点的坐标
box = cv2.boxPoints(rect)
# 将矩形转换为水平的矩形
x_min = int(np.min(box[:, 0]))
y_min = int(np.min(box[:, 1]))
x_max = int(np.max(box[:, 0]))
y_max = int(np.max(box[:, 1]))
png_img = png_img[y_min:y_max, x_min:x_max, :]
cv2.imwrite("mask_images/mask.png", png_img)
