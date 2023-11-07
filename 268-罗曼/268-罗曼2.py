#最邻近插值法
import cv2
import numpy as np

# def function(img):
#     hight,wight,channels = img.shape            #导入H\W\C通道,返回图像尺寸
#     emptyImage = np.zeros((800,800,channels),np.uint8)          #空图像
#     sh = 800/hight
#     sw = 800/wight
#     for i in range(800):
#         for j in range(800):
#             x = int(i/sh+0.5)
#             y = int(j/sw+0.5)
#             emptyImage[i,j] = img[x,y]
#     return emptyImage
#
# img = cv2.imread('1.jpeg')
# z = function(img)
# print(z)
# print(z.shape)
# print('*************************************')

#双线插值
# def bilinear_interpolation(img,out_dim):
#     src_h,src_w,channle = img.shape                      #原图尺寸
#     dst_h,dst_w = out_dim[1],out_dim[0]                  #目标尺寸
#     if src_h == dst_h and src_w == dst_w :               #如果一样直接复制
#         return img.copy()
#     dst_emp = np.zeros((dst_h,dst_w,3),dtype=np.uint8)       #空白矩阵
#     y,x = float(src_h)/dst_h,float(src_w)/dst_w             #比例关系
#     for i in range(3):
#         for dst_x in range(dst_w):
#             for dst_y in range(dst_h):
#                 src_x = ((dst_x)+0.5)*x-0.5                 #中心对称
#                 src_y = ((dst_y)+0.5)*y-0.5
#
#                 src_x0 = int(np.floor(src_x))               #防呆检查   #向下取整
#                 src_x1 = min(src_x0+1,src_w-1)                        #最小值
#                 src_y0 = int(np.floor(src_y))
#                 src_y1 = min(src_y0+1,src_h-1)
#
#                 fy0 = (src_x1-src_x)*img[src_y0,src_x0,i]+(src_x-src_x1)*img[src_y0,src_x1,i]
#                 fy1 = (src_x1-src_x)*img[src_y1,src_x0,i]+(src_x-src_x1)*img[src_y1,src_x1,i]
#                 dst_emp[dst_y,dst_x,i] = int((src_y1-src_y)*fy0+(src_y-src_y0)*fy1)
#
#     return dst_emp
#
#
# if __name__ == '__main__':
#     img = cv2.imread('1.jpeg')
#     dst = bilinear_interpolation(img,(700,700))
#     cv2.imshow('bilinear_interp',dst)
#     cv2.waitKey()

import cv2
import numpy as np
from matplotlib import pyplot as plt

#获取灰度图像
# img = cv2.imread('1.jpeg',1)
# g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# dst = cv2.equalizeHist(g)                               #
# hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#
# plt.figure()
# plt.hist(dst.ravel(),256)
# plt.show()
# cv2.imshow("Histogram Equalization",np.hstack([g,dst]))
# cv2.waitKey()

#彩色
img = cv2.imread('1.jpeg',1)
cv2.imshow('src',img)

(b,g,r) = cv2.split(img)        #split拆分通道的函数
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
c = [bh,gh,rh]

result = cv2.merge(c)           #merge合并函数

cv2.imshow('dst_bgr',result)
cv2.waitKey()
