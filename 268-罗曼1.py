from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2                                                      #cv2是BGR的顺序

#灰度化
# img = cv2.imread('123.png')                                     #图片不写路径，要放在和当前.py同一个路径下
# h,w = img.shape[:2]                                             #获取图片的h/w
# img_gray = np.zeros([h,w],img.dtype)                            #创建一张和当前大小一样的单通道图片
# for i in range(h):
#     for n in range(w):
#         m = img[i,n]                                            #取出当前h/w中的bgr坐标
#         img_gray[i,n] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)       #将bgr坐标转化为gray坐标并赋值给新图像
# print(img_gray)
# print('image show gray: %s'%img_gray)
# cv2.imshow('image show gray',img_gray)

plt.subplot(221)
img = plt.imread('6468.png')                                       #读取
# img = cv2.imread('abc.png'.false)
# img = '6468.png'
plt.imshow(img)                                                    #展示
print("---image 6468---")
print(img)

img_gray = rgb2gray(img)                                            #rgb转灰度
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                     #bgr转灰度
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray,cmap = 'gray')
print("---image gray---")
print(img_gray)

#二值化
img_2 = np.where(img >= 0.5,1,0)
print(img_2)
print(img_2.shape)