
from re import A
import cv2
import numpy as np
from time import sleep
import os
#read image
img = cv2.imread('Switch.png')

#rotate image in all directions and save  to file
# for i in range(0, 360, 90):
#     M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), i, 1)
#     dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
#     cv2.imwrite('./dk/' + str(i) + '.png', dst)
#     #randomly add transforms to image
#     for j in range(0, 10):
#         x = np.random.randint(0, img.shape[1])
#         y = np.random.randint(0, img.shape[0])
#         dst[y, x] = [0, 0, 0]
#         cv2.imwrite('./dk/' + str(i) + " "+ str(j) + '.png', dst)
#         cv2.imshow('img', dst)
        
#         #resize image to random sizes
#         for k in range(0, 10):
#             x = np.random.randint(0, img.shape[1])
#             y = np.random.randint(0, img.shape[0])
#             dst[y, x] = [0, 0, 0]
#             cv2.imwrite('./dk/' + str(i) + " "+ str(j) + " "+ str(k) + '.png', dst)
#             cv2.imshow('img', dst)
#             cv2.waitKey(1)
    
#resize image into smaller 90x90 sub images
# s= 0
# for i in range(0, img.shape[0], 90):
#     for j in range(0, img.shape[1], 90):
#         s +=1
#         dst = img[i:i+90, j:j+90]
#         cv2.imwrite('./back/' + str(s) + '.png', dst)
s=50
folder = './back/'
for i in range(47):
        img = cv2.imread(folder + str(i+1) + '.png')
        #rotate image in all directions and save to file
        for i in range(0, 360, 90):
            s+=1
            M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), i, 1)
            dst = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            cv2.imwrite('./back/' + str(s) + '.png', dst)
            