# -*- coding:utf-8 _*-
"""
@Author  : Yu Cheng
@Time    : 2019/12/25 9:35
"""
import glob, cv2, os
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)

ImgPaths = glob.glob("./OriImage/*.jpg")
H = np.array([[  -0.61869793,   -2.24344654,  672.50410257],
              [   0.00583877,    0.05218149, -226.93620917],
              [  -0.00011433,   -0.00451613,    1.        ]])

# 相机参数
Dist = np.array([-0.26538, 0.08153, -0.00109, -0.00233, 0.00000], dtype=np.float32)
K = np.array([[331.71415, 0, 321.54719],
              [0, 331.80738, 201.23948],
              [0, 0, 1]], dtype=np.float32)

for ImgPath in ImgPaths:
	print(ImgPath)
	Img = cv2.imread(ImgPath)
	UndistImg = cv2.undistort(Img, K, Dist)
	WarpedImg = cv2.warpPerspective(UndistImg, H, (1000, 1000))
	# plt.imshow(WarpedImg)
	# 	# plt.show()
	SavePath = ImgPath.replace('OriImage', 'WarpedImg')
	os.makedirs(os.path.dirname(SavePath), exist_ok=True)
	cv2.imwrite(SavePath, WarpedImg)
