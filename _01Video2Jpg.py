# 导入所需要的库
import cv2
import numpy as np

# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image,addr,num):
    num = '%04d' % num
    address = addr + num + '.jpg'
    cv2.imwrite(address,image)

if __name__ == '__main__':
    # 读取视频文件
    videoCapture = cv2.VideoCapture("./video.mp4")

    #读帧
    success, frame = videoCapture.read()
    i = 0
    timeF = 12
    j = 0
    print(success)
    while success:
        i = i + 1
        if (i % timeF == 0):
            j = j + 1
            save_image(frame,'./OriImage/', j)
            print('save image:', j)
        success, frame = videoCapture.read()
