import os
import cv2
import numpy as np
import torch
from numpy import core

import logging
from Timer import *
from _02PipeDatasetLoader import *
from _03Unet import *
from _04Loss import *

if __name__ == '__main__':
    Unet = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to('cuda')

    model = torch.load('../_03Training/Output/0700.pt')  # 载入模型文件

    model.eval()
