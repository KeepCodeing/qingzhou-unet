#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng Yutao
@Time    : 2019/10/25 20:50
"""
import numpy as np
import torch.nn as nn
import torch

class DiceLoss(nn.Module):
	'''Dice系数 https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient'''
	def __init__(self):
		super(DiceLoss, self).__init__()
		self.smooth = 1.0

	def forward(self, y_pred, y_true):  # BS*1*256*256
		assert y_pred.size() == y_true.size()
		y_pred = y_pred[:, 0].contiguous().view(-1)  # 将变量展开，view需要tensor的内存是整块的
		y_true = y_true[:, 0].contiguous().view(-1)
		intersection = (y_pred * y_true).sum()      # 正样本的交集（即，mask为1的部分交集）
		dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
		return 1. - dsc

