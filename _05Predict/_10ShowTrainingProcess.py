#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng Yutao
@Time    : 2019/12/20 22:03
"""
import os
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import matplotlib.pyplot as plt


SaveFolders = ['Unet_BCELoss_Adam_Weight_22']

for SaveFolder in SaveFolders:
	TrainLosses = []        # 相同学习率的loss
	ValidLosses = []
	with open(os.path.join(SaveFolder, 'log.txt'), 'r') as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if 'Train' in line:
				Epoch = float(line.strip().split('\t')[2].split(':')[1])
				Lr = float(line.strip().split('\t')[3].split(':')[1])
				Loss = float(line.strip().split('\t')[4].split(':')[1])
				if Loss<1:
					TrainLosses.append(np.array([Epoch, Lr, Loss]))

			elif 'Valid' in line:
				Epoch = float(line.strip().split('\t')[16].split(':')[1])
				Lr = float(line.strip().split('\t')[17].split(':')[1])
				Loss = float(line.strip().split('\t')[18].split(':')[1])
				if Loss<1:
					ValidLosses.append(np.array([Epoch, Lr, Loss]))

	TrainLosses = np.vstack(TrainLosses)
	ValidLosses = np.vstack(ValidLosses)
	Lrs = np.unique(TrainLosses[..., 1])
	Colors = ['r','g','b','y','c','m','k']

	# %% 根据Lrs对TrainLosses进行分割
	NewTrainLosses = []
	for Lr in Lrs:
		Indx = np.where(TrainLosses[:,1] == Lr)
		TrainLoss = TrainLosses[Indx, :][0]
		NewTrainLosses.append(TrainLoss)

	fig = plt.figure(SaveFolder)
	for TrainLoss in NewTrainLosses:
		plt.plot(TrainLoss[..., 0], TrainLoss[..., 2])
	plt.plot(ValidLosses[..., 0], ValidLosses[..., 2])
	# plt.xticks(np.arange(0,2500,500))
plt.show()





