#-*- coding:utf-8 _*-
"""
计算各项参数指标
@Author  : Xiaoqi Cheng Yutao
@Time    : 2019/10/30 21:08
"""
import sys
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import numpy.random as r
import sklearn.metrics as m
import matplotlib.pyplot as plt


def ROC_AUC(LabelFlatten, OutputFlatten, ShowROC = False):
	fpr, tpr, th = m.roc_curve(LabelFlatten, OutputFlatten)
	AUC = m.auc(fpr, tpr)  # AUC其实就是ROC曲线下边的面积
	if ShowROC:
		plt.figure('ROC curve')
		plt.plot(fpr, tpr)
		plt.xlabel('fpr')
		plt.ylabel('tpr')
		# plt.show()
	return fpr, tpr, AUC

def PRC_AP_MF(LabelFlatten, OutputFlatten, ShowPRC = False):
	precision, recall, th = m.precision_recall_curve(LabelFlatten, OutputFlatten)
	F1ScoreS = 2 * (precision * recall) / ((precision + recall)+sys.float_info.min)
	MF = F1ScoreS[np.argmax(F1ScoreS)]  # Maximum F-measure at optimal dataset scale
	AP = m.average_precision_score(LabelFlatten, OutputFlatten)  # AP其实就是PR曲线下边的面积
	if ShowPRC:
		plt.figure('Precision recall curve')
		plt.plot(recall, precision)
		plt.ylim([0.0, 1.0])
		plt.xlabel('recall')
		plt.ylabel('precision')
		# plt.show()
	return recall, precision, MF, AP
