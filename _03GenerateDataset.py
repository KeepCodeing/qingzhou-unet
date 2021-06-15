# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2019/12/11 9:35
"""
import os,glob
LabelPaths = glob.glob('./OriLabelDataset/*.json')

for LabelPath in LabelPaths:
	print(LabelPath)
	Name = os.path.basename(LabelPath).split('.')[0]
	cmd = 'labelme_json_to_dataset {0} -o {1}'.format(LabelPath, Name)
	os.system(cmd)