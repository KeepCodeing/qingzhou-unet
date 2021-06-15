#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng Yutao
@Time    : 2019/12/20 9:52
"""
import torch, os
from Timer import *
from _02PipeDatasetLoader import *
from _03Unet import *
from _21CalEvaluationIndicator import *
Device = torch.device("cuda:0")

# %% 载入数据、模型
# FolderPath = '/home/cxq/workspace2/2019.10.23PipeEdgeDetecion/2019.10.23LossFunctionTest/Test/Dataset'
FolderPath = 'E:\\Python\\2019.05.13 159YT\\2019.11.11SRoadSegmentation\\Dataset'
TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, 1)
# Unet_BCELoss_Adam
SaveFolder = 'Unet_BCELoss_Adam_Weight_22'
Unet = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast = True, ActivateFunLast = torch.sigmoid).to(Device)
Unet.load_state_dict(torch.load(os.path.join(SaveFolder, '0700.pt'), map_location = Device))

# %% 测试
Unet.eval()  # 训练模式
torch.set_grad_enabled(False)
OutputS = []        # 存储检测数据，用于指标计算
LabelS = []
for Iter, (Input, Label, SampleName) in enumerate(ValDataLoader):
	end = timer(8)
	print(SampleName)
	InputImg = Input.float().to(Device)
	OutputImg = Unet(InputImg)
	Output = OutputImg.cpu().numpy()[0]
	Label = Label.detach().cpu().numpy()[0]
	OutputS.append(Output)
	LabelS.append(Label)
	end('5555')
	# 生成效果图
	OutputImg = OutputImg.cpu().numpy()[0, 0]
	OutputImg = (OutputImg*255).astype(np.uint8)
	Input = Input.numpy()[0][0]
	Input = (Normalization(Input) * 255).astype(np.uint8)
	ResultImg = cv2.cvtColor(Input, cv2.COLOR_GRAY2RGB)
	ResultImg[...,2] = OutputImg
	# plt.imshow(Label[0])
	plt.show()
	cv2.imwrite(os.path.join(SaveFolder, SampleName[0] + '.png'), ResultImg)

# %% 计算指标
OutputFlatten = np.vstack(OutputS).ravel()
LabelFlatten = np.vstack(LabelS).ravel()
#%% ROC, AUC
fpr, tpr, AUC = ROC_AUC(LabelFlatten, OutputFlatten, ShowROC = True)
print('AUC:', AUC)
#%% POC, AP
recall, precision, MF, AP = PRC_AP_MF(LabelFlatten, OutputFlatten, ShowPRC = True)
print('MF:', MF)
print('AP:', AP)
plt.show()






