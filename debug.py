# import evaluate
import torchvision.transforms as transforms
import math
from PIL import Image
import numpy as np
import cv2
import torch
import os

def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


HR_address="/Users/liumugeng/Desktop/my SRCNN/myDIV2K/DIV2K_valid_HR"
train_HR_address="/Users/liumugeng/Desktop/my SRCNN/myDIV2K/DIV2K_valid_HR_result"
bicubi_address="/Users/liumugeng/Desktop/my SRCNN/myDIV2K/DIV2K_valid_LR_mildx4"

imagelist=sorted(os.listdir(train_HR_address))
HRlist=sorted(os.listdir(HR_address))
bicubilist=sorted(os.listdir(bicubi_address))

trans=transforms.ToTensor()
for i in range(len(imagelist)):
    if i<=1:
        continue
    HR = Image.open(HR_address+'/'+imagelist[i]).convert('RGB')
    train_HR = Image.open(train_HR_address+'/'+HRlist[i]).convert('RGB')
    # train_HR=torch.tensor(train_HR)
    bicubi = Image.open(bicubi_address+'/'+bicubilist[i]).convert('RGB')
    # bicubi=torch.tensor(bicubi)
    print("pic",i,":")
    print("HR:train ",psnr(np.asarray(HR),np.asarray(train_HR)))
    # print("HR:HR",psnr(np.asarray(HR),np.asarray(HR)))
    print("HR:bicubic",psnr(np.asarray(HR),np.asarray(bicubi)))

# HR=cv2.imread(HR_address)
# train_HR=cv2.imread(train_HR_address)
# bicubi=cv2.imread(bicubi_address)
#
# print("HR:train ",psnr(HR,train_HR))
# print("HR:HR",psnr(HR,HR))
# print("HR:bicubic",psnr(HR,bicubi))