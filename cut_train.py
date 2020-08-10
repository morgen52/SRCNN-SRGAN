import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transform
import os
from PIL import Image
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


cut_size=224 #vgg


def mycut(LR, HR):
    tran = transform.ToTensor()
    LR=tran(LR)
    HR=tran(HR)
    sp = LR.shape  # obtain the image shape
    # print("shape: ",LR.shape)
    random_size=torch.abs(torch.randn(2))
    sz1 = (random_size[0]*10000)%(sp[1]-cut_size)  # height(rows) of image
    sz2 = (random_size[1]*10000)%(sp[2]-cut_size)
    a = int(sz1)  # x start
    b = int(sz1+cut_size)  # x end
    c = int(sz2)  # y start
    d = int(sz2+cut_size)
    cropLR= LR[0:3,a:b,c:d]  # crop the image
    cropHR=HR[0:3,a:b,c:d]
    # print("cropLR.shape: ",cropLR.shape)
    # print("cropHR.shape: ",cropHR.shape)
    return cropLR,cropHR

