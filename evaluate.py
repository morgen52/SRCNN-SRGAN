import torch
import os
import cv2
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import matplotlib as plt

valid_HR_address="./myDIV2K/DIV2K_valid_HR"
valid_LR_address="./myDIV2K/DIV2K_valid_LR_mild"
# save_valid_LR_address="./myDIV2K/DIV2K_valid_LR_mildx4"
# save_result_HR_address="./myDIV2K/DIV2K_valid_HR_vggresult_2"
save_valid_LR_address="./myDIV2K/DIV2K_valid_LR_mildx4"
save_result_HR_address="./myDIV2K/DIV2K_valid_HR_vggresult_2"

def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

network = torch.load('./gpuvggnetwork_1.pth',map_location=torch.device('cpu'))
imagelist=sorted(os.listdir(save_valid_LR_address))
HRlist=sorted(os.listdir(valid_HR_address))
trans=transforms.ToTensor()

for i in range(len(imagelist)):
    valid=Image.open(save_valid_LR_address+'/'+imagelist[i]).convert('RGB')
    HR = Image.open(valid_HR_address + '/' +HRlist[i]).convert('RGB')
    # print(cv2.imread(valid_HR_address + '/' +HRlist[i]))
    HR=trans(HR)
    valid=trans(valid).unsqueeze(dim=0)
    pred=network(valid).squeeze(dim=0)

    # print(isinstance(pred, np.ndarray))#判断图像数据是否是OpenCV格式
    # pred = cv2.cvtColor(np.array(pred), cv2.COLOR_RGB2BGR)#PIL.Image转换成OpenCV格式
    # HR = cv2.cvtColor(np.array(HR),cv2.COLOR_RGB2BGR)
    # print("picture[",i,"].psnr:",psnr(pred.detach().numpy().astype(np.uint8),HR.numpy().astype(np.uint8)))

    unloader=transforms.ToPILImage()
    pred = unloader(pred).convert('RGB')
    pred.save(save_result_HR_address+'/'+HRlist[i])


    pred=np.array(pred)#保证像素点在0到255之间
    pred*=(pred >= 0)
    pred=pred*(pred<=255)+255*(pred>255)
    pred=pred.astype(np.uint8)
    # pred=Image.fromarray(pred)
    plt.image.imsave(save_result_HR_address+'/'+HRlist[i], pred)

    pred=cv2.imread(save_result_HR_address+'/'+HRlist[i])
    HR=cv2.imread(valid_HR_address+'/'+HRlist[i])
    print("picture[",i,"].psnr:",psnr(pred,HR))
    # print("picture[",i,"].psnr:",psnr(np.asarray(pred), np.asarray(HR)))
