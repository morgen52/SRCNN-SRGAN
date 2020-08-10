from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transform
import cut_train


class Imgdataset(Dataset):
    def __init__(self, path):
        super(Imgdataset, self).__init__()
        print("dataloader")
        self.data = []
        if os.path.exists(path):
            dir_list = sorted(os.listdir(path))
            LR_dir = 'DIV2K_train_LR_mildx4'  # !!!
            HR_dir = 'DIV2K_train_HR'
            LR_path = path + '/' + LR_dir
            HR_path = path + '/' + HR_dir
            if os.path.exists(LR_path) and os.path.exists(HR_path):
                LR_data = sorted(os.listdir(LR_path))
                HR_data = sorted(os.listdir(HR_path))
                self.data = [{'LR': LR_path + '/' + LR_data[i], 'HR': HR_path + '/' + HR_data[i]} for i in
                             range(len(LR_data))]
            else:
                raise FileNotFoundError('path doesnt exist!')

    def __getitem__(self, index):
        LR_path, HR_path = self.data[index]["LR"], self.data[index]["HR"]
        # tran = transform.ToTensor()
        LR_img = Image.open(LR_path).convert('RGB')
        HR_img = Image.open(HR_path).convert('RGB')
        LR_img, HR_img = cut_train.mycut(LR_img, HR_img)
        return LR_img, HR_img

    def __len__(self):
        return len(self.data)
