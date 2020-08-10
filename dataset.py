import os
import numpy as np
import glob
from bicubic import bcb

train_HR_address = "./DIV2K/DIV2K_train_HR"
train_LR_address="./myDIV2K/DIV2K_valid_LR_mild"
save_train_LR_address="./myDIV2K/DIV2K_valid_LR_mildx4"


def read_LR(directory_name,save_directory_name):
    # imagelist = sorted(glob.glob(directory_name + '/' + '*.png'))  # 读取带有相同关键字的图片名字
    imagelist=sorted(os.listdir(directory_name))
    for filename in imagelist:
        # img = cv2.imread(filename)
        # img=img/255.
        # img_array = np.array(img)
        # pic = np.reshape(img_array, (150, 150, 3)).astype(np.uint8)
        # plt.imshow(pic)
        # plt.show()
        print(filename)
        # if filename<="0755x4m.png":
        #     continue
        if filename==".DS_Store":
            continue
        bcb.function(bcb,4,directory_name+'/'+filename,save_directory_name+'/'+filename)
        # image3 = BiCubic_interpolation(image, image.shape[0] * k, image.shape[1] * k) #双三次插值
        # image3 = Image.fromarray(image3.astype('uint8')).convert('RGB')
        # image3.save('./4.png')
        

read_LR(train_LR_address,save_train_LR_address)

