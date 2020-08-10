import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class vgg19_Net(nn.Module):
    def __init__(self, in_img_rgb=3, in_img_size=64, out_class=1000, in_fc_size=25088):
        super(vgg19_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_img_rgb, out_channels=in_img_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_img_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_img_size, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.fc17 = nn.Sequential(
            nn.Linear(in_features=in_fc_size, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc18 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc19 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=out_class, bias=True)
        )

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                          self.conv8,
                          self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14, self.conv15,
                          self.conv16]

        self.fc_list = [self.fc17, self.fc18, self.fc19]

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        fc = x.view(x.size(0), -1)
        # 查看全连接层的参数：in_fc_size  的值
        # print("vgg19_model_fc:",fc.size(1))
        for fc_item in self.fc_list:
            fc = fc_item(fc)
        return fc

# model=models.vgg19(pretrained=True).features[:37]
model=models.vgg19(pretrained=True).features[:19]
print("vgg prepared!")
# device = torch.device('cpu')
# model = vgg19_Net()
# model=torch.load('./vgg19.pth')
# model=model.load_state_dict(torch.load('./vgg19.pth', map_location=device))
# model = torch.load()
model=model.eval()

def extract_feature(input):
    model.eval()  # 必须要有，不然会影响特征提取结果
    print("extract_feature begin!")
    # img = Image.open(imgpath)  # 读取图片
    # # img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    # tensor = transforms.ToTensor(img)  # 将图片转化成tensor

    # output=model.features(input)
    output=model(input)
    # output_npy=output.cpu().detach().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错

    return output # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]


# feature=torch.nn.Sequential(*list(model.children())[:])
# print(feature)
# print(model._modules.keys())
# print(model(torch.randn(100,3,224,224)))






