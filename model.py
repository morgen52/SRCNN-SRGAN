import torch.nn as nn


class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1,padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1,padding=5//2)

    def forward(self, data):
        out = self.conv1(data)
        out = self.conv2(out)
        out = self.conv3(out)
        return out