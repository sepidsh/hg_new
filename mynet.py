import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
             nn.MaxPool2d(2),
             double_conv(in_ch, out_ch)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class MyNet(nn.Module):
    def __init__(self, n_channels=4):
        super(MyNet, self).__init__()
        base = 16
        self.conv1 = down(11,64)
        self.conv2 = down(64,128)
        self.conv3 = down(128,256)
        self.conv4 = down(256,256)
        self.conv5 = down(256,1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 =  nn.Linear(1024, 1024)
        self.fc3 =  nn.Linear(1024, 100)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.squeeze()
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
