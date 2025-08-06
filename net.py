from torch import nn
from torch.nn import functional as F
import torch

class Conv_Block(nn.Module):
    def __init__(self, in_channel,out_channel):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=1,padding=1,padding_mode="reflect",bias=False),
            #padding_mode="reflect"用数据内部的镜像值填充边，能让填充更贴合原始数据的特征，常见于对边缘敏感的深度学习任务,如图像分割
            nn.BatchNorm2d(out_channel),
            #每层的输入进行归一化，强制让输入的分布更稳定（均值接近 0，方差接近 1）
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,padding_mode="reflect",bias=False),
            #后续接 BatchNorm 层时，会对输入进行归一化。此时卷积层的偏置会被BatchNorm的归一化操作 “抵消”，变得冗余，因此设置bias=False
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x=self.layers(x)
        return x

#下采样时用带步长的卷积替代池化，能在降维的同时保留细节信息
class DownSample(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        x=self.layer(x)
        return x
    


class UpSample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # # 转置卷积实现2倍上采样 + 通道从 channel → channel//2
        # self.trans_conv = nn.ConvTranspose2d(
        #     in_channels=channel,        # 输入通道数
        #     out_channels=channel // 2,  # 输出通道数（减半）
        #     kernel_size=2,             # 卷积核大小
        #     stride=2,                  # 步长=2 → 实现2倍上采样
        #     padding=0,                 # padding=0（配合kernel_size=2、stride=2实现精确上采样）
        # )
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    
    def forward(self, x, feature_map):
        # # 1. 转置卷积：同时完成 2倍上采样 + 通道调整（channel → channel//2）
        # up = self.trans_conv(x)
        # # 2. 拼接：与feature_map在通道维度（dim=1）拼接
        # return torch.cat((up, feature_map), dim=1)
        up=F.interpolate(x,scale_factor=2,mode='nearest')   #最近邻插值法对特征图放大
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)
    
class UNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1=Conv_Block(3,64)
        self.down1=DownSample(64)
        self.conv2=Conv_Block(64,128)
        self.down2=DownSample(128)
        self.conv3=Conv_Block(128,256)
        self.down3=DownSample(256)
        self.conv4=Conv_Block(256,512)
        self.down4=DownSample(512)
        self.conv5=Conv_Block(512,1024)
        self.up1=UpSample(1024)
        self.conv6=Conv_Block(1024,512)
        self.up2=UpSample(512)
        self.conv7=Conv_Block(512,256)
        self.up3=UpSample(256)
        self.conv8=Conv_Block(256,128)
        self.up4=UpSample(128)
        self.conv9=Conv_Block(128,64)
        self.out=nn.Conv2d(64,3,3,1,1)
        #64通道转换为最终要输出的3通道，为了形状大小不变，设置padding=1，进行1x1卷积
        self.Th=nn.Sigmoid()
    def forward(self,x):
        R1=self.conv1(x)
        R2=self.conv2(self.down1(R1))
        R3=self.conv3(self.down2(R2))
        R4=self.conv4(self.down3(R3))
        R5=self.conv5(self.down4(R4))
        O1=self.conv6(self.up1(R5,R4))
        O2=self.conv7(self.up2(O1,R3))
        O3=self.conv8(self.up3(O2,R2))
        O4=self.conv9(self.up4(O3,R1))
        logits=self.out(O4)
        return self.Th(logits)
    
if __name__=='__main__':
    x=torch.randn(2,3,256,256)#生成了“2张3通道、256x256分辨率的随机彩色图像数据”，是深度学习中模拟图像输入的常用方式
    #在 PyTorch 中，4 维张量通常用于表示 批量的图像数据，各维度的含义如下：
    # 第 0 维（2）：batch_size（批量大小）→ 表示包含 2 张图像；
    # 第 1 维（3）：channels（通道数）→ 表示每张图像是 3 通道（通常对应 RGB 彩色图像）；
    # 第 2 维（256）：height（高度）→ 每张图像的高度为 256 像素；
    # 第 3 维（256）：width（宽度）→ 每张图像的宽度为 256 像素
    net=UNet()
    print(net(x).shape)  #验证输出的形状和输入形状是否一样




    
