import os
from net import *
from utils import *
from data import *
from torchvision.utils import save_image

# 加载模型（含预训练权重）
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
net=UNet().to(device)
weight_path='params/unet.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print("successfully load weights")
else:
    print("no loading")

# 图像输出与预处理
_input=input("please input image path:")
img=keep_image_size_open(_input)
img_data=transform(img)
print(img_data.shape)
img_data=torch.unsqueeze(img_data,dim=0) #torch.unsqueeze(input, dim)用于给张量（Tensor）增加维度
# 增加batch维度 → [1,3,H,W]（模型要求批量输入）

out=net(img_data) #模型输出out是分割预测的张量（值范围通常经sigmoid后在 0~1 之间，代表每个像素的类别概率）
save_image(out,'test_result/result.jpg')
print(out)
