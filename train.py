from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import*
from torchvision.utils import save_image

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
weight_path='params/unet.pth'
data_path=r'C:\Users\17512\Desktop\U-net分割\dataset\VOCdevkit\VOC2007'
save_path='train_image'

if __name__=='__main__':
    dataloader=DataLoader(dataset=MyDataset(data_path),batch_size=4,shuffle=True)
    net=UNet().to(device)  #实例化网络
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("successfully load weight!")
    else:
        print("not successfully load weight!")
    
    optimizer=optim.Adam(net.parameters())
    loss_fn=nn.BCELoss()
    loss_fn.to(device)

    epochs=5
    for epoch in range(epochs):
        print(f"第{epoch+1}轮训练开始")
        for i,(image,segment_image) in enumerate(dataloader):
            #遍历可迭代对象时，同时获取元素的索引和对应的值
            image,segment_image=image,segment_image.to(device)
            out_image=net(image)
            train_loss=loss_fn(out_image,segment_image)

            # 优化器三步
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            #每训练5个batch，打印当前轮次、batch索引和训练损失；
            if i%5==0:
                print(f"{epoch+1}-{i}-train_loss:{train_loss.item()}")
            #每训练50个batch，保存模型的参数到 weight_path保存模型参数
            if i%50==0:
                torch.save(net.state_dict(),weight_path)

            #取当前batch的第一张图片，将三个张量在第0度拼接，方便后续可视化
            _image=image[0]
            _segement_image=segment_image[0]
            _out_image=out_image[0]
            img=torch.stack([_image,_segement_image,_out_image],dim=0)
            #第0维创建一个新的维度，将三个形状为 (C, H, W) 的张量拼接后，结果的形状会变成 (3, C, H, W)，保持单个图像的完整性
            save_image(img,f"{save_path}/{i}.png")