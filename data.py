from torch.utils.data import Dataset
import os
from utils import *
from torchvision import transforms


transform=transforms.Compose([
    transforms.ToTensor()
])
class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'SegmentationClass'))  
        #获取所有SegmentationClass中的文件名
    def __len__(self):
        return len(self.name)
    def __getitem__(self, index):
        segment_name=self.name[index]  #图片的文件名xxx.png
        segment_path=os.path.join(self.path,"SegmentationClass",segment_name)
        #拼接获得每个图片的地址
        image_path=os.path.join(self.path,"JPEGImages",segment_name.replace("png","jpg"))
        #拼接获得原图地址，并把segment_name中的png转变成jpg
        
        #统一尺寸读取图片
        segment_image=keep_image_size_open(segment_path)
        image=keep_image_size_open(image_path)
        return transform(image),transform(segment_image)
        #__getitem__(index)返回两个图片的PyTorch Tensor,组成了元组
    
if __name__=='__main__':
    data=MyDataset(r"C:\Users\17512\Desktop\U-net分割\dataset\VOCdevkit\VOC2007")
    print(data[0][0].shape)  #输出第一张图片原图image的tensor
    print(data[0][1].shape)  #输出第一张图片分割图segment_image的tensor

