import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


"""
DeepCVAE数据加载
    加载红外和微光图像，不做任何处理
参数列表：
    img_path:红外图像路径
    lab_path:微光图像路径
    class_list:类别列表
返回值：
    img:单通道红外图像
    lab:单通道微光图像
    img_name:图像名称
"""

class DataLoadSignalChannelKuiHua(Dataset):
    def __init__(self, file_path,
                 class_list_img=['tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16', 'SAZ','waterway'],
                 class_list_lab=['albedo_01', 'albedo_02', 'albedo_03','albedo_04', 'albedo_05', 'albedo_06']):
        self.img_names = []
        self.file_path = file_path
        self.class_list_img = class_list_img
        self.class_list_lab = class_list_lab
        self.TF = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        for img_name in os.listdir(os.path.join(file_path,class_list_img[0])):
            self.img_names.append(img_name)
        # self.img_names = self.img_names[:10000]

    def __getitem__(self, index):
        reversal_list = ['tbb_07', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16']
        images = torch.zeros((len(self.class_list_img),256,256))
        labels = torch.zeros((len(self.class_list_lab),256,256))
        name = self.img_names[index]
        for i,c in enumerate(self.class_list_img):
            if c!="waterway":
                img = Image.open(os.path.join(self.file_path,c,name)).convert("L")
                img = self.TF(img)
            else:
                name_list = name.split("_")
                now_name = name_list[2]+"_"+name_list[3]
                img = Image.open(os.path.join(self.file_path,c,now_name)).convert("L")
                img = self.TF(img)
            if c in reversal_list:
                img = 1.0-img
            images[i:i+1,...]=img
        for i,c in enumerate(self.class_list_lab):
            lab = Image.open(os.path.join(self.file_path,c,name)).convert("L")
            lab = self.TF(lab)
            labels[i:i+1,...]=lab

        return images,labels,self.img_names[index]

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    file_path = "ProcessImage256/"

    train_dataset = DataLoadSignalChannelKuiHua(file_path=file_path)
    print("数据个数：", train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=False)

    for batch, [img, lab, name] in enumerate(train_loader):
        print(batch, name,len(name),img.shape,lab.shape,torch.max(img),torch.min(img))
        save_image(img[0,...].unsqueeze(1),'img.png',nrow=3)
        save_image(lab[0,...].unsqueeze(1),'lab.png',nrow=3)
        break
    # pass
