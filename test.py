# -*- coding:utf-8 -*-
import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tqdm import tqdm
import torch
from Dataloading.DataLoadSignalChannelKuiHua import DataLoadSignalChannelKuiHua
from models.RFVAE import RFVAE
from torchvision.utils import save_image

""" set flags / seeds """
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    # data option
    parser.add_argument('--file_path', type=str,
                        default="/home/Swap/zhangwenbo/KuiHuaDatasets/ProcessImageTest256/")
    # ['tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16', 'SAZ','waterway']
    parser.add_argument('--class_list_img', type=list,
                        default=['tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16', 'SAZ','waterway'])
    # ['albedo_01', 'albedo_02', 'albedo_03','albedo_04', 'albedo_05', 'albedo_06']
    parser.add_argument('--class_list_lab', type=list,
                        default=['albedo_01', 'albedo_02', 'albedo_03','albedo_04', 'albedo_05', 'albedo_06'])
        
    parser.add_argument('--model_name', type=str, default='RFVAE')
    parser.add_argument('--iter_number_all', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='results/')
    parser.add_argument('--pth_path', type=str, default='xxxx.pt')
    
    opt = parser.parse_args()

    

    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = DataLoadSignalChannelKuiHua(file_path=opt.file_path)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    """ instantiate network"""
    if opt.model_name == "RFVAE":
        model = RFVAE(target_channels=len(opt.class_list_lab),
                            condition_channels=len(opt.class_list_img),
                            out_channels=len(opt.class_list_lab))

    """model init or load checkpoint"""
    model.load_state_dict(torch.load(opt.pth_path))
    model = model.to(device=device)

    """
    清空存储文件下的所有文件，并创建分类文件夹
    """
    for iter_number in range(opt.iter_number_all):
        now_path_file = os.path.join(opt.result_dir,"test"+str(iter_number))
        if not os.path.exists(now_path_file):
            os.makedirs(now_path_file)
        # 每个通道存储一下
        for c in opt.class_list_lab:
            now_path = os.path.join(now_path_file,c)
            if not os.path.exists(now_path):
                os.makedirs(now_path)
            for img_name in os.listdir(now_path):
                os.remove(os.path.join(now_path,img_name))

    with tqdm(total= len(train_loader)) as _tqdm:
        _tqdm.set_description('epoch train: {}/{}'.format(1,1))
        with torch.no_grad():
            model.eval()
            #得到的图像为红外，微光
            for j, (imgs,labs, img_name) in enumerate(train_loader):
                imgs = imgs.to(device)
                # 迭代多少次
                for iter_number in range(opt.iter_number_all):
                    mean, logstd, out = model(None,imgs)
                    for iii in range(imgs.shape[0]):
                        now_img_name  = img_name[iii]
                        for kkk in range(6):
                            now_write_path = os.path.join(opt.result_dir,"test"+str(iter_number),'albedo_0'+str(kkk+1),now_img_name)
                            save_image(out[iii:iii+1,kkk,...],now_write_path)
                _tqdm.update(1)
