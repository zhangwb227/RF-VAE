# -*- coding:utf-8 -*-
import os, argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from tqdm import tqdm
import torch
from models.loss import JointLoss
from Dataloading.DataLoadSignalChannelKuiHua import DataLoadSignalChannelKuiHua
from models.RFVAE import RFVAE
from utils.utils import print_options
from torchvision.utils import save_image

""" set flags / seeds """
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    """ Hpyer parameters """
    parser = argparse.ArgumentParser(description="")
    # training option
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--Epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--fine_tuning', type=bool, default=True)
    # data option
    parser.add_argument('--file_path', type=str,
                        default="/home/Swap/zhangwenbo/KuiHuaDatasets/ProcessImage256/")
    # ['tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16', 'SAZ','waterway']
    parser.add_argument('--class_list_img', type=list,
                        default=['tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16', 'SAZ','waterway'])
    # ['albedo_01', 'albedo_02', 'albedo_03','albedo_04', 'albedo_05', 'albedo_06']
    parser.add_argument('--class_list_lab', type=list,
                        default=['albedo_01', 'albedo_02', 'albedo_03','albedo_04', 'albedo_05', 'albedo_06'])

    parser.add_argument('--model_name', type=str, default='RFVAE')
    parser.add_argument('--result_dir', type=str, default='results/')
    
    opt = parser.parse_args()
    if not os.path.exists(opt.result_dir): os.mkdir(opt.result_dir)

    if not os.path.exists(os.path.join(opt.result_dir,"model_pth")): os.mkdir(os.path.join(opt.result_dir,"model_pth"))
    if not os.path.exists(os.path.join(opt.result_dir,"result_images")): os.mkdir(os.path.join(opt.result_dir,"result_images"))
    if not os.path.exists(os.path.join(opt.result_dir,"test_best")): os.mkdir(os.path.join(opt.result_dir,"test_best"))

    print_options(parser, opt)

    """ device configuration """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    """ dataset and dataloader """
    train_dataset = DataLoadSignalChannelKuiHua(file_path=opt.file_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    """ instantiate network"""
    if opt.model_name == "RFVAE":
        model = RFVAE(target_channels=len(opt.class_list_lab),
                            condition_channels=len(opt.class_list_img),
                            out_channels=len(opt.class_list_lab))
    model = model.to(device=device)

    """ instantiate loss function"""
    loss = JointLoss()
    loss = loss.to(device=device)
    
    """model init optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    """setting learing scheduler"""
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=opt.Epochs,
        eta_min=opt.weight_decay)
    """ training part """
    imgs_sample_xa2 = None
    labs_sample_xa1 = None
    best_loss = 1e2
    global_step = 0
    while (global_step < opt.Epochs):
        with tqdm(total= len(train_loader)) as _tqdm:
            _tqdm.set_description('epoch train: {}/{}'.format(global_step + 1, opt.Epochs))
            model.train()
            loss_sum = 0
            for j, (imgs,labs, img_name) in enumerate(train_loader):
                imgs = imgs.to(device)
                labs = labs.to(device)

                mean, logstd, out = model(labs,imgs)

                full_loss = loss(mean, logstd, out, labs)

                optimizer.zero_grad()
                full_loss.backward()
                optimizer.step()

                loss_sum += full_loss.item()
                _tqdm.set_postfix(j='({},{:.6f})'.format(j,full_loss.item()))
                _tqdm.update(1)
                # if j>5:
                #     break
            cosineScheduler.step()
            imgs_sample_xa2 = imgs.clone()
            labs_sample_xa1 = labs.clone()
        with torch.no_grad():
            model.eval()
            l = loss_sum/len(train_loader)
            # 存储模型权重
            if global_step%49==0:
                temp = '{:.6f}'.format(l)
                torch.save(model.state_dict(),
                        os.path.join(opt.result_dir,'model_pth',
                                        str(global_step)+'_'+str(temp)+'.pt'))
            if global_step%9==0:
                mean, logstd, xxa1 = model(None, imgs_sample_xa2)
                # 存储中间过程
                save_img = torch.cat([imgs_sample_xa2,
                                    labs_sample_xa1,
                                    xxa1],
                                    dim=1)
                save_img = save_img.view(-1, 1, 256, 256)
                save_image(save_img,
                        os.path.join(opt.result_dir,'result_images',
                                        str(global_step)+'.png'),nrow=3)
            # 存储最佳模型权重
            if l<best_loss:
                torch.save(model.state_dict(),os.path.join(opt.result_dir,'model_pth','best.pt'))
                best_loss = l
        print(f"global_step:{global_step}\tloss:{l}\tLR:{optimizer.state_dict()['param_groups'][0]['lr']}")
        global_step += 1
