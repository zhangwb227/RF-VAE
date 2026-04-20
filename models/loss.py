import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from math import exp

"""
直方图相交损失
    目的在于约束生成图像和原始图像的直方图分布之间的相似性
输入：
    生成图像，目标图像，
返回值：
    l1损失+l2损失+cos损失
"""
class SoftHistogram(nn.Module):
    def __init__(self, bins=128, min=0.0, max=1.0, sigma=0.02):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma

        centers = torch.linspace(min, max, bins)
        self.register_buffer("centers", centers)

    def forward(self, x):
        B = x.shape[0]

        x = x.view(B, -1)           # [B, N]
        x = x.unsqueeze(1)          # [B, 1, N]
        centers = self.centers.view(1, -1, 1)  # [1, bins, 1]

        weights = torch.exp(- (x - centers) ** 2 / (2 * self.sigma ** 2))
        hist = weights.sum(dim=-1)  # [B, bins]

        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-6)

        return hist


class SoftHistLoss(nn.Module):
    def __init__(self, bins=128,
                 l1_weight=1.0,
                 l2_weight=1.0,
                 cos_weight=1.0):
        super().__init__()

        self.hist = SoftHistogram(bins=bins)

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.cos_weight = cos_weight

    def forward(self, real, fake):
        h_real = self.hist(real)   # [B, bins]
        h_fake = self.hist(fake)

        # L1（分布差）
        l1_loss = self.l1(h_real, h_fake)

        # L2（强调大误差）
        l2_loss = self.l2(h_real, h_fake)

        # Cosine（分布形状一致性）
        cos_sim = F.cosine_similarity(h_real, h_fake, dim=1, eps=1e-8)
        cos_loss = (1 - cos_sim).mean()

        # 总损失
        loss = (self.l1_weight * l1_loss +
                self.l2_weight * l2_loss +
                self.cos_weight * cos_loss)
        return loss


"""
边缘梯度损失
    目的在于约束生成图像和原始图像的边缘纹理相似
输入：
    生成图像，目标图像，
返回值：
    loss
""" 
class EdgeTextureLoss(nn.Module):
    def __init__(self,soble_lamda=1.0,lap_lamda=1.0):
        super(EdgeTextureLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.soble_lamda = soble_lamda
        self.lap_lamda = lap_lamda
    def soble(self,image):
        B,C,H,W = image.shape
        if C==1:
            # 定义梯度算子
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            sobel_x = torch.tensor(kernel_x, dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
            sobel_y = torch.tensor(kernel_y, dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
            # 计算梯度
            grad_x = F.conv2d(image, sobel_x, padding=1)
            grad_y = F.conv2d(image, sobel_y, padding=1)
            # 计算梯度幅值
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        elif C==3:
            # 定义梯度算子
            kernel_x = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
            kernel_y = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
            sobel_x = torch.tensor(kernel_x, dtype=torch.float32).view(1, 3, 3, 3).to(image.device)
            sobel_y = torch.tensor(kernel_y, dtype=torch.float32).view(1, 3, 3, 3).to(image.device)
            # 计算梯度
            grad_x = F.conv2d(image, sobel_x, padding=1)
            grad_y = F.conv2d(image, sobel_y, padding=1)
            # 计算梯度幅值
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        elif C==6:
            # 定义梯度算子
            kernel_x = [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
            kernel_y = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
            sobel_x = torch.tensor(kernel_x, dtype=torch.float32).view(1, 6, 3, 3).to(image.device)
            sobel_y = torch.tensor(kernel_y, dtype=torch.float32).view(1, 6, 3, 3).to(image.device)
            # 计算梯度
            grad_x = F.conv2d(image, sobel_x, padding=1)
            grad_y = F.conv2d(image, sobel_y, padding=1)
            # 计算梯度幅值
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        return grad_mag
        
    def laplacian(self,image):
        B,C,H,W = image.shape
        if C==1:
            # 定义梯度算子
            kernel_x = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
            lap = torch.tensor(kernel_x, dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
            # 计算梯度
            second_deriv = F.conv2d(image, lap, padding=1)
        elif C==3:
            # 定义梯度算子
            kernel_x = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
            lap = torch.tensor(kernel_x, dtype=torch.float32).view(1, 3, 3, 3).to(image.device)
            # 计算梯度
            second_deriv = F.conv2d(image, lap, padding=1)
        elif C==6:
            # 定义梯度算子
            kernel_x = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
            lap = torch.tensor(kernel_x, dtype=torch.float32).view(1, 6, 3, 3).to(image.device)
            # 计算梯度
            second_deriv = F.conv2d(image, lap, padding=1)
        return second_deriv
        
    def forward(self, real_tensor, gen_tensor):
        loss_soble = self.MSE(self.soble(real_tensor), self.soble(gen_tensor))
        loss_lap = self.MSE(self.laplacian(real_tensor), self.laplacian(gen_tensor))
        loss = self.soble_lamda*loss_soble+self.lap_lamda*loss_lap
        return loss
    
"""
纹理特征损失
    目的在于约束生成图像和原始图像的网络前几层的纹理相似
输入：
    生成图像，目标图像，
返回值：
    loss
""" 
class VGGTextureLoss(nn.Module):
    def __init__(self, layers=[1, 3, 8, 17, 26, 35]):
        super(VGGTextureLoss, self).__init__()
        # 加载预训练的VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        # 不梯度更新
        for param in vgg.parameters():
            param.requires_grad = False
        # 提取指定层的特征
        self.layers = layers
        self.vgg_layers = nn.ModuleList()
        prev_layer = 0
        for layer in layers:
            self.vgg_layers.append(nn.Sequential(*list(vgg.children())[prev_layer:layer]))
            prev_layer = layer
        # 归一化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # MSE
        self.MSE = torch.nn.MSELoss()
        
    def forward(self, real_tensor, gen_tensor):
        # 归一化
        gen_tensor = (gen_tensor - self.mean.to(gen_tensor.device)) / self.std.to(gen_tensor.device)
        real_tensor = (real_tensor - self.mean.to(real_tensor.device)) / self.std.to(real_tensor.device)
        # 计算各层特征
        gen_features = []
        real_features = []
        # 初始特征
        x_gen = gen_tensor
        x_real = real_tensor
        # 逐层计算特征
        for layer in self.vgg_layers:
            x_gen = layer(x_gen)
            x_real = layer(x_real)
            gen_features.append(x_gen)
            real_features.append(x_real)
        # 计算损失
        loss = 0
        for gf, rf in zip(gen_features, real_features):
            loss += self.MSE(gf,rf)
        return loss / len(self.layers)
    

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
 
 
# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
 
 
# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
 
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
 
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    else:
        window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
 
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
 
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
 
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
 
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
 
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
 
    if full:
        return ret, cs
    return ret
 
 
 
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
 
        return 1.0-ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


"""
联合损失函数
    1. KL损失
    2. MSE损失
    3. SSIM损失
"""
class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.criterionMSE = torch.nn.MSELoss()
        self.ssmi = SSIM()

    def forward(self, mean, logstd, xxa1, xa1_label):
        var = torch.pow(torch.exp(logstd), 2)
        loss_KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
        loss_MSE = self.criterionMSE(xxa1, xa1_label)
        loss_ssmi = self.ssmi(xxa1, xa1_label)
        loss_full = loss_KLD + loss_MSE + loss_ssmi

        return loss_full

"""
联合损失函数
    1. KL损失
    2. 修改MSE损失
        采用迭代最小MSE，寻找由于配准偏移导致产生的MSE累计误差
    3. 修改SSIM损失
        采用迭代最小MSE的结果，再次计算SSIM
    4. 直方图相交损失
        直方图统计约束灰度差异
    5. 纹理特征损失
        VGG计算深度语义特征的一致性
    6. 边缘梯度损失
        采用迭代最小MSE的结果，再次计算边缘梯度损失
"""
class JointLoss(nn.Module):
    def __init__(self, weight=[1.0,1.0,0.7,0.7,0.4,0.4],T=8):
        super(JointLoss, self).__init__()
        # 预估偏移量
        self.T = T
        self.weight = weight
        self.mse = torch.nn.MSELoss()
        self.ssmi = SSIM()
        self.edge = EdgeTextureLoss(soble_lamda=0.6,lap_lamda=0.4)
        self.histc = SoftHistLoss()
        self.vggfeature = VGGTextureLoss()

    def minMSE(self,pred,lab):
        b,c,w,h = lab.shape
        TT = self.T*2
        B,C,W,H = b,c,w+TT,h+TT
        # 为标签添加边框
        lab = F.pad(lab, pad=(self.T, self.T, self.T, self.T), mode='constant', value=0)
        # 非边框区域在lab中的位置
        a_x1, a_y1 = self.T, self.T
        a_x2, a_y2 = W-self.T, H-self.T
        # 定义最小损失
        min_mse_loss = 65026
        # 2025_7_30修改：将初始为None修改成初始化为模型输出，防止返回空值
        min_mse_pred = pred
        min_mse_lab = lab
        # 滑动窗口范围（例如：步长为 1）
        for i in range(0, TT+1):
            for j in range(0, TT+1):
                # 图像 B 滑动后的位置框
                b_x1, b_y1 = i, j
                b_x2, b_y2 = i + w, j + h
                # A 中对应的位置
                a_crop_x1 = max(a_x1, b_x1)
                a_crop_y1 = max(a_y1, b_y1)
                a_crop_x2 = min(a_x2, b_x2)
                a_crop_y2 = min(a_y2, b_y2)
                # 计算交集区域尺寸
                hh = a_crop_y2 - a_crop_y1
                ww = a_crop_x2 - a_crop_x1
                # 从 A 中取出交集区域
                a_patch = lab[:,:,a_crop_y1:a_crop_y2, a_crop_x1:a_crop_x2]
                # 从 B 中也取出对应区域
                b_patch = pred[:,:,a_crop_y1 - b_y1 : a_crop_y1 - b_y1 + hh,
                            a_crop_x1 - b_x1 : a_crop_x1 - b_x1 + ww]
                now_loss = self.mse(a_patch,b_patch)
                if now_loss<=min_mse_loss:
                    min_mse_loss = now_loss
                    min_mse_pred = b_patch
                    min_mse_lab = a_patch
        return min_mse_loss,min_mse_pred,min_mse_lab
    
    def forward(self, mean, logstd, xxa1, xa1_label):
        var = torch.pow(torch.exp(logstd), 2)
        loss_KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
        loss_MSE,min_mse_pred,min_mse_lab = self.minMSE(xxa1, xa1_label)
        loss_ssmi = self.weight[2]*self.ssmi(xxa1, xa1_label)
        loss_edge = self.weight[3]*self.edge(xxa1, xa1_label)
        loss_histc = self.weight[4]*self.histc(xxa1,xa1_label)
        loss_vggfeature = self.weight[5]*0.5*(self.vggfeature(xxa1[:,0:3,...],xa1_label[:,0:3,...])+self.vggfeature(xxa1[:,3:6,...],xa1_label[:,3:6,...]))
        loss_full = self.weight[0]*loss_KLD + self.weight[1]*loss_MSE + loss_ssmi+ loss_edge+ loss_histc+ loss_vggfeature
        return loss_full

