import torch
import torch.nn as nn
from torch.nn import init

"""
对微光图像进行编码，隐特征对齐到N(0,1)
条件使用多通道红外图像(红外图像，一阶梯度，二阶梯度，直方图深度图)，编码到128特征，
编码器，解码器是携带残差和注意力机制的Unet结构
"""

"""
Swish 是由Google提出的一个非线性激活函数, 在某些深度网络中表现优于ReLU。
"""
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

"""
多选择注意力机制
Multi-feature Selection Attention Mechanism(MSAM)
选择激活两个特征图中置信度最大的特征
从空间维度和通道维度选择
参数列表：
    in_channels：输入通道数
    ratio：MLP缩减比率
    
    d, x：两个特征图
返回值：
    out：选择融合后的特征图
"""
class MFSAtt(nn.Module):
    def __init__(self, in_size, ratio):
        super(MFSAtt, self).__init__()
        self.in_size = in_size
        self.ratio = ratio
        
        # 通道选择：自适应池化
        self.max_pooling1 = nn.AdaptiveMaxPool2d(1)
        self.max_pooling2 = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pooling2 = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(self.in_size, self.in_size // self.ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.in_size // self.ratio, self.in_size, kernel_size=1),
            nn.Sigmoid()
            )
        
        self.conv_MLP = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3),
            nn.Sigmoid()
            )
        
        self.softmax = nn.Softmax(dim=1)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, d, x):
        # 通道维度选择
        d_max = self.max_pooling1(d)
        x_max = self.max_pooling2(x)
        d_avg = self.avg_pooling1(d)
        x_avg = self.avg_pooling2(x)
        
        d_max = self.shared_MLP(d_max)
        x_max = self.shared_MLP(x_max)
        d_avg = self.shared_MLP(d_avg)
        x_avg= self.shared_MLP(x_avg)

        d_x_ma = torch.cat([d_max.unsqueeze(1),
                            d_avg.unsqueeze(1),
                            x_max.unsqueeze(1),
                            x_avg.unsqueeze(1)],dim=1)
        d_x_ma = self.softmax(d_x_ma)
        
        d_ma = d_x_ma[:,0,:,:]+d_x_ma[:,1,:,:]
        x_ma = d_x_ma[:,2,:,:]+d_x_ma[:,3,:,:]
        
        d = d_ma*d
        x = x_ma*x
        
        # 空间维度选择
        d_max,_ = torch.max(d, dim=1, keepdim=True)
        x_max,_ = torch.max(x, dim=1, keepdim=True)
        d_avg = torch.mean(d, dim=1, keepdim=True)
        x_avg = torch.mean(x, dim=1, keepdim=True)
        
        d_max = self.conv_MLP(d_max)
        x_max = self.conv_MLP(x_max)
        d_avg = self.conv_MLP(d_avg)
        x_avg = self.conv_MLP(x_avg)
        
        d_x_ma = torch.cat([d_max.unsqueeze(1),
                            d_avg.unsqueeze(1),
                            x_max.unsqueeze(1),
                            x_avg.unsqueeze(1)],dim=1)
        d_x_ma = self.softmax(d_x_ma)
        
        d_ma = d_x_ma[:,0,:,:]+d_x_ma[:,1,:,:]
        x_ma = d_x_ma[:,2,:,:]+d_x_ma[:,3,:,:]
        
        d = d_ma*d
        x = x_ma*x
        
        return d+x
class MFSAttNone(nn.Module):
    def __init__(self,):
        super(MFSAttNone, self).__init__()
    def forward(self, d, x):
        return d+x
"""
频域特征计算
    channels：输入特征维度
"""
class FDC(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_real = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            Swish()
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            Swish()
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        fft = torch.fft.fft2(x)
        real, imag = fft.real, fft.imag
        real = self.conv_real(real)
        imag = self.conv_imag(imag)
        fft_new = torch.complex(real, imag)
        return torch.fft.ifft2(fft_new).real
"""
双卷积残差模块
    in_channels：输入通道
    out_channels：输出通道维度

频域通道：傅里叶变换提取实部和虚部特征
空域通道：提取空域特征
多特征选择注意力机制：对空域特征和频域特征进行选择
skip：跳接结构
"""
class FRDC(nn.Module):
    def __init__(self, in_channels, out_channels, attn=False, fft=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            Swish()
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if attn:
            self.attn = MFSAtt(in_size=out_channels,ratio=4)
        else:
            self.attn = MFSAttNone()

        if fft:
            if in_channels!=out_channels:
                self.fft_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.fft_conv = nn.Identity()
            self.fft = FDC(channels=out_channels)
        else:
            self.fft_conv = nn.Identity()
            self.fft = nn.Identity()

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        # 频域特征提取通道
        xifft = self.fft(self.fft_conv(x))
        # 空域特征提取通道
        xspace = self.double_conv(x)
        # skip and attention
        xatt = self.attn(xifft,xspace)
        return xatt + self.shortcut(x)


"""
下采样
    使用卷积核为2，步长为2的卷积进行下采样
"""
class Down(nn.Module):
    def __init__(self, in_ch,out_ch,attn=True,fft=True,down=False):
        super().__init__()
        self.down = down
        self.conv = FRDC(in_channels=in_ch,out_channels=out_ch,attn=attn,fft=fft)
        self.main = nn.Conv2d(in_ch, in_ch, 2, stride=2, padding=0)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        x1 = self.conv(x)
        if self.down:
            x2 = self.main(x1)
        else:
            x2 = None
        return x1,x2
"""
上采样
    使用邻近插值和3*3的卷积进行上采样
""" 
class Up(nn.Module):
    """double conv then Upscaling"""

    def __init__(self, in_channels, out_channels1, out_channels2, attn=False,fft=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up_conv = nn.Sequential(
            FRDC(in_channels, out_channels1,attn=attn,fft=fft),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels1, out_channels2, kernel_size=3, padding=1)
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return x


class RFVAE(nn.Module):
    def __init__(self, target_channels=1, condition_channels=1, out_channels=1):
        super(RFVAE, self).__init__()

        """
        encoder
        生成目标图像编码到高斯分布 
        """
        self.target_inconv = nn.Conv2d(target_channels, 128, kernel_size=3, padding=1)
        self.target_down1 = Down(128, 128,attn=False,fft=False,down=True)
        self.target_down2 = Down(128, 128,attn=False,fft=False,down=True)
        self.target_down3 = Down(128, 128,attn=True,fft=True,down=True)
        self.target_down4 = Down(128, 128,attn=True,fft=True,down=True)
        self.target_centre = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        """
        encoder
        输入条件编码到隐空间
        """
        self.condition_inconv = nn.Conv2d(condition_channels, 128, kernel_size=3, padding=1)
        self.condition_down1 = Down(128, 128,attn=False,fft=True,down=True)
        self.condition_down2 = Down(128, 128,attn=False,fft=True,down=True)
        self.condition_down3 = Down(128, 128,attn=True,fft=True,down=True)
        self.condition_down4 = Down(128, 128,attn=True,fft=True,down=True)
        self.condition_centre = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        """
        decoder
        将采样的噪声和输入条件解码到目标图像
        """
        self.up1 = Up(128+64, 128, 128,attn=True,fft=True)
        self.up2 = Up(256, 128, 128,attn=True,fft=True)
        self.up3 = Up(256, 128, 128,attn=False,fft=True)
        self.up4 = Up(256, 128, 128,attn=False,fft=True)

        self.end_conv = nn.Sequential(
             FRDC(256, 128,attn=True,fft=True),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0)
            )
        

    # 此处的y是需要重建的目标图像，x是输入的条件
    def forward(self, y, x):
        """
        encoder
        输入条件编码到隐空间
        """
        c1 = self.condition_inconv(x)
        c1,c2 = self.condition_down1(c1)
        c2,c3 = self.condition_down2(c2)
        c3,c4 = self.condition_down3(c3)
        c4,c5 = self.condition_down4(c4)
        c5 = self.condition_centre(c5)
        """
        encoder
        生成目标图像编码到高斯分布 
        """
        if y!=None:
            t1 = self.target_inconv(y)
            t1, t2 = self.target_down1(t1)
            t2, t3 = self.target_down2(t2)
            t3, t4= self.target_down3(t3)
            t4, t5 = self.target_down4(t4)
            t5 = self.target_centre(t5)
            mean = t5[:, 0:64, :, :]  # mean
            logstd = t5[:, 64:128, :, :]  # standard deviation
            za1 = self.reparametrize(mean, logstd)
        else:
            za1 = torch.randn(x.shape[0],64,c5.shape[-2],c5.shape[-1]).to(x.device)
            mean = torch.randn_like(za1)
            logstd = torch.randn_like(za1)
        """
        decoder
        将采样的噪声和输入条件解码到目标图像
        """
        dey0 = torch.cat([za1, c5], dim=1)
        dey1 = self.up1(dey0, c4)
        dey2 = self.up2(dey1, c3)
        dey3 = self.up3(dey2, c2)
        dey4 = self.up4(dey3, c1)
        out = self.end_conv(dey4)
        return mean, logstd, out

    def reparametrize(self, mean, log_std):
        std = torch.exp(log_std)
        # sample from standard normal distribution
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

if  __name__ == "__main__":
    img = torch.randn((4,1,256,256))
    img1 = torch.randn((4,1,256,256))
    model = RFVAE(target_channels=1, condition_channels=1, out_channels=1)
    _,_,out = model(None,img)
    print(out.shape)
    # model = Up(128*2,128,128,attn=True,fft=True)
    # out = model(img,img1)
    # print(out.shape)