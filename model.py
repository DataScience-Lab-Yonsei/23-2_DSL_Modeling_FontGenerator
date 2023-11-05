import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=True):
        super().__init__()
        self.apply_batchnorm = apply_batchnorm
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.apply_batchnorm:
            x = self.batch_norm(x)
        x = self.relu(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv_block = Conv(in_ch, out_ch, apply_batchnorm=apply_batchnorm)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x


class DeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, apply_batchnorm=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(out_ch*2,out_ch,2,2)
        self.conv = Conv(in_ch,out_ch,apply_batchnorm=apply_batchnorm)

    def forward(self, x1, x2, x3):
        x = self.deconv(x1)
        x = torch.cat((x, x2 ,x3),dim=1)
        x = self.conv(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp_conv = Conv(1,32,apply_batchnorm=False)
        self.down1 = ConvBlock(32,64, apply_batchnorm=False)
        self.down2 = ConvBlock(64,128, apply_batchnorm=True)
        self.down3 = ConvBlock(128,256, apply_batchnorm=True)
        self.down4 = ConvBlock(256,512, apply_batchnorm=True)
        self.down5 = ConvBlock(512,1024, apply_batchnorm=True)

    def forward(self, inputs):
        x1 = self.inp_conv(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        return x1, x2, x3, x4, x5, x6
    

class Decoder(nn.Module):
    def __init__(self,cat=False):
        super().__init__()
        self.up0 = Conv(1024*2,1024, apply_batchnorm=False)
        self.up1 = DeConvBlock(512*3,512, apply_batchnorm=False)
        self.up2 = DeConvBlock(256*3,256, apply_batchnorm=False)
        self.up3 = DeConvBlock(128*3,128, apply_batchnorm=False)
        self.up4 = DeConvBlock(64*3,64, apply_batchnorm=False)
        self.up5 = DeConvBlock(32*3,32, apply_batchnorm=False)
        self.conv = Conv(32,32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self, x1, x2, x3, x4, x5, x6,
                      e1, e2, e3, e4, e5, e6):
        x = torch.cat((x6,e6),dim=1)
        x = self.up0(x)
        x = self.up1(x,x5,e5)
        x = self.up2(x,x4,e4)
        x = self.up3(x,x3,e3)
        x = self.up4(x,x2,e2)
        x = self.up5(x,x1,e1)
        x = self.conv(x)
        x = self.out_conv(x)
        return x
    

class GenerativeModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.x_encoder = Encoder()
    self.x_decoder = Decoder()

  def forward(self, inputs,e1,e2,e3,e4,e5,e6):
    x = self.x_encoder(inputs)
    x = self.x_decoder(*x,e1,e2,e3,e4,e5,e6)
    return x
  

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.down1 = Conv(2,64, apply_batchnorm=False)
    self.down2 = Conv(64,64*2, apply_batchnorm=False)
    self.down3 = Conv(64*2,64*4, apply_batchnorm=False)
    self.down4 = Conv(64*4,64*8, apply_batchnorm=False)
    self.out_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False)
    self.maxpool = nn.MaxPool2d(2)

  def forward(self, inp, tar):
    x = torch.cat((inp,tar),axis=1)
    x = self.down1(x)
    x = self.maxpool(x)
    x = self.down2(x)
    x = self.maxpool(x)
    x = self.down3(x)
    x = self.maxpool(x)
    x = self.down4(x)
    x = self.out_conv(x)
    return x
  

def load_trained_model(generator_result_path='.'):
    # generator_result_path: a path of generator_result.pt
    #                        ex."/content/drive/MyDrive/FancyFont/현빈"
    generator = GenerativeModel()

    ### 이미 학습해둔 게 있어서 추가학습
    generator_save_path = generator_result_path

    generator.load_state_dict(torch.load(generator_save_path
                                         +"/generator_result.pt"))

    return generator


# Loading extraced features of each Korean word from UNet model
def load_UNet_result_feat(UNet_result_feat_path="."):
    # UNet_result_file_path: a path of UNet_result_feat.npz
    #                        ex."/content/drive/MyDrive/FancyFont/현빈"
    UNet_result_ = np.load(UNet_result_feat_path+"/UNet_result_feat.npz")
    UNet_result_feat = {}
    UNet_result_feat['cl1'] = torch.Tensor(UNet_result_['cl1'])
    UNet_result_feat['cl2'] = torch.Tensor(UNet_result_['cl2'])
    UNet_result_feat['cl3'] = torch.Tensor(UNet_result_['cl3'])
    UNet_result_feat['cl4'] = torch.Tensor(UNet_result_['cl4'])
    UNet_result_feat['cl5'] = torch.Tensor(UNet_result_['cl5'])
    UNet_result_feat['cl6'] = torch.Tensor(UNet_result_['cl6'])
    return UNet_result_feat


