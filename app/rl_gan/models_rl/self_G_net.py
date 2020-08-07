
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models_rl.misc.spectral import SpectralNorm
import numpy as np
from models_rl.layers import *
__all__= ['self_gen_net']

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class Generator(nn.Module):
   

    def __init__(self,  image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size

        layer1 = []
        layer2 = []
        layer3 = []
        # layern = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 3, 2, 2))) # 4,2,1
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 3, 2, 2)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        # curr_dim = int(curr_dim / 2)
        #
        # layern.append(SpectralNorm(nn.ConvTranspose1d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        # layern.append(nn.BatchNorm2d(int(curr_dim / 2)))
        # layern.append(nn.ReLU())


        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

       # self.ln = nn.Sequential(*layern)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(64, 1, 2, 2, 1)) # curr_dim
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 64, 'relu') #128
        self.attn2 = Self_Attn( 64,  'relu')
        self.input1d2d = nn.ConvTranspose1d(144,128,1)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
       # out=self.l4(out)
        #out,p2 = self.attn2(out)
        out=self.last(out)

        out = out.view(-1, 1, 144)
        out = out.transpose(1, 2)
        out= self.input1d2d(out)
        out = out.transpose(2, 1)

        out = out.view(-1,1,1,128)

        return out, p1#, p2


class self_G_net(nn.Module):
    def __init__(self,args):
        super(self_G_net, self).__init__()
        self.G = Generator( args.image_size, args.z_dim,
                            args.g_conv_dim)

    def forward(self, x):
        G = self.G(x)
        return G





def self_gen_net(args,data=None):

    model = self_G_net(args)
    model.G.load_state_dict(data)

    return model