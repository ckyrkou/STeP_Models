import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def make_Haarweights(w=5, h=5, n_filters=32, n_channels=16, n_boxes=2, groups=1):
    if(w==1 and h==1):
        mask = np.zeros((n_channels, n_filters, 1, 1))
        for c in range(n_channels):
            for z in range(n_filters):
                sign = (np.random.randint(2) * 2) - 1
                mask[c, z, 0, 0] = -sign
    else:
        if (groups > 1):
            mask = np.zeros((n_channels, n_filters // groups, w, h))
            vec_len = (3 * 4 - 4)
            for c in range(n_channels):
                for z in range(n_filters // groups):

                    mask[c, z, :, :] = 0
                    lw = np.random.randint(1, w)
                    lh = np.random.randint(1, h)
                    sign = (np.random.randint(2) * 2) - 1
                    for b in range(1, n_boxes + 1):
                        s_iw = np.random.randint(w - 1)
                        s_ih = np.random.randint(h - 1)
                        mask[c, z, s_iw:s_iw + lw, s_ih:s_ih + lh] = -sign
                        sign = -sign
        else:
            mask = np.zeros((n_channels, n_filters, w, h))
            vec_len = (3 * 4 - 4)
            for c in range(n_channels):
                for z in range(n_filters):

                    mask[c, z, :, :] = 0
                    lw = np.random.randint(1, w)
                    lh = np.random.randint(1, h)
                    sign = (np.random.randint(2) * 2) - 1
                    for b in range(1, n_boxes + 1):
                        s_iw = np.random.randint(w - 1)
                        s_ih = np.random.randint(h - 1)
                        mask[c, z, s_iw:s_iw + lw, s_ih:s_ih + lh] = -sign
                        sign = -sign

    return mask


def make_1x1lbp(n_filters=32, n_channels=16):
    mask = np.zeros((n_channels, n_filters, 1, 1))
    for c in range(n_channels):
        for z in range(n_filters):
            sign = (np.random.randint(2) * 2) - 1
            mask[c, z, 0, 0] = -sign

    return mask

	
def make_bin_weights(w=5, h=5, n_filters=32, n_channels=16):
    mask = np.random.randint(2,size=(n_channels,n_filters,w,h)).astype(np.float32)
    return mask

def make_ter_weights(w=5, h=5, n_filters=32, n_channels=16):
    mask = np.random.randint(3,size=(n_channels,n_filters,w,h)).astype(np.float32)-1
    return mask

def make_cslbpweights(n_filters=32, n_channels=16, w=3, h=3, groups=1):
    if (w == 1 and h == 1):
        mask = np.zeros((n_channels, n_filters, w, h))
        for c in range(n_channels):
            for z in range(n_filters):
                sign = (np.random.randint(2) * 2) - 1
                mask[c, z, 0, 0] = -sign
    else:
        if (groups > 1):
            mask = np.zeros((n_channels, n_filters // groups, w, h))
            for c in range(n_channels):
                for z in range(n_filters // groups):
                    x = np.zeros(8)
                    s_i = np.random.randint(9)
                    length = np.random.randint(4) + 1
                    inds = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0]]
                    for i in range(length):
                        sign = (np.random.randint(2) * 2) - 1
                        ind = (s_i + i) % len(x)
                        x[ind] = sign
                        mask[c, z, inds[ind][0], inds[ind][1]] = sign
                        ind = ((s_i + 4 + i)) % len(x)
                        x[ind] = -sign
                        mask[c, z, inds[ind][0], inds[ind][1]] = -sign
        else:
            mask = np.zeros((n_channels, n_filters, w, h))

            for c in range(n_channels):
                for z in range(n_filters):
                    x = np.zeros(8)
                    s_i = np.random.randint(9)
                    length = np.random.randint(4) + 1
                    inds = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [1, 0]]
                    for i in range(length):
                        sign = (np.random.randint(2) * 2) - 1
                        ind = (s_i + i) % len(x)
                        x[ind] = sign
                        mask[c, z, inds[ind][0], inds[ind][1]] = sign
                        ind = ((s_i + 4 + i)) % len(x)
                        x[ind] = -sign
                        mask[c, z, inds[ind][0], inds[ind][1]] = -sign

    return mask

class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)

class Conv1x1LBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, padding=0, bias=False)
        weights = next(self.parameters())
        binary_weights = make_1x1lbp(in_channels,out_channels)
        weights.data = torch.FloatTensor(binary_weights)
        #weights.requires_grad_(False)

class ConvCSLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1,groups=1, padding=1):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=False,stride=stride,groups=groups)
        weights = next(self.parameters())
        binary_weights = make_cslbpweights(in_channels,out_channels,kernel_size,kernel_size,groups)
        weights.data = torch.FloatTensor(binary_weights)
        weights.requires_grad_(False)

class ConvHaar(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1,groups=1,padding=1):
        super().__init__(in_channels, out_channels, 1, bias=False,stride=stride,groups=groups,padding=padding)
        weights = next(self.parameters())
        binary_weights = make_Haarweights(kernel_size,kernel_size,in_channels,out_channels,groups=groups)
        weights.data = torch.FloatTensor(binary_weights)
        weights.requires_grad_(False)

class ConvRandomBin(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,padding=1,stride = 1,groups = 1,type='lbp'):
        super().__init__(in_channels, out_channels, 1, padding=padding, bias=False,stride = stride,groups = groups)
        weights = next(self.parameters())
        binary_weights = make_bin_weights(kernel_size,kernel_size,in_channels,out_channels)
        weights.data = torch.FloatTensor(binary_weights)
        weights.requires_grad_(False)

class ConvBin(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,padding=1,stride = 1,groups = 1,bias=False,type='random'):
        super().__init__(in_channels, out_channels, 1, padding=padding, bias=bias,stride = stride,groups = groups)
        weights = next(self.parameters())

        if (type == 'ternary'):
            if (groups == in_channels):
                binary_weights = make_ter_weights(kernel_size, kernel_size, 1, in_channels)
            else:
                binary_weights = make_ter_weights(kernel_size, kernel_size, in_channels, out_channels)

        if(type == 'random'):
            if(groups == in_channels):
                binary_weights = make_bin_weights(kernel_size,kernel_size,1,in_channels)
            else:
                binary_weights = make_bin_weights(kernel_size, kernel_size, in_channels, out_channels)
        if (type == 'lbp'):
            if (groups == in_channels):
                binary_weights = make_cslbpweights(w=kernel_size,h=kernel_size,n_filters=1,n_channels=in_channels)
            else:
                binary_weights = make_cslbpweights(w=kernel_size,h=kernel_size,n_channels=out_channels,n_filters=in_channels)
        if (type == 'haar'):
            if (groups == in_channels):
                binary_weights = make_Haarweights(kernel_size, kernel_size, 1, in_channels)
            else:
                binary_weights = make_Haarweights(kernel_size, kernel_size, in_channels, out_channels)

        weights.data = torch.FloatTensor(binary_weights)
        weights.requires_grad_(False)

class ConvRandomBin_dw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,padding=1,stride = 1,groups = 1):
        super().__init__(in_channels, out_channels, 1, padding=padding, bias=False,stride = stride,groups = groups)
        weights = next(self.parameters())
        binary_weights = make_bin_weights_dw(kernel_size,kernel_size,in_channels,out_channels)
        weights.data = torch.FloatTensor(binary_weights)
        weights.requires_grad_(False)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class cappedLeakyRelu(nn.Module):
    def __init__(self,ceil=np.inf,a=0.1):
        super().__init__()
        self.ceil = ceil
        self.a = a

    def forward(self, x):
        x = F.leaky_relu(x,negative_slope=self.a,inplace=True)
        return torch.clamp(x,min=0,max=self.ceil)

class conv_step(nn.Module):

    def __init__(self, numChannels, numWeights, kernel_size=3,stride=1,padding=0, bias=False,groups=1,mix=True):
        super().__init__()
        if(numChannels == 3):
            step_ch = numChannels
        else:
            step_ch = numChannels//2
        if(groups ==1):
            g=1
        else:
            g=groups

        self.mix=mix
        self.conv_lbpCS = ConvCSLBP(numChannels, step_ch, kernel_size=kernel_size,stride = stride,groups=g)
        self.conv_haar = ConvHaar(numChannels, step_ch, kernel_size=kernel_size,stride = stride,groups=g,padding=padding if kernel_size==3 else padding+1)
        if (self.mix):
            self.conv_1x1 = nn.Conv2d(step_ch*2, numWeights, kernel_size=1,groups=2)


    def forward(self, x):
        residual = x
        x1 = self.conv_lbpCS(x)
        x2 = self.conv_haar(x)
        x = torch.cat((x1,x2),1)
        if(self.mix):
            x = self.conv_1x1(x)
        return x
		
class step_block(nn.Module):

    def __init__(self, numChannels, numWeights, kernel_size=3):
        super().__init__()
        if(numChannels == 3):
            step_ch = numChannels
            g=3
        else:
            step_ch = numChannels//2
            g = numChannels//4

        self.conv_lbpCS = ConvCSLBP(numChannels, step_ch, kernel_size=kernel_size)
        self.conv_haar = ConvHaar(numChannels, step_ch, kernel_size=kernel_size)
        self.conv_1x1 = nn.Conv2d(step_ch*2, numWeights, kernel_size=1,groups=2)
        self.af = nn.LeakyReLU(0.1,inplace=True)
        self.bn = nn.BatchNorm2d(numWeights)

    def forward(self, x):
        residual = x
        x1 = self.conv_lbpCS(x)
        x2 = self.conv_haar(x)
        x = self.conv_1x1(torch.cat((x1,x2),1))
        x = self.bn(x)
        x = self.af(x)
        return x
		
class rand_bin_block(nn.Module):

    def __init__(self, numChannels, numWeights, kernel_size=3):
        super().__init__()
        if(numChannels == 3):
            step_ch = numChannels
            g=3
        else:
            step_ch = numChannels//2
            g = numChannels//4

        self.conv_bin = ConvRandomBin(numChannels, numChannels, kernel_size=kernel_size)
        self.conv_1x1 = nn.Conv2d(numChannels, numWeights, kernel_size=1,padding=0)
        self.af = cappedLeakyRelu(ceil=255,a=0.1)
        self.bn = nn.BatchNorm2d(numWeights)

    def forward(self, x):
        residual = x
        x = self.conv_bin(x)
        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.af(x)
        return x

class LBP_block(nn.Module):

    def __init__(self, numChannels, numWeights, kernel_size=3):
        super().__init__()
        if(numChannels == 3):
            step_ch = numChannels
            g=3
        else:
            step_ch = numChannels//2
            g = numChannels//4

        self.conv_bin = ConvLBP(numChannels, numChannels, kernel_size=kernel_size)
        self.conv_1x1 = nn.Conv2d(numChannels, numWeights, kernel_size=1,padding=0)
        self.af = cappedLeakyRelu(ceil=255,a=0.1)
        self.bn = nn.BatchNorm2d(numWeights)

    def forward(self, x):
        residual = x
        x = self.conv_bin(x)
        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.af(x)
        return x

class HAAR_block(nn.Module):

    def __init__(self, numChannels, numWeights, kernel_size=3):
        super().__init__()
        if(numChannels == 3):
            step_ch = numChannels
            g=3
        else:
            step_ch = numChannels//2
            g = numChannels//4

        self.conv_bin = ConvHaar(numChannels, numChannels, kernel_size=kernel_size)
        self.conv_1x1 = nn.Conv2d(numChannels, numWeights, kernel_size=1,padding=0)
        self.af = cappedLeakyRelu(ceil=255,a=0.1)
        self.bn = nn.BatchNorm2d(numWeights)

    def forward(self, x):
        residual = x
        x = self.conv_bin(x)
        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.af(x)
        return x
        


class StepNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.block1 = step_block(numChannels=3, numWeights=64, kernel_size=5)
        self.block2 = step_block(numChannels=64, numWeights=64, kernel_size=5)

        self.block3 = step_block(numChannels=64, numWeights=128, kernel_size=5)
        self.block4 = step_block(numChannels=128, numWeights=128, kernel_size=5)

        self.avgpool1 = nn.MaxPool2d(2,2)

        self.block5 = step_block(numChannels=128, numWeights=256, kernel_size=3)
        self.block6 = step_block(numChannels=256, numWeights=256, kernel_size=3)

        self.avgpool2 = nn.MaxPool2d(2, 2)

        self.block7 = step_block(numChannels=256, numWeights=256, kernel_size=3)
        self.block8 = step_block(numChannels=256, numWeights=256, kernel_size=3)

        self.avgpool3 = nn.MaxPool2d(2, 2)

        self.block9 = step_block(numChannels=256, numWeights=256, kernel_size=3)
        self.block10 = step_block(numChannels=256, numWeights=256, kernel_size=3)

        self.avgpool4 = nn.MaxPool2d(2, 2)

        self.block11 = step_block(numChannels=256, numWeights=num_classes, kernel_size=1)

        self.gavgpool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool1(x)

        x = self.block5(x)
        x = self.block6(x)

        x = self.avgpool2(x)

        x = self.block7(x)
        x = self.block8(x)

        x = self.avgpool3(x)

        x = self.block9(x)
        x = self.block10(x)

        x = self.avgpool4(x)


        x = self.block11(x)
        x = self.gavgpool(x)
        x = x.view(x.size(0), -1)

        return x
