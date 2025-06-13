import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch
import math
from math import pi
import warnings
from einops import rearrange


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, patch_size=0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        # self.criterion = torch.nn.MSELoss()
        self.ps = patch_size

    def forward(self, pred, target):
        
        if self.ps > 0:
            B, C, H, W = pred.size()

            grid_height, grid_width = H // self.ps, W//self.ps
            pred_patch = rearrange(
                pred, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw", 
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps) 
            
            target_patch = rearrange(
                target, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw", 
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps) 
            
            pred_fft = torch.fft.rfft2(pred_patch, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target_patch, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        
        else:
            pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
    
    
class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, max_value):
        mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
        PSNR = 20 * np.log10(max_value / np.sqrt(mse))
        return PSNR


class VGG19bn_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19bn_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19_bn(weights='VGG19_BN_Weights.IMAGENET1K_V1')
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(3):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(3, 6):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(6, 10):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(10, 13):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(13, 17):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(17, 20):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(20, 23):
           self.relu3_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(26, 30):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(30, 33):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(33, 36):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(36, 39):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(39, 43):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(43, 46):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(46, 49):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(49, 52):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], resize=False, criterion='l1'):
        super(PerceptualLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'sl1':
            self.criterion = nn.SmoothL1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(criterion))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_module('vgg', VGG19bn_relu())
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = weights
        self.resize = resize
        self.transformer = torch.nn.functional.interpolate

    def att(self, in_feat):
        return torch.sigmoid(torch.mean(in_feat, 1).unsqueeze(1))

    def __call__(self, x, y):
        if self.resize:
            x = self.transformer(x, mode='bicubic', size=(224, 224), align_corners=True)
            y = self.transformer(y, mode='bicubic', size=(224, 224), align_corners=True)
        
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0.0
        loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return loss


class EASLoss(nn.Module):
    ''' edge aware smoothness loss '''
    def __init__(self):
        super(EASLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def gradient_xy(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        gy = img[:,:,:,:-1] - img[:,:,:,1:]
        return gx, gy
    
    def forward(self, pred, gt):
        pred_grad_x, pred_grad_y = self.gradient_xy(pred)
        gt_grad_x, gt_grad_y = self.gradient_xy(gt)

        weights_x = torch.exp(-torch.mean(torch.abs(gt_grad_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(gt_grad_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_grad_x) * weights_x
        smoothness_y = torch.abs(pred_grad_y) * weights_y

        loss = (torch.mean(smoothness_x) + torch.mean(smoothness_y))

        return loss


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners,
                          recompute_scale_factor=True)
        return out


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-5

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        loss = torch.mean(torch.sqrt( diff * diff + self.eps))
        # loss = torch.mean(torch.sqrt(torch.clamp(diff * diff, min=self.eps)))
        return loss


class AngularLoss(nn.Module):
    def __init__(self, shrink=True, eps=1e-6):
        super(AngularLoss, self).__init__()
        self.eps = eps
        self.shrink = shrink

    def forward(self, pred, gt):
        cossim = torch.clamp(torch.sum(pred * gt, dim=1) / (torch.norm(pred, dim=1) * torch.norm(gt, dim=1) + 1e-9), -1, 1.)
        if self.shrink:
            angle = torch.acos(cossim * (1-self.eps))
        else:
            angle = torch.acos(cossim)
        
        angle = angle * 180 / math.pi
        error = torch.mean(angle)
        return error
