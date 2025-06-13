import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from datas.utils import create_datasets
import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import sys
import time
import glob
from torch.utils.tensorboard import SummaryWriter
from utils import ldr_f2u
import numpy as np
import cv2
import random
from losses import FFTLoss
from torchvision.utils import save_image
from scheduler import GradualWarmupScheduler


parser = argparse.ArgumentParser(description='SSUFSR')
## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/SSUFSR_light_x2.yml', help = 'pre-config file for training')
# parser.add_argument('--config', type=str, default='./configs/SSUFSR_light_x3.yml', help = 'pre-config file for training')
# parser.add_argument('--config', type=str, default='./configs/SSUFSR_light_x4.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')


if __name__ == '__main__':

    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
    
    print('## SSUFSR model for scale: X{}'.format(args.scale))

    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('## use cuda & cudnn for acceleration! ##')
        print('## the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('## use cpu for training! ##')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}_network'.format(args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)


    ## load pretrain  
    if args.pretrain is not None:  
        print('## load pretrained model: {}! ##'.format(args.pretrain))  
        ckpt = torch.load(args.pretrain)  
        model.load_state_dict(ckpt['model_state_dict'])  

    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False


    ## create folder for test results
    save_path = os.path.join(args.log_path, 'testing_results_x' + str(args.scale))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ## testing
    test_log = ''
    with torch.no_grad():
        for valid_dataloader in valid_dataloaders:
            avg_psnr, avg_ssim = 0.0, 0.0
            name = valid_dataloader['name']
            loader = valid_dataloader['dataloader']
            name = valid_dataloader['name']
            count = 0

            for lr, hr, img_name in tqdm(loader, ncols=80):
                count += 1
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                
                if args.save_image:
                    os.makedirs(save_path,exist_ok=True)
                    save_image(sr, os.path.join(save_path, img_name[0]))   

                # conver to ycbcr
                if args.colors == 3:
                    hr_ycbcr = utils.rgb_to_ycbcr(hr)
                    sr_ycbcr = utils.rgb_to_ycbcr(sr)
                    hr = hr_ycbcr[:, 0:1, :, :]
                    sr = sr_ycbcr[:, 0:1, :, :]

                # crop image for evaluation
                hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                if args.rgb_range == 1:
                    hr, sr = hr*255., sr*255.
                # calculate psnr and ssim
                psnr = utils.calc_psnr(sr, hr)       
                ssim = utils.calc_ssim(sr, hr)         
                avg_psnr += psnr
                avg_ssim += ssim

            avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
            avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
            test_log += f"{name} dataset: avg_psnr is {avg_psnr},avg_ssim is {avg_ssim}."

    # print log & flush out
    tqdm.write(test_log)
    sys.stdout.flush()

        