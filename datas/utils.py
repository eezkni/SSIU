import os
from datas.benchmark import Benchmark
from datas.div2k import DIV2K
from datas.df2k import DF2K
from torch.utils.data import DataLoader

import numpy as np
import torch

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor


def create_datasets(args):
    if args.training_dataset == "div2k":
        div2k = DIV2K(
            os.path.join(args.data_path, 'DIV2K/DIV2K_train_HR'), 
            os.path.join(args.data_path, 'DIV2K/DIV2K_train_LR_bicubic'), 
            os.path.join(args.data_path, 'div2k_cache'),
            train=True, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
        )
        train_dataloader = DataLoader(dataset=div2k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    
    elif args.training_dataset == 'df2k':
        df2k = DF2K(
            os.path.join(args.data_path, 'DF2K/DF2K_train_HR'), 
            os.path.join(args.data_path, 'DF2K/DF2K_train_LR_bicubic'), 
            os.path.join(args.data_path, 'df2k_cache'),
            train=True, 
            augment=args.data_augment, 
            scale=args.scale, 
            colors=args.colors, 
            patch_size=args.patch_size, 
            repeat=args.data_repeat, 
        )
        train_dataloader = DataLoader(dataset=df2k, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        raise NotImplementedError("=== dataset [{}] is not found ===".format(args.training_dataset))
   
    valid_dataloaders = []
    if 'Set5' in args.eval_sets:
        set5_hr_path = os.path.join(args.data_path, 'benchmark/Set5/HR')
        set5_lr_path = os.path.join(args.data_path, 'benchmark/Set5/LR_bicubic')
        set5  = Benchmark(set5_hr_path, set5_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'set5', 'dataloader': DataLoader(dataset=set5, batch_size=1, shuffle=True)}]

    if 'Set14' in args.eval_sets:
        set14_hr_path = os.path.join(args.data_path, 'benchmark/Set14/HR')
        set14_lr_path = os.path.join(args.data_path, 'benchmark/Set14/LR_bicubic')
        set14 = Benchmark(set14_hr_path, set14_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'set14', 'dataloader': DataLoader(dataset=set14, batch_size=1, shuffle=True)}]

    if 'B100' in args.eval_sets:
        b100_hr_path = os.path.join(args.data_path, 'benchmark/B100/HR')
        b100_lr_path = os.path.join(args.data_path, 'benchmark/B100/LR_bicubic')
        b100  = Benchmark(b100_hr_path, b100_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'b100', 'dataloader': DataLoader(dataset=b100, batch_size=1, shuffle=True)}]

    if 'Urban100' in args.eval_sets:
        u100_hr_path = os.path.join(args.data_path, 'benchmark/Urban100/HR')
        u100_lr_path = os.path.join(args.data_path, 'benchmark/Urban100/LR_bicubic')
        u100  = Benchmark(u100_hr_path, u100_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'u100', 'dataloader': DataLoader(dataset=u100, batch_size=1, shuffle=True)}]
        
    if 'Manga109' in args.eval_sets:
        manga_hr_path = os.path.join(args.data_path, 'benchmark/Manga109/HR')
        manga_lr_path = os.path.join(args.data_path, 'benchmark/Manga109/LR_bicubic')
        manga = Benchmark(manga_hr_path, manga_lr_path, scale=args.scale, colors=args.colors)
        valid_dataloaders += [{'name': 'manga109', 'dataloader': DataLoader(dataset=manga, batch_size=1, shuffle=True)}]
    
    if len(valid_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(0, len(valid_dataloaders)):
            selected += " " + valid_dataloaders[i]['name']
        print('##=== select {} for evaluation! ===##'.format(selected))

    return train_dataloader, valid_dataloaders