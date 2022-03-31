from tqdm import tqdm
import numpy as np
import glob
import os
from PIL import Image

import argparse

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from setting import config

# 利用训练好的生成器生成数据


if torch.cuda.is_available():
    torch.cuda.set_device(1)
parser = argparse.ArgumentParser(description='Self-Attention GAN trainer')
parser.add_argument('--batch', default=4, type=int, help='batch size')   # 8
parser.add_argument('--iter', default=50000, type=int, help='maximum iterations')    # 5000
parser.add_argument(
    '--code', default=224, type=int, help='size of code to input generator'    # 256？什么意思
)
parser.add_argument(
    '--lr_g', default=1e-4, type=float, help='learning rate of generator'
)
parser.add_argument(
    '--lr_d', default=4e-4, type=float, help='learning rate of discriminator'
)
parser.add_argument(
    '--n_d',
    default=1,
    type=int,
    help=('number of discriminator update ' 'per 1 generator update'),
)
parser.add_argument(
    '--model', default='sagan', choices=['sagan', 'resnet'], help='choice model class'   # dcgan
)
parser.add_argument('--path', default='./data', type=str, help='Path to image directory')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

transform = transforms.Compose(
    [
        transforms.Resize((224,224)),    # 256
        # transforms.CenterCrop(128),   # 注释
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)



def generate(args, n_class, generator, path, dataset):
    # dataset = iter(sample_data(config['server_path']+'sagan-pytorch/data/', args.batch))


    generator.load_state_dict(torch.load(path))
    generator.train(False)

    for i in tqdm(range(n_class)):
        preset_code = torch.randn(N, args.code).to(device)  # n_class类别数量
        # [N]
        input_class = torch.arange(i,i+1).long().repeat(N).to(device)  
        # preset_code: [N,code], input_class: [N]
        # fake_image: [N,3,code,code]
        fake_image = generator(preset_code, input_class)

        # 类名
        fold=dataset.classes[i]
        if not os.path.exists(os.path.join(config['save'],'train',fold)):
            os.makedirs(os.path.join(config['save'],'train',fold))
        
        # 图片排列存储：nrow行数n_class。
        # 一张图片存储多行多列
        for j in range(N):
            utils.save_image(
                fake_image[j].cpu().data,
                os.path.join(config['save'],'train',fold,f'{str(j + 1).zfill(7)}.png'),
                # nrow=n_class,
                normalize=True,
                # range=(-1, 1),
            )


if __name__ == '__main__':
    # 每一个类生成的样本数
    N=100
    dataset = datasets.ImageFolder(config['dataset'], transform=transform)
    args = parser.parse_args()
    print(args)
    sample_num=config['sample_num']
    model_path=os.path.join(config['server_path'],'sagan-pytorch/checkpoint','generator_0050000.pth')
    # 类别数
    n_class = len(glob.glob(os.path.join(config['dataset'], '*/')))

    if args.model == 'sagan':
        from model import Generator
    if args.model == 'dcgan':
        from model2 import Generator
    elif args.model == 'resnet':
        from model_resnet import Generator


    generator = Generator(args.code, n_class).to(device)

    generate(args, n_class, generator, model_path, dataset)
