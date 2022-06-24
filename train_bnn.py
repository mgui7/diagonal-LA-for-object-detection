from layers.functions.detection import Detect
from data.kitti import KittiDetection
from models.curvatures import Diagonal
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
DEVICE_LIST = [0]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import tqdm
import pickle
from matplotlib import pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

PATH_TO_WEIGHTS = 'weights/KITTI_30000.pth'
ROOT = os.path.join('data')

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='KITTI', choices=['VOC', 'COCO', 'KITTI'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default= PATH_TO_WEIGHTS, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--gpus', default='0', type=str,
                    help='GPUs to run on')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available() and args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def laplace_diag():
    cfg = kitti_config
    dataset = KittiDetection(root=ROOT,
                             transform=SSDAugmentation(cfg['min_dim'],
                                                        MEANS))


    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])            # initialize SSD
    weight_path = args.resume

    ssd_net.load_weights(weight_path)
    ssd_net.cuda()
    net = ssd_net

    if args.cuda and torch.cuda.is_available():
        # speed up using multiple GPUs
        # net = torch.nn.DataParallel(ssd_net,device_ids=DEVICE_LIST)
        cudnn.benchmark = True
        net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.3, True, 0, True, 3, 0.5,
                             False, args.cuda)

    hessian_direc   = os.path.join('weights/')
    hessian_filename= os.path.join('weights/diag.obj')

    if not os.path.exists(hessian_direc):
        os.mkdir(hessian_direc)

    if not os.path.isfile(hessian_filename):
        # compute diagonal Fisher Information Matrix
        diag = Diagonal(net)

        for _ in tqdm.tqdm(range(args.start_iter, 1871)):

            images, targets = next(batch_iterator)
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]

            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()

            diag.update(batch_size=images.size(0))

        # inversion and sampling
        estimator = diag
        estimator.invert(add=0.1, multiply=1)

        # saving kfac
        file_pi = open(hessian_filename, 'wb')
        pickle.dump(estimator, file_pi)
        print('Hessian saved.')
    else:
        print('Hessian file',hessian_filename,'already exists!')

if __name__ == '__main__':
    laplace_diag()
