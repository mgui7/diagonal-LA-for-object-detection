import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import argparse
import pickle
from matplotlib import pyplot as plt
import random

from layers.functions.detection import Detect
from data.kitti import KittiDetection
from data import *
from ssd import build_ssd
from data import KITTI_CLASSES as labels

# helper function
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
parser.add_argument('--boundary', default=False, type=str2bool,
                    help='Output the boundary uncertainty values')
parser.add_argument('--images', default=10, type=int,
                    help='Number of images for output')
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

def eval_unvertainty_diag(model, x, H, diag, boundary):
    threshold = 0.5
    x = Variable(x.cuda(), requires_grad=True)
    model.softmax = nn.Softmax(dim=-1)
    model.detect = Detect()
    model.phase = 'test'
    model.cuda()

    detections = model.forward(x)
    out = torch.Tensor([[0,0,0,0,0,0]])
    for i in range(detections.size(1)):
        for j in range(detections.size(2) - 1):
            if detections[0,i,j,0] >= threshold:
                # out.append(torch.cat((torch.Tensor([i]), detections[0,i,j,:])))
                out = torch.cat(( out , torch.cat((torch.Tensor([i]), detections[0,i,j,:])).unsqueeze(dim=0) ))

    uncertainties = []
    endIndex = 6 if boundary else 2
    for _ in range(1,out.size(0)):
        uncertainty = []
        for i in range(1,endIndex):

            # retaining graph for multiple backward propagations
            out[_,i].backward(retain_graph = True)

            # Loading all gradients from layers
            all_grad = torch.Tensor()
            for layer in model.modules():
                if layer.__class__.__name__ in diag.layer_types:
                    if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                        grads = layer.weight.grad.contiguous().view(layer.weight.grad.shape[0], -1)
                        if layer.bias is not None:
                            grads = torch.cat([grads, layer.bias.grad.unsqueeze(dim=1)], dim=1)
                        all_grad = torch.cat((all_grad, torch.flatten(grads)))
                        model.zero_grad()

            J = all_grad.unsqueeze(0)
            pred_std = torch.abs(J * H * J).sum()
            del J, all_grad
            uncertainty.append(pred_std)
        uncertainties.append(torch.FloatTensor(uncertainty))
    uncertainties = torch.stack(uncertainties) if uncertainties else torch.tensor([])

    return out[1:], uncertainties

def test_bnn(num_iterations = 10, boundary=False):
    cfg = kitti_config
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])            # initialize SSD
    weight_path = args.resume
    ssd_net.load_weights(weight_path)
    ssd_net.cuda()
    net = ssd_net

    hessian_filename= os.path.join('weights/diag.obj')

    print('Loading diagonalized Hessian...')
    filehandler = open(hessian_filename, 'rb')
    diag = pickle.load(filehandler)
    print('Finished!')

    tic = time.time()
    for _ in range(num_iterations):
        testset = KittiDetection(root=ROOT)
        img_id = random.randint(0,len(testset)-1)
        image = testset.pull_image(img_id)

        # Resize
        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # View the sampled input image before transform
        plt.figure(figsize=(12,50))
        plt.imshow(rgb_image)
        plt.axis('off')

        h = []
        for i,layer in enumerate(list(diag.model.modules())[1:]):
            if layer in diag.state:
                H_i = diag.inv_state[layer]
                h.append(torch.flatten(H_i))
        H = torch.cat(h, dim=0)

        mean_predictions, uncertainty = eval_unvertainty_diag(net, xx, H, diag, boundary)
        mean_predictions = mean_predictions.detach()
        uncertainty = uncertainty.detach() ** 0.5

        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        colors = plt.cm.hsv(np.linspace(0, 1, 9)).tolist()

        for prediction,unc in zip(mean_predictions,uncertainty):
            index = int(prediction[0])
            label_name = labels[index - 1]
            score = prediction[1]

            coords = prediction[2:]
            pt = (coords*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1

            color = colors[index]
            label_name = labels[index - 1]
            display_txt = '%s: %.2f'%(label_name, score) + ' '


            currentAxis = plt.gca()
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1] - 12 if index == 1 else pt[3], \
                display_txt + str( "{:.2f}".format(float(unc[0])) ), \
                bbox={'facecolor':color, 'alpha':1}, fontsize = 12)

            # For boundary uncertainty, subsitute eval_unvertainty_diag(net, xx, H, diag) as eval_unvertainty_diag(net, xx, H, diag,True)
            # And uncomment the code below:
            # currentAxis.text(pt[0], (pt[1]+pt[3])/2, "{:.2f}".format(float(unc[1])*10), bbox={'facecolor':color, 'alpha':0.5}, fontsize=5)
            # currentAxis.text((pt[0]+pt[2])/2, pt[1], "{:.2f}".format(float(unc[2])*10), bbox={'facecolor':color, 'alpha':0.5}, fontsize=5)
            # currentAxis.text(pt[2], (pt[1]+pt[3])/2, "{:.2f}".format(float(unc[3])*10), bbox={'facecolor':color, 'alpha':0.5}, fontsize=5)
            # currentAxis.text((pt[0]+pt[2])/2, pt[3], "{:.2f}".format(float(unc[4])*10), bbox={'facecolor':color, 'alpha':0.5}, fontsize=5)
        
        plt.show()

    toc = time.time()
    print('Average Bayesian inference duration:',  "{:.2f}s".format((toc-tic) / num_iterations))

if __name__ == '__main__':
    print(args.images, args.boundary)
    test_bnn(args.images, args.boundary)