# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-17 10:06:40
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from rsden.models import get_model
from rsden.loader import get_loader, get_data_path
from rsden.metrics import runningScore
from rsden.loss import *
from rsden.augmentations import *
import os


def train(args):

    # Setup Augmentations
    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])
    loss_rec=[]
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    # t_loader = data_loader(data_path, is_transform=True,
    #                        split='nyu2_train', img_size=(args.img_rows, args.img_cols))
    v_loader = data_loader(data_path, is_transform=True,
                           split='test_region', img_size=(args.img_rows, args.img_cols),task='region')

   # n_classes = t_loader.n_classes
    #trainloader = data.DataLoader(
    #    t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=4)

    # Setup Metrics
    #running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss',
                                         legend=['Loss']))
        pre_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='predict!', caption='predict.'),
        )
        ground_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='ground!', caption='ground.'),
        )
    # Setup Model
    model = get_model(args.arch)
    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # Check if model has custom optimizer / loss
    # modify to adam, modify the learning rate
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = l1
    trained=0
    scale=100
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            trained=checkpoint['epoch']
            print('load success!')
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            print('Initialize from resnet50!')
            resnet50=torch.load('/home/lidong/Documents/RSDEN/RSDEN/resnet34-333f7ec4.pth')
            model_dict=model.state_dict()            
            pre_dict={k: v for k, v in resnet50.items() if k in model_dict}
            model_dict.update(pre_dict)
            model.load_state_dict(model_dict)
            print('load success!')
    #model_dict=model.state_dict()
    best_error=100
    best_rate=100
    # it should be range(checkpoint[''epoch],args.n_epoch)
    #for epoch in range(trained, args.n_epoch):

       
    print('testing!')
    model.train()
    error_lin=[]
    error_log=[]
    error_va=[]
    error_rate=[]
    error_absrd=[]
    error_squrd=[]
    thre1=[]
    thre2=[]
    thre3=[]
    for i_val, (images_val, labels_val,segs) in tqdm(enumerate(valloader)):
        print(r'\n')
        images_val = Variable(images_val.cuda(), requires_grad=False)
        labels_val = Variable(labels_val.cuda(), requires_grad=False)
        with torch.no_grad():
            outputs = model(images_val)
            #outputs=segs
            pred = outputs.data.cpu().numpy()
            gt = labels_val.data.cpu().numpy()+1e-12
            ones=np.ones((gt.shape))
            zeros=np.zeros((gt.shape))
            pred=np.reshape(pred,(gt.shape))+1e-12
            #gt=np.reshape(gt,[4,480,640])
            dis=np.square(gt-pred)
            error_lin.append(np.sqrt(np.mean(dis)))
            dis=np.square(np.log(gt)-np.log(pred))
            error_log.append(np.sqrt(np.mean(dis)))
            alpha=np.mean(np.log(gt)-np.log(pred))
            dis=np.square(np.log(pred)-np.log(gt)+alpha)
            error_va.append(np.mean(dis)/2)
            dis=np.mean(np.abs(gt-pred))/gt
            error_absrd.append(np.mean(dis))
            dis=np.square(gt-pred)/gt
            error_squrd.append(np.mean(dis))
            thelt=np.where(pred/gt>gt/pred,pred/gt,gt/pred)
            thres1=1.25
            
            thre1.append(np.mean(np.where(thelt<thres1,ones,zeros)))
            thre2.append(np.mean(np.where(thelt<thres1*thres1,ones,zeros)))
            thre3.append(np.mean(np.where(thelt<thres1*thres1*thres1,ones,zeros)))
            #a=thre1[i_val]
            #error_rate.append(np.mean(np.where(dis<0.6,ones,zeros)))
            print("error_lin=%.4f,error_log=%.4f,error_va=%.4f,error_absrd=%.4f,error_squrd=%.4f,thre1=%.4f,thre2=%.4f,thre3=%.4f"%(
                error_lin[i_val],
                error_log[i_val],
                error_va[i_val],
                error_absrd[i_val],
                error_squrd[i_val],
                thre1[i_val],
                thre2[i_val],
                thre3[i_val]))
            #loss = loss_fn(input=outputs, target=labels_val)
            #print("Loss: %.4f" % (loss.item()))
    np.save('/home/lidong/Documents/RSDEN/RSDEN//error_train.npy',[error_lin[i_val],error_log[i_val],error_va[i_val],error_absrd[i_val],error_squrd[i_val],thre1[i_val],thre2[i_val],thre3[i_val]])
    error_lin=np.mean(error_lin)
    error_log=np.mean(error_log)
    error_va=np.mean(error_va)
    error_absrd=np.mean(error_absrd)
    error_squrd=np.mean(error_squrd)
    thre1=np.mean(thre1)
    thre2=np.mean(thre2)
    thre3=np.mean(thre3)

    print('Final Result!')
    print("error_lin=%.4f,error_log=%.4f,error_va=%.4f,error_absrd=%.4f,error_squrd=%.4f,thre1=%.4f,thre2=%.4f,thre3=%.4f"
        %(error_lin,error_log,error_va,error_absrd,error_squrd,thre1,thre2,thre3))
    #np.save('/home/lidong/Documents/RSDEN/RSDEN//error_train.npy',error)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rsnet',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='nyu',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=480,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=640,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=4000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=4,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu1_best_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=False,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
