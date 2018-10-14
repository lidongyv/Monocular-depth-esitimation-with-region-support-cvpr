# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-19 00:23:09
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
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
    best_error=2
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True,
                           split='train_region', img_size=(args.img_rows, args.img_cols),task='visualize')
    v_loader = data_loader(data_path, is_transform=True,
                           split='visual', img_size=(args.img_rows, args.img_cols),task='visualize')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=2)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization

    cuda0=torch.device('cuda:0')
    cuda1=torch.device('cuda:1')
    cuda2=torch.device('cuda:2')
    cuda3=torch.device('cuda:3')
    # Setup Model
    rsnet = get_model('rsnet')
    rsnet = torch.nn.DataParallel(rsnet, device_ids=[0])
    rsnet.cuda(cuda0)
    drnet=get_model('drnet')
    drnet = torch.nn.DataParallel(drnet, device_ids=[2])
    drnet.cuda(cuda2)
    parameters=list(rsnet.parameters())+list(drnet.parameters())
    # Check if model has custom optimizer / loss
    # modify to adam, modify the learning rate
    if hasattr(drnet.module, 'optimizer'):
        optimizer = drnet.module.optimizer
    else:
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999))
        optimizer = torch.optim.SGD(
            parameters, lr=args.l_rate,momentum=0.99, weight_decay=5e-4)
    if hasattr(rsnet.module, 'loss'):
        print('Using custom loss')
        loss_fn = rsnet.module.loss
    else:
        loss_fn = l1_r
    trained=0
    scale=100

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #model_dict=model.state_dict()  
            #opt=torch.load('/home/lidong/Documents/RSDEN/RSDEN/exp1/l2/sgd/log/83/rsnet_nyu_best_model.pkl')
            model.load_state_dict(checkpoint['model_state'])
            #optimizer.load_state_dict(checkpoint['optimizer_state'])
            #opt=None
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            trained=checkpoint['epoch']
            best_error=checkpoint['error']
            
            #print('load success!')
            loss_rec=np.load('/home/lidong/Documents/RSDEN/RSDEN/loss.npy')
            loss_rec=list(loss_rec)
            loss_rec=loss_rec[:1632*trained]
            # for i in range(300):
            #     loss_rec[i][1]=loss_rec[i+300][1]
            for l in range(int(len(loss_rec)/1632)):
                if args.visdom:
                    
                    vis.line(
                        X=torch.ones(1).cpu() * loss_rec[l*1632][0],
                        Y=np.mean(np.array(loss_rec[l*1632:(l+1)*1632])[:,1])*torch.ones(1).cpu(),
                        win=old_window,
                        update='append')
            
    else:

        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize seperately!')
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/exp1/region/trained/rsnet_nyu_best_model.pkl')
        rsnet.load_state_dict(checkpoint['model_state'])
        trained=checkpoint['epoch']
        print('load success from rsnet %.d'%trained)
        best_error=checkpoint['error']
        checkpoint=torch.load('//home/lidong/Documents/RSDEN/RSDEN/exp1/seg/drnet_nyu_best_model.pkl')
        drnet.load_state_dict(checkpoint['model_state'])
        #optimizer.load_state_dict(checkpoint['optimizer_state'])
        trained=checkpoint['epoch']
        print('load success from drnet %.d'%trained)
        trained=0
            
    min_loss=10
    samples=[]


    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, args.n_epoch):

        rsnet.train()
        drnet.train()
           
   
        if epoch%1==0:    
            print('testing!')
            rsnet.train()
            drnet.train()
            error_lin=[]
            error_log=[]
            error_va=[]
            error_rate=[]
            error_absrd=[]
            error_squrd=[]
            thre1=[]
            thre2=[]
            thre3=[]

            for i_val, (images, labels,segments,sample) in tqdm(enumerate(valloader)):
                #print(r'\n')
                images = images.cuda(cuda2)
                labels = labels.cuda(cuda2)
                segments=segments.cuda(cuda2)
                optimizer.zero_grad()
                #print(i_val)

                with torch.no_grad():
                    #region_support = rsnet(images)
                    coarse_depth=torch.cat([images,segments],1)
                    #coarse_depth=torch.cat([coarse_depth,segments],1)
                    outputs=drnet(coarse_depth)
                    #print(outputs[2].item())
                    pred = [outputs[0].data.cpu().numpy(),outputs[1].data.cpu().numpy(),outputs[2].data.cpu().numpy()]
                    pred=np.array(pred)
                    #print(pred.shape)
                    #pred=region_support.data.cpu().numpy()
                    gt = labels.data.cpu().numpy()
                    ones=np.ones((gt.shape))
                    zeros=np.zeros((gt.shape))
                    pred=np.reshape(pred,(gt.shape[0],gt.shape[1],gt.shape[2],3))
                    #pred=np.reshape(pred,(gt.shape))
                    print(np.max(pred))
                    #print(gt.shape)
                    #print(pred.shape)
                    #gt=np.reshape(gt,[4,480,640])
                    dis=np.square(gt-pred[:,:,:,2])
                    #dis=np.square(gt-pred)
                    loss=np.sqrt(np.mean(dis))
                    #print(min_loss)
                    if min_loss>0:
                        #print(loss)
                        min_loss=loss
                        #pre=pred[:,:,0]
                        #region_support=region_support.item()
                        #rgb=rgb
                        #segments=segments
                        #labels=labels.item()
                        #sample={'loss':loss,'rgb':rgb,'region_support':region_support,'ground_r':segments,'ground_d':labels}
                        #samples.append(sample)
                        #pred=pred.item()
                        #pred=pred[0,:,:]
                        #pred=pred/np.max(pred)*255
                        #pred=pred.astype(np.uint8)
                        #print(pred.shape)
                        #cv2.imwrite('/home/lidong/Documents/RSDEN/RSDEN/exp1/pred/seg%.d.png'%(i_val),pred)
                        np.save('/home/lidong/Documents/RSDEN/RSDEN/exp1/pred/seg%.d.npy'%(i_val),pred)
                        np.save('/home/lidong/Documents/RSDEN/RSDEN/exp1/visual/seg%.d.npy'%(i_val),sample)
            break;








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rsdin',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='nyu',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=480,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=640,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=4000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_30_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
