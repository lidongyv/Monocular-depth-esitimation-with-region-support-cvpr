# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-15 15:57:11
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
    best_error=2
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True,
                           split='train_region', img_size=(args.img_rows, args.img_cols),task='all')
    v_loader = data_loader(data_path, is_transform=True,
                           split='test_region', img_size=(args.img_rows, args.img_cols),task='all')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=2)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        # old_window = vis.line(X=torch.zeros((1,)).cpu(),
        #                        Y=torch.zeros((1)).cpu(),
        #                        opts=dict(xlabel='minibatches',
        #                                  ylabel='Loss',
        #                                  title='Trained Loss',
        #                                  legend=['Loss']))
        loss_window1 = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss1',
                                         legend=['Loss1']))
        loss_window2 = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss2',
                                         legend=['Loss']))
        loss_window3 = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss3',
                                         legend=['Loss3']))                                                 
        pre_window1 = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='predict1!', caption='predict1.'),
        )
        pre_window2 = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='predict2!', caption='predict2.'),
        )
        pre_window3 = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='predict3!', caption='predict3.'),
        )

        ground_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='ground!', caption='ground.'),
        )
    cuda0=torch.device('cuda:0')
    cuda1=torch.device('cuda:1')
    cuda2=torch.device('cuda:2')
    cuda3=torch.device('cuda:3')
    # Setup Model
    rsnet = get_model('rsnet')
    rsnet = torch.nn.DataParallel(rsnet, device_ids=[0,1])
    rsnet.cuda(cuda0)
    drnet=get_model('drnet')
    drnet = torch.nn.DataParallel(drnet, device_ids=[2,3])
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
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_best_model.pkl')
        rsnet.load_state_dict(checkpoint['model_state'])
        trained=checkpoint['epoch']
        print('load success from rsnet %.d'%trained)
        best_error=checkpoint['error']
        checkpoint=torch.load('/home/lidong/Documents/RSDEN/RSDEN/drnet_nyu_best_model.pkl')
        drnet.load_state_dict(checkpoint['model_state'])
        #optimizer.load_state_dict(checkpoint['optimizer_state'])
        trained=checkpoint['epoch']
        print('load success from drnet %.d'%trained)
        trained=0
            




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

            for i_val, (images, labels,segments) in tqdm(enumerate(valloader)):
                #print(r'\n')
                images = images.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                print(i_val)

                with torch.no_grad():
                    region_support = rsnet(images)
                    coarse_depth=torch.cat([images,region_support],1)
                    coarse_depth=torch.cat([coarse_depth,region_support],1)
                    outputs=drnet(coarse_depth)
                    pred = outputs[2].data.cpu().numpy()
                    gt = labels.data.cpu().numpy()
                    ones=np.ones((gt.shape))
                    zeros=np.zeros((gt.shape))
                    pred=np.reshape(pred,(gt.shape))
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
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
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
