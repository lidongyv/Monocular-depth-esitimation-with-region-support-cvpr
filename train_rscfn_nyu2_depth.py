# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-10-23 15:38:21
import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from cluster_visual import *
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
from rsden.cluster_loss import *
from rsden.models import get_model
from rsden.loader import get_loader, get_data_path
from rsden.metrics import runningScore
from rsden.loss import *
from rsden.augmentations import *
import os
import cv2

def train(args):
    torch.backends.cudnn.benchmark=True
    # Setup Augmentations
    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])
    loss_rec=[]
    best_error=2
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True,
                           split='train', img_size=(args.img_rows, args.img_cols),task='region')
    v_loader = data_loader(data_path, is_transform=True,
                           split='test', img_size=(args.img_rows, args.img_cols),task='region')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=4, shuffle=False)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env='nyu2_depth')


        depth_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='depth!', caption='depth.'),
        )
        # cluster_window = vis.image(
        #     np.random.rand(480, 640),
        #     opts=dict(title='cluster!', caption='cluster.'),
        # )
        region_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='region!', caption='region.'),
        )
        ground_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='ground!', caption='ground.'),
        )
        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss',
                                         legend=['Loss']))
        # old_window = vis.line(X=torch.zeros((1,)).cpu(),
        #                Y=torch.zeros((1)).cpu(),
        #                opts=dict(xlabel='minibatches',
        #                          ylabel='Loss',
        #                          title='Trained Loss',
        #                          legend=['Loss']))   
        error_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='error',
                                 title='error',
                                 legend=['Error']))      
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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999),amsgrad=True)
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=args.l_rate,momentum=0.90, weight_decay=5e-4)
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = log_loss
    trained=0
    scale=100

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')
            #model_dict=model.state_dict()  
            #opt=torch.load('/home/lidong/Documents/RSDEN/RSDEN/exp1/l2/sgd/log/83/rsnet_nyu_best_model.pkl')
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            #opt=None
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            trained=checkpoint['epoch']
            best_error=checkpoint['error']
            print(best_error)
            print(trained)
            loss_rec=np.load('/home/lidong/Documents/RSCFN/loss.npy')
            loss_rec=list(loss_rec)
            loss_rec=loss_rec[:199*trained]
            test=0
            #exit()
            
    else:
        best_error=100
        best_error_r=100
        trained=0
        print('random initialize')
        
        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize from rsn!')
        rsn=torch.load('/home/lidong/Documents/RSCFN/rsn_cluster_nyu2_35_pretrain_best_model.pkl',map_location='cpu')
        model_dict=model.state_dict()  
        #print(model_dict)          
        #pre_dict={k: v for k, v in rsn['model_state'].items() if k in model_dict and rsn['model_state'].items()}
        pre_dict={k: v for k, v in rsn['model_state'].items() if k in model_dict and rsn['model_state'].items()}
        #print(pre_dict)
        key=[]
        for k,v in pre_dict.items():
            if v.shape!=model_dict[k].shape:
                key.append(k)
        for k in key:
            pre_dict.pop(k)
        #print(pre_dict)
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        #optimizer.load_state_dict(rsn['optimizer_state'])
        trained=rsn['epoch']

        print('load success!')

        print(trained)

        del rsn
        test=0
        trained=0
        #exit()
        

    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, args.n_epoch):
    #for epoch in range(0, args.n_epoch):
        #scheduler.step()
        #trained
        print('training!')
        model.train()
        
        for i, (images, labels,regions,segments) in enumerate(trainloader):
            #break
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            segments = Variable(segments.cuda())
            regions = Variable(regions.cuda())

            optimizer.zero_grad()

            #depth,feature,loss_var,loss_dis,loss_reg = model(images,segments)
            #depth,loss_var,loss_dis,loss_reg = model(images,segments)
            depth,mask,loss_var,loss_dis,loss_reg= model(images,segments,0,'train')

            loss_d=berhu(depth,labels)
            loss=torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg)
            loss=loss/4+loss_d
            #loss/=feature.shape[0]
            # depth = model(images,segments)
            # loss_d=berhu(depth,labels)
            #loss=loss_d
            lin=torch.mean(torch.abs(depth-labels))
            # loss=loss_d
            loss.backward()
            optimizer.step()
            #print(torch.mean(depth).item())
            if args.visdom:
                with torch.no_grad():
   
                    vis.line(
                        X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*199,
                        Y=loss.item()*torch.ones(1).cpu(),
                        win=loss_window,
                        update='append')
                    depth = depth.data.cpu().numpy().astype('float32')
                    depth = depth[0, :, :, :]
                    depth = (np.reshape(depth, [480, 640]).astype('float32')-np.min(depth))/(np.max(depth)-np.min(depth)+1)
                    vis.image(
                        depth,
                        opts=dict(title='depth!', caption='depth.'),
                        win=depth_window,
                    )

                    region = mask.data.cpu().numpy().astype('float32')
                    region = region[0,...]
                    region = (np.reshape(region, [480, 640]).astype('float32')-np.min(region))/(np.max(region)-np.min(region)+1)
                    vis.image(
                        region,
                        opts=dict(title='region!', caption='region.'),
                        win=region_window,
                    )                 
                    ground=labels.data.cpu().numpy().astype('float32')
                    ground = ground[0, :, :]
                    ground = (np.reshape(ground, [480, 640]).astype('float32')-np.min(ground))/(np.max(ground)-np.min(ground)+1)
                    vis.image(
                        ground,
                        opts=dict(title='ground!', caption='ground.'),
                        win=ground_window,
                    )
            loss_rec.append([i+epoch*199,torch.Tensor([loss.item()]).unsqueeze(0).cpu()])
            
            print("data [%d/199/%d/%d] Loss: %.4f loss_d: %.4f " % (i, epoch, args.n_epoch,loss.item(),lin.item()))
            # print("data [%d/199/%d/%d] Loss: %.4f linear: %.4f " % (i, epoch, args.n_epoch,loss.item(),lin.item()
            #                    ))
        
        if epoch>40:
            check=3
            #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
        else:
            check=5
            #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=1)
        if epoch>70:
            check=2
            #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.25)
        if epoch>90:
            check=1
        check=1
            #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)  
        #epoch=3          
        if epoch%check==0:
                
            print('testing!')
            model.train()
            loss_ave=[]
            loss_d_ave=[]
            loss_lin_ave=[]
            loss_r_ave=[]
            for i_val, (images_val, labels_val,regions,segments) in tqdm(enumerate(valloader)):
                #print(r'\n')
                images_val = Variable(images_val.cuda(), requires_grad=False)
                labels_val = Variable(labels_val.cuda(), requires_grad=False)
                segments_val = Variable(segments.cuda(), requires_grad=False)
                regions_val = Variable(regions.cuda(), requires_grad=False)
                with torch.no_grad():
                    depth,mask,loss_var,loss_dis,loss_reg= model(images_val,segments_val,0,'test')
                    loss_d=berhu(depth,labels_val)
                    loss=torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg)
                    # loss=loss+loss_d
                    loss=loss_d+loss/4
                    lin=torch.mean(torch.abs(depth-labels_val))
                    loss_ave.append(loss.data.cpu().numpy())
                    print('error:')
                    print(loss_ave[-1])

                    print("loss_d=%.4f loss=%.4f"%(lin.item(),loss.item()))
                if args.visdom:
                    vis.line(
                        X=torch.ones(1).cpu() * i_val+torch.ones(1).cpu() *test*163,
                        Y=lin.item()*torch.ones(1).cpu(),
                        win=error_window,
                        update='append')

            error=np.mean(loss_ave)
            #error_d=np.mean(loss_d_ave)
            #error_lin=np.mean(loss_lin_ave)
            #error_rate=np.mean(error_rate)
            print("error_r=%.4f"%(error))
            test+=1

            if error<= best_error:
                best_error = error
                state = {'epoch': epoch+1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error,
                         }
                torch.save(state, "depth_{}_{}_{}_best_model.pkl".format(
                    args.arch, args.dataset,str(epoch)))
                print('save success')
            np.save('/home/lidong/Documents/RSCFN/loss.npy',loss_rec)


        # if epoch%15==0:
        #     #best_error = error
        #     state = {'epoch': epoch+1,
        #              'model_state': model.state_dict(),
        #              'optimizer_state': optimizer.state_dict(), 
        #              'error': error_r,
        #              'error_r': error_r,}
        #     torch.save(state, "{}_{}_{}_model.pkl".format(
        #         args.arch, args.dataset,str(epoch)))
        #     print('save success')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rsn_cluster',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='nyu2',
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
    parser.add_argument('--resume', nargs='?', type=str, default=r'/home/lidong/Documents/RSCFN/rsn_cluster_nyu2_35_pretrain_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSDEN/RSDEN/depth_rsn_cluster_nyu2_best_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
