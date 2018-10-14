# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-08-06 10:36:36
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
        v_loader, batch_size=args.batch_size, num_workers=4)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()


        depth_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='depth!', caption='depth.'),
        )
        cluster_window = vis.image(
            np.random.rand(480, 640),
            opts=dict(title='cluster!', caption='cluster.'),
        )
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
        old_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss',
                                 title='Trained Loss',
                                 legend=['Loss']))      
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
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.l_rate,momentum=0.90, weight_decay=5e-4)
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
            #best_error_d=checkpoint['error_d']
            best_error_d=checkpoint['error_d']
            print(best_error)
            print(trained)
            loss_rec=np.load('/home/lidong/Documents/RSDEN/RSDEN/loss.npy')
            loss_rec=list(loss_rec)
            loss_rec=loss_rec[:179*trained]
            # for i in range(300):
            #     loss_rec[i][1]=loss_rec[i+300][1]
            for l in range(int(len(loss_rec)/179)):
                if args.visdom:
                    
                    vis.line(
                        X=torch.ones(1).cpu() * loss_rec[l*179][0],
                        Y=np.mean(np.array(loss_rec[l*179:(l+1)*179])[:,1])*torch.ones(1).cpu(),
                        win=old_window,
                        update='append')
            #exit()
            
    else:
        best_error=100
        best_error_d=100
        trained=0
        print('random initialize')
        """
        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize from rsn!')
        rsn=torch.load('/home/lidong/Documents/RSDEN/RSDEN/depth_rsn_cluster_nyu2_best_model.pkl',map_location='cpu')
        model_dict=model.state_dict()  
        #print(model_dict)          
        #pre_dict={k: v for k, v in rsn['model_state'].items() if k in model_dict and rsn['model_state'].items()}
        pre_dict={k: v for k, v in rsn.items() if k in model_dict and rsn.items()}
        key=[]
        for k,v in pre_dict.items():
            if v.shape!=model_dict[k].shape:
                key.append(k)
        for k in key:
            pre_dict.pop(k)
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        #trained=rsn['epoch']
        #best_error=rsn['error']
        #best_error_d=checkpoint['error_d']
        #best_error_d=rsn['error_d']
        print('load success!')
        print(best_error)
        print(trained)
        print(best_error_d)
        del rsn
        """
        

    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, args.n_epoch):
    #for epoch in range(0, args.n_epoch):
        
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

            # depth,feature,loss_var,loss_dis,loss_reg = model(images,segments)
            # loss_d=l2(depth,labels)
            # loss=torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg)
            # loss=loss/4+loss_d
            # loss/=2
            depth = model(images,segments)
            loss_d=berhu(depth,labels)
            lin=l2(depth,labels)
            loss=loss_d
            loss.backward()
            optimizer.step()
            if loss.item()<=0.000001:
                feature = feature.data.cpu().numpy().astype('float32')[0,...]
                feature=np.reshape(feature,[1,feature.shape[0],feature.shape[1],feature.shape[2]])
                feature=np.transpose(feature,[0,2,3,1])
                print(feature.shape)
                #feature = feature[0,...]
                masks=get_instance_masks(feature, 0.7)
                print(masks.shape)
                #cluster = masks[0]
                cluster=np.sum(masks,axis=0)
                cluster = (np.reshape(cluster, [480, 640]).astype('float32')-np.min(cluster))/(np.max(cluster)-np.min(cluster)+1)

                vis.image(
                    cluster,
                    opts=dict(title='cluster!', caption='cluster.'),
                    win=cluster_window,
                )                
            if args.visdom:
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*179,
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

                region = regions.data.cpu().numpy().astype('float32')
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
            loss_rec.append([i+epoch*179,torch.Tensor([loss.item()]).unsqueeze(0).cpu()])

            # print("data [%d/179/%d/%d] Loss: %.4f loss_var: %.4f loss_dis: %.4f loss_reg: %.4f loss_d: %.4f" % (i, epoch, args.n_epoch,loss.item(), \
            #                     torch.sum(loss_var).item()/4,torch.sum(loss_dis).item()/4,0.001*torch.sum(loss_reg).item()/4,loss_d.item()))
            print("data [%d/179/%d/%d] Loss: %.4f linear: %.4f " % (i, epoch, args.n_epoch,loss.item(),lin.item()
                               ))
           
        if epoch>30:
            check=3
        else:
            check=5
        if epoch>50:
            check=2
        if epoch>70:
            check=1   
        #epoch=3          
        if epoch%check==0:
                
            print('testing!')
            model.eval()
            loss_ave=[]
            loss_d_ave=[]
            loss_lin_ave=[]
            for i_val, (images_val, labels_val,regions,segments) in tqdm(enumerate(valloader)):
                #print(r'\n')
                images_val = Variable(images_val.cuda(), requires_grad=False)
                labels_val = Variable(labels_val.cuda(), requires_grad=False)
                segments_val = Variable(segments.cuda(), requires_grad=False)
                regions_val = Variable(regions.cuda(), requires_grad=False)
                with torch.no_grad():

                    #depth,feature,loss_var,loss_dis,loss_reg = model(images_val,segments_val)
                    depth = model(images_val,segments_val)
                    # loss=torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg)
                    # loss=loss/4
                    loss_d = log_loss(input=depth, target=labels_val)
                    loss_d=torch.sqrt(loss_d)
                    loss_lin=l2(depth,labels_val)
                    loss_lin=torch.sqrt(loss_lin)
                    # loss_r=(loss+loss_d)/2
                    # loss_ave.append(loss_r.data.cpu().numpy())
                    loss_d_ave.append(loss_d.data.cpu().numpy())
                    loss_lin_ave.append(loss_lin.data.cpu().numpy())
                    print('error:')
                    print(loss_d_ave[-1])
                    # print(loss_ave[-1])
                    print(loss_lin_ave[-1])
                    #exit()

                    # feature = feature.data.cpu().numpy().astype('float32')[0,...]
                    # feature=np.reshape(feature,[1,feature.shape[0],feature.shape[1],feature.shape[2]])
                    # feature=np.transpose(feature,[0,2,3,1])
                    # #print(feature.shape)
                    # #feature = feature[0,...]
                    # masks=get_instance_masks(feature, 0.7)
                    # #print(len(masks))
                    # cluster = np.array(masks)
                    # cluster=np.sum(masks,axis=0)
                    # cluster = np.reshape(cluster, [480, 640]).astype('float32')/255

                    # vis.image(
                    #     cluster,
                    #     opts=dict(title='cluster!', caption='cluster.'),
                    #     win=cluster_window,
                    # ) 
                    # ground=segments.data.cpu().numpy().astype('float32')
                    # ground = ground[0, :, :]
                    # ground = (np.reshape(ground, [480, 640]).astype('float32')-np.min(ground))/(np.max(ground)-np.min(ground)+1)
                    # vis.image(
                    #     ground,
                    #     opts=dict(title='ground!', caption='ground.'),
                    #     win=ground_window,
                    # ) 
            #error=np.mean(loss_ave)
            error_d=np.mean(loss_d_ave)
            error_lin=np.mean(loss_lin_ave)
            #error_rate=np.mean(error_rate)
            print("error_d=%.4f error_lin=%.4f"%(error_d,error_lin))
            #exit()
            #continue
            # if error_d<= best_error:
            #     best_error = error
            #     state = {'epoch': epoch+1,
            #              'model_state': model.state_dict(),
            #              'optimizer_state': optimizer.state_dict(),
            #              'error': error,
            #              'error_d': error_d,
            #              }
            #     torch.save(state, "{}_{}_best_model.pkl".format(
            #         args.arch, args.dataset))
            #     print('save success')
            # np.save('/home/lidong/Documents/RSDEN/RSDEN/loss.npy',loss_rec)
            if error_lin<= best_error:
                best_error = error_lin
                state = {'epoch': epoch+1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error_lin,
                         'error_d': error_d,
                         }
                torch.save(state, "depth_{}_{}_best_model.pkl".format(
                    args.arch, args.dataset))
                print('save success')
            np.save('/home/lidong/Documents/RSDEN/RSDEN/loss.npy',loss_rec)
        if epoch%15==0:
            #best_error = error
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), 
                     'error': error_lin,
                     'error_d': error_d,}
            torch.save(state, "depth_{}_{}_{}_model.pkl".format(
                args.arch, args.dataset,str(epoch)))
            print('save success')





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
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/RSDEN/RSDEN/depth_rsn_cluster_nyu2_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSDEN/RSDEN/depth_rsn_cluster_nyu2_best_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
