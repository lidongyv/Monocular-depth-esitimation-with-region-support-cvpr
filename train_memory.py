# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-12-27 16:54:04
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
alpha=0.7132995128631592
#alpha=0.7
#alphat=0.3
beta=9.99547004699707

scale=2
def train(args):
    scale=2
    cuda_id=0
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

    train_len=t_loader.length/args.batch_size
    trainloader = data.DataLoader(
        t_loader, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=True)
    valloader = data.DataLoader(
        v_loader, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=False)


    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env='nyu_proup_refine')


        proup_refine_window = vis.image(
            np.random.rand(228, 304),
            opts=dict(title='depth!', caption='depth.'),
        )
        accurate_window = vis.image(
            np.random.rand(228, 304),
            opts=dict(title='accurate!', caption='accurate.'),
        )

        ground_window = vis.image(
            np.random.rand(228, 304),
            opts=dict(title='ground!', caption='ground.'),
        )
        image_window = vis.image(
            np.random.rand(228, 304),
            opts=dict(title='img!', caption='img.'),
        )
        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Training Loss',
                                         legend=['Loss']))
        lin_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='error',
                                 title='linear Loss',
                                 legend=['linear error']))   
        error_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='error',
                                 title='error',
                                 legend=['Error']))      
    # Setup Model
    model = get_model(args.arch)
    memory=get_model(memory)
    # model = torch.nn.DataParallel(
    #     model, device_ids=range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(
        model, device_ids=[0,1,2,3])
    model.cuda()
    memory = torch.nn.DataParallel(
        memory, device_ids=[0,1,2,3])
    memory.cuda()
    # Check if model has custom optimizer / loss
    # modify to adam, modify the learning rate
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.l_rate,betas=(0.9,0.999),amsgrad=True)
        optimizer2 = torch.optim.Adam(
            memory.parameters(), lr=args.l_rate,betas=(0.9,0.999),amsgrad=True)
    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = log_loss
    trained=0
    #scale=100

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cpu')
            #model_dict=model.state_dict()  
            #opt=torch.load('/home/lidong/Documents/RSDEN/RSDEN/exp1/l2/sgd/log/83/rsnet_nyu_best_model.pkl')
            model.load_state_dict(checkpoint['model_state'])
            #optimizer.load_state_dict(checkpoint['optimizer_state'])
            #opt=None
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            trained=checkpoint['epoch']
            best_error=checkpoint['error']+0.1
            mean_loss=best_error/2
            print(best_error)
            print(trained)
            # loss_rec=np.load('/home/lidong/Documents/RSCFN/loss.npy')
            # loss_rec=list(loss_rec)
            # loss_rec=loss_rec[:train_len*trained]
            test=0
            #exit()
            #trained=0
            
    else:
        best_error=100
        best_error_r=100
        trained=0
        mean_loss=1.0
        print('random initialize')
        
        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize from rsn!')
        rsn=torch.load('/home/lidong/Documents/RSCFN/proup_refine_rsn_cluster_nyu_0_0.59483826_coarse_best_model.pkl',map_location='cpu')
        model_dict=model.state_dict()  
        #print(model_dict)          
        pre_dict={k: v for k, v in rsn['model_state'].items() if k in model_dict and rsn['model_state'].items()}
        #pre_dict={k: v for k, v in rsn.items() if k in model_dict and rsn.items()}
        #print(pre_dict)
        key=[]
        for k,v in pre_dict.items():
            if v.shape!=model_dict[k].shape:
                key.append(k)
        for k in key:
            pre_dict.pop(k)
        #print(pre_dict)
        # pre_dict['module.regress1.0.conv1.1.weight']=pre_dict['module.regress1.0.conv1.1.weight'][:,:256,:,:]
        # pre_dict['module.regress1.0.downsample.1.weight']=pre_dict['module.regress1.0.downsample.1.weight'][:,:256,:,:]
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        #optimizer.load_state_dict(rsn['optimizer_state'])
        trained=rsn['epoch']
        best_error=rsn['error']+0.5
        mean_loss=best_error/2
        print('load success!')
        print(best_error)
        #best_error+=1
        #del rsn
        test=0
        trained=0
        # loss_rec=np.load('/home/lidong/Documents/RSCFN/loss.npy')
        # loss_rec=list(loss_rec)
        # loss_rec=loss_rec[:train_len*trained]
        #exit()
        
    zero=torch.zeros(1).cuda()
    one=torch.ones(1).cuda()
    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, args.n_epoch):
    #for epoch in range(0, args.n_epoch):
        #scheduler.step()
        #trained

        print('training!')
        model.train()
        loss_error=0
        loss_error_d=0
        mean_loss_ave=[]
        memory_bank=0
        for i, (images, labels,regions,segments,image,index) in enumerate(trainloader):

            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
            segments = Variable(segments.cuda(0))
            regions = Variable(regions.cuda(0))
            index = Variable(index.cuda(0))
            iterative_count=0
            if epoch==trained:
                with torch.no_grad():
                    optimizer.zero_grad()
                    optimizer2.zero_grad()
                    feature,accurate = model(images,regions,labels,0,'train')
                    feature=feature.detach()
                    representation=memory(feature)
                    labels=labels.view_as(accurate)
                    segments=segments.view_as(accurate)
                    regions=regions.view_as(accurate)3
                    mask=(labels>alpha)&(labels<beta)
                    mask=mask.float().detach()
                    loss_a=berhu(accurate,labels,mask)
                    loss=loss_a
                    accurate=torch.where(accurate>beta,beta*one,accurate)
                    accurate=torch.where(accurate<alpha,alpha*one,accurate)
                    lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels,2),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1)))
                    log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels)),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1))
                    #loss.backward()
                    # optimizer.step()
                    # optimizer2.step()
                    if memory_bank==0:
                        memory_bank=representation
                        index_bank=index
                        loss_bank=lin
                    else:
                        memory_bank=torch.cat([memory_bank,representation],dim=0)
                        index_bank=torch.cat([index_bank,index],dim=0)
                        loss_bank=torch.cat([loss_bank,lin],dim=0)
            else:
                optimizer.zero_grad()
                optimizer2.zero_grad()
                feature,accurate = model(images,regions,labels,0,'train')
                feature=feature.detach()
                representation=memory(feature)
                labels=labels.view_as(accurate)
                segments=segments.view_as(accurate)
                regions=regions.view_as(accurate)3
                mask=(labels>alpha)&(labels<beta)
                mask=mask.float().detach()
                loss_a=berhu(accurate,labels,mask)
                loss=loss_a
                accurate=torch.where(accurate>beta,beta*one,accurate)
                accurate=torch.where(accurate<alpha,alpha*one,accurate)
                lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels,2),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1)))
                log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels)),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1))
                loss.backward()
                #optimizer.step()
                loss_m=memory_loss(representation,re_repre,lin,re_loss)
                optimizer.step()
                if memory_bank==0:
                    memory_bank=representation
                    index_bank=index
                    loss_bank=lin
                else:
                    memory_bank=torch.cat([memory_bank,representation],dim=0)
                    index_bank=torch.cat([index_bank,index],dim=0)
                    loss_bank=torch.cat([loss_bank,lin],dim=0)
            loss_rec.append([i+epoch*train_len,torch.Tensor([loss.item()]).unsqueeze(0).cpu()])
            loss_error+=lin.item()

            loss_error_d+=log_d.item()
            print("data [%d/%d/%d/%d] Loss: %.4f lin: %.4f lin_d:%.4f loss_d:%.4f loss_a:%.4f loss_var:%.4f loss_dis:%.4f loss_reg: %.4f" % (i,train_len, epoch, args.n_epoch,loss.item(),lin.item(),lin_d.item(), loss_d.item(),loss_a.item(), \
                torch.sum(loss_v).item(),torch.sum(loss_dis).item(),0.001*torch.sum(loss_reg).item()))

        if epoch==trained:
            sigma=torch.mean(loss_bank)/train_len*10
            re_index=[]
            re_loss=[]
            re_repre=[]
            while(loss_bank.shape[0]>len(re_index)):
                candidate=loss_bank.non_zero()
                t_index=candidate[torch.randint(high=candidate.shape[0])]
                t_loss=loss_bank[t_loss]
                while(True):
                    t_related=torch.where(torch.abs(loss_bank-t_loss)<sigma,one,zero).non_zero()
                    t_loss2=torch.mean(loss_bank[t_related])
                    if t_loss==t_loss2:
                        t_loss=t_loss2
                        break
                re_index.append(index_bank[t_related])
                re_loss.append(torch.mean(loss_bank[t_related]))
                re_repre.append(torch.mean(memory_bank[t_related],dim=1))
                loss_bank[t_related]=0
            re_index=re_index
            re_loss=torch.stack(re_loss)
            re_repre=torch.stack(re_repre)

        mean_loss=np.mean(mean_loss_ave)
        print("mean_loss:%.4f"%(mean_loss))
        if epoch>50:
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
            #scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)  
        check=1
        #epoch=10
        if epoch%check==0:
                
            print('testing!')
            model.eval()
            loss_ave=[]
            loss_d_ave=[]
            loss_lin_ave=[]
            loss_log_ave=[]
            loss_r_ave=[]
            error_sum=0
            for i_val, (images_val, labels_val,regions,segments,images) in tqdm(enumerate(valloader)):
                #print(r'\n')
                images_val = Variable(images_val.cuda(0), requires_grad=False)
                labels_val = Variable(labels_val.cuda(0), requires_grad=False)
                segments_val = Variable(segments.cuda(0), requires_grad=False)
                regions_val = Variable(regions.cuda(0), requires_grad=False)
                model_t=model
                with torch.no_grad():
                    feature,accurate = model_t(images_val,regions,labels,0,'train')
                    feature=feature.detach()
                    representation=memory(feature)
                    labels_val=labels_val.view_as(accurate)
                    target_index=torch.argmax(torch.nn.functional.softmax(-torch.sum(torch.pow(re_repre-representation,2),dim=1),dim=0))
                    retrain_samples=re_index[target_index]
                    rt_loader = data_loader(data_path, is_transform=True,
                       split='train', img_size=(args.img_rows, args.img_cols),task='region',index_bank=retrain_samples)
                    rtrainloader = data.DataLoader(
                        t_loader, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=True)
                while(True):
                    loss_t=0
                    for i, (images, labels,regions,segments,image,index) in enumerate(rtrainloader):
                        images = Variable(images.cuda(0))
                        labels = Variable(labels.cuda(0))
                        segments = Variable(segments.cuda(0))
                        regions = Variable(regions.cuda(0))
                        index = Variable(index.cuda(0))
                        iterative_count=0
                        optimizer.zero_grad()
                        feature,accurate = model_t(images,regions,labels,0,'train')
                        labels=labels.view_as(accurate)
                        segments=segments.view_as(accurate)
                        regions=regions.view_as(accurate)3
                        mask=(labels>alpha)&(labels<beta)
                        mask=mask.float().detach()
                        loss_a=berhu(accurate,labels,mask)
                        loss=loss_a
                        accurate=torch.where(accurate>beta,beta*one,accurate)
                        accurate=torch.where(accurate<alpha,alpha*one,accurate)
                        lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels,2),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1)))
                        log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels)),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1))
                        loss.backward()
                        optimizer.step()
                        loss_t+=loss*images.shape[0]
                    loss_t/=len(retrain_samples)
                    if loss_t<mean_loss:
                        break
                feature,accurate = model(images_val,regions,labels,0,'train')
                feature=feature.detach()
                representation=memory(feature)
                labels_val=labels_val.view_as(accurate)
                segments=segments.view_as(accurate)
                regions=regions.view_as(accurate)3
                mask=(labels_val>alpha)&(labels_val<beta)
                mask=mask.float().detach()
                loss_a=berhu(accurate,labels_val,mask)
                loss=loss_a
                accurate=torch.where(accurate>beta,beta*one,accurate)
                accurate=torch.where(accurate<alpha,alpha*one,accurate)
                lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1)))
                log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels_val)),mask).view(labels_val.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1))
                loss_ave.append(lin.item())
                loss_d_ave.append(lin.item())
                loss_log_ave.append(log_d.item())
            error=np.mean(loss_ave)

            print("error_r=%.4f,error_d=%.4f,error_log=%.4f"%(error,np.mean(loss_d_ave),np.mean(loss_log_ave)))
            test+=1
            print(error_sum/654)
            if error<= best_error:
                best_error = error
                state = {'epoch': epoch+1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error,
                         }
                torch.save(state, "proup_refine_{}_{}_{}_{}_coarse_best_model.pkl".format(
                    args.arch, args.dataset,str(epoch),str(error)))
                print('save success')
            np.save('/home/lidong/Documents/RSCFN/loss.npy',loss_rec)
            #exit()

        if epoch%3==0:
            #best_error = error
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), 
                     'error': error,}
            torch.save(state, "proup_refine_{}_{}_{}_ceoarse_model.pkl".format(
                args.arch, args.dataset,str(epoch)))
            print('save success')



if __name__ == '__main__':
    scale=2
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rsn_cluster',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='nyu2',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=228,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=304,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=4000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/RSCFN/proup_refine_rsn_cluster_nyu_19_0.5952838_coarse_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSCFN/proup_refine_rsn_cluster_nyu_27_ceoarse_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=False,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
