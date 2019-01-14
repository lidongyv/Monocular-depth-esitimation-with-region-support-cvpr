# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2019-01-11 11:13:05
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
        vis = visdom.Visdom(env='nyu_memory_retrain')


        memory_retrain_window = vis.image(
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
    # model = torch.nn.DataParallel(
    #     model, device_ids=range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(
        model, device_ids=[1])
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda(1)

    # Check if model has custom optimizer / loss
    # modify to adam, modify the learning rate
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.l_rate,betas=(0.9,0.999),amsgrad=True)
        # optimizer = torch.optim.SGD(
        #     model.parameters(), lr=args.l_rate,momentum=0.90)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)
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
            mean_loss=checkpoint['mean_loss']
            #mean_loss=checkpoint['error']
            print(best_error)
            print(trained)
            print(mean_loss)
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
        mean_loss=10.0
        print('random initialize')
        
        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize from rsn!')
        rsn=torch.load('/home/lidong/Documents/RSCFN/memory_retrain_rsn_cluster_nyu_4_0.5681759_coarse_best_model.pkl',map_location='cpu')
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
        
    zero=torch.zeros(1).cuda(1)
    one=torch.ones(1).cuda(1)
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
        for i, (images, labels,regions,segments,image) in enumerate(trainloader):
            #break
            # if i==100:
            #     break
            
            images = Variable(images.cuda(1))
            labels = Variable(labels.cuda(1))
            segments = Variable(segments.cuda(1))
            regions = Variable(regions.cuda(1))

            iterative_count=0
            while(True):
                #mask = (labels > 0)
                optimizer.zero_grad()

                depth,accurate,loss_var,loss_dis,loss_reg = model(images,regions,labels,0,'train')
                #depth,loss_var,loss_dis,loss_reg = model(images,segments)
                #depth,masks,loss_var,loss_dis,loss_reg = model(images,segments,1,'train')
                # depth,accurate = model(images,regions,0,'eval')
                labels=labels.view_as(depth)
                segments=segments.view_as(depth)
                regions=regions.view_as(depth)
                mask=(labels>alpha)&(labels<beta)
                mask=mask.float().detach()
                #print(torch.sum(mask))

                # print('depth',torch.mean(depth).item(),torch.min(depth).item(),torch.max(depth).item())
                # print('accurate',torch.mean(accurate).item(),torch.min(accurate).item(),torch.max(accurate).item())
                # print('ground',torch.mean(labels).item(),torch.min(labels).item(),torch.max(labels).item())
                loss_d=berhu(depth,labels,mask)
                #loss_d=relative_loss(depth,labels,mask)
                #loss_i=berhu_log(intial,labels)
                loss_a=berhu(accurate,labels,mask)
                loss_v=v_loss(accurate,depth,labels,mask)
                #print(depth.requires_grad)
                print('mean_variance:%.4f,max_variance:%.4f'%((torch.sum(torch.abs(accurate-depth))/torch.sum(mask)).item(),torch.max(torch.abs(accurate-depth)).item()))
                #loss_a=relative_loss(accurate,labels,mask)
                #loss_d=log_loss(depth,labels)
                # loss_a=log_loss(depth[mask],labels[mask])
                # loss_d=log_loss(accurate[mask],labels[mask])
                # if epoch<30:
                #     loss=0.3*loss_d+0.7*loss_a+loss_v
                # else:
                #     loss=loss_a
                #loss=0.3*loss_d+0.35*loss_a+0.35*loss_v
                #loss=0.7*loss_a+0.4*loss_d-0.1*loss_v
                #loss=loss_a+0.3*loss_d+0.1*(loss_a-loss_d)+0.5*loss_v
                loss=loss_a+0.3*loss_d+0.3*loss_v
                #loss=loss_a
                #mask=mask.float()
                #mask=(labels>alpha)&(labels<beta)&(labels<torch.max(labels))&(labels>torch.min(labels))
                #loss=loss+0.5*(torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg))
                #loss=loss/4+loss_d
                #loss/=feature.shape[0]
                # depth = model(images,segments)
                # loss_d=berhu(depth,labels)
                #lin=torch.sqrt(torch.mean(torch.pow(accurate[mask]-labels[mask],2)))
                #print(torch.min(accurate),torch.max(accurate))
                #exit()

                accurate=torch.where(accurate>beta,beta*one,accurate)
                accurate=torch.where(accurate<alpha,alpha*one,accurate)
                labels=torch.where(labels>beta,beta*one,labels)
                labels=torch.where(labels<alpha,alpha*one,labels)
                depth=torch.where(depth>beta,beta*one,depth)
                depth=torch.where(depth<alpha,alpha*one,depth)
                lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels,2),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1)))
                lin_d=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(depth-labels,2),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1)))
                lin=lin.detach()
                #print(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels,2),mask).view(labels.shape[0],-1),dim=-1)/torch.sum(mask.view(labels.shape[0],-1),dim=-1)))
                #log_d=torch.sqrt(torch.mean(torch.pow(torch.log10(depth[mask])-torch.log10(labels[mask]),2)))
                log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels)),mask).view(labels.shape[0],-1),dim=-1)/(torch.sum(mask.view(labels.shape[0],-1),dim=-1)+1))
                
                #print(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(torch.log10(labels)-torch.log10(accurate),2),mask).view(labels.shape[0],-1),dim=-1)/torch.sum(mask.view(labels.shape[0],-1),dim=-1)))
                #print(torch.sum(mask.view(labels.shape[0],-1),dim=-1))
                #accurate=torch.where(accurate>torch.mean(accurate)*4,torch.mean(accurate)*4,accurate)
                #depth=torch.where(depth>torch.mean(depth)*4,torch.mean(accurate)*4,depth)
                #exit()
                # loss.backward()
                # mean_loss_ave.append(lin.item())
                # optimizer.step()
                # break
                if epoch<=trained+2:

                    loss.backward()
                    mean_loss_ave.append(lin.item())
                    optimizer.step()
                    break
                if (lin<=mean_loss) :
                    #loss_bp=loss*torch.pow(100,-(mean_loss-lin)/mean_loss)
                    #loss_bp=loss*zero
                    print('no back')
                    loss=0.1*loss
                    #optimizer.step()
                    loss.backward()
                    mean_loss_ave.append(lin.item())
                    optimizer.step()
                    break
                else:
                    print(torch.pow(10,torch.min(one,(lin-mean_loss)/mean_loss)).item())
                    print('back')
                    #loss_bp=loss*torch.pow(10,torch.min(one,(lin-mean_loss)/mean_loss))
                    #mean_loss_ave.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    #break
                # print(loss-mean_loss)
                # print(torch.exp(loss-mean_loss).item())
                # loss=loss*torch.exp(loss-mean_loss)
                # loss.backward()
                # optimizer.step()
                #loss=loss/torch.pow(100,(loss-mean_loss)/loss)
                #break
                # if epoch==trained:
                #     mean_loss_ave.append(loss.item())
                #     break
                # if i==0:
                #     mean_loss=loss.item()
                #or ((loss-mean_loss)/mean_loss<0.2)
                if lin<=mean_loss or iterative_count>8 :
                    mean_loss_ave.append(lin.item())
                    # mean_loss=np.mean(mean_loss_ave)
                    break
                else:
                    iterative_count+=1
                    print("repeat data [%d/%d/%d/%d] Loss: %.4f lin: %.4f " % (i,train_len, epoch, args.n_epoch,loss.item(),lin.item()))
            #print(torch.mean(depth).item())
            if args.visdom:
                with torch.no_grad():

                    vis.line(
                        X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*train_len,
                        Y=loss.item()*torch.ones(1).cpu(),
                        win=loss_window,
                        update='append')
                    vis.line(
                        X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*train_len,
                        Y=lin.item()*torch.ones(1).cpu(),
                        win=lin_window,
                        update='append')
                    #labels=F.interpolate(labels,scale_factor=1/2,mode='bilinear',align_corners=False).squeeze()
                    ground=labels.data.cpu().numpy().astype('float32')
                    ground = ground[0, :, :]
                    ground = (np.reshape(ground, [228, 304]).astype('float32'))/(np.max(ground)+0.001)
                    vis.image(
                        ground,
                        opts=dict(title='ground!', caption='ground.'),
                        win=ground_window,
                    )


                    depth = accurate.data.cpu().numpy().astype('float32')
                    depth = depth[0, :, :]
                    #depth=np.where(depth>np.max(ground),np.max(ground),depth)
                    depth =np.where(ground>0, np.abs((np.reshape(depth, [228, 304]).astype('float32'))/(np.max(depth)+0.001)-ground),0)
                    depth=depth/(np.max(depth)+0.001)
                    vis.image(
                        depth,
                        opts=dict(title='depth!', caption='depth.'),
                        win=memory_retrain_window,
                    )
                    accurate = accurate.data.cpu().numpy().astype('float32')
                    accurate = accurate[0,...]
                    accurate = (np.reshape(accurate, [228, 304]).astype('float32'))/(np.max(accurate)+0.001)
                    vis.image(
                        accurate,
                        opts=dict(title='accurate!', caption='accurate.'),
                        win=accurate_window,
                    )  
                    image=image.data.cpu().numpy().astype('float32')
                    image = image[0,...]
                    #image=image[0,...]
                    #print(image.shape,np.min(image))
                    image = np.reshape(image, [3,228, 304]).astype('float32')
                    vis.image(
                        image,
                        opts=dict(title='image!', caption='image.'),
                        win=image_window,
                    ) 
            loss_rec.append([i+epoch*train_len,torch.Tensor([loss.item()]).unsqueeze(0).cpu()])
            loss_error+=loss.item()

            loss_error_d+=log_d.item()
            print("data [%d/%d/%d/%d] Loss: %.4f lin: %.4f lin_d:%.4f loss_d:%.4f loss_a:%.4f loss_var:%.4f loss_dis:%.4f loss_reg: %.4f" % (i,train_len, epoch, args.n_epoch,loss.item(),lin.item(),lin_d.item(), loss_d.item(),loss_a.item(), \
                torch.sum(0.3*loss_v).item(),torch.sum(0.3*(loss_a-loss_d)).item(),0.001*torch.sum(loss_reg).item()))

            
            
            if (i+1)%(1000)==0:
                mean_loss=np.mean(mean_loss_ave)
                mean_loss_ave=[]
                print("mean_loss:%.4f"%(mean_loss))
                print('testing!')
                model.eval()
                loss_ave=[]
                loss_d_ave=[]
                loss_lin_ave=[]
                loss_r_ave=[]
                loss_log_ave=[]
                for i_val, (images_val, labels_val,regions,segments,images) in tqdm(enumerate(valloader)):
                    #print(r'\n')
                    images_val = Variable(images_val.cuda(1), requires_grad=False)
                    labels_val = Variable(labels_val.cuda(1), requires_grad=False)
                    segments_val = Variable(segments.cuda(1), requires_grad=False)
                    regions_val = Variable(regions.cuda(1), requires_grad=False)

                    with torch.no_grad():
                        #depth,loss_var,loss_dis,loss_reg = model(images_val,segments_val,1,'test')
                        depth,accurate,loss_var,loss_dis,loss_reg = model(images_val,regions_val,labels_val,0,'eval')
                        # loss_d=berhu(depth,labels_val)
                        # loss=torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg)
                        # loss=loss+loss_d
                        accurate=torch.where(accurate>beta,beta*one,accurate)
                        accurate=torch.where(accurate<alpha,alpha*one,accurate)
                        labels_val=torch.where(labels_val>beta,beta*one,labels_val)
                        labels_val=torch.where(labels_val<alpha,alpha*one,labels_val)
                        depth=torch.where(depth>beta,beta*one,depth)
                        depth=torch.where(depth<alpha,alpha*one,depth)
                        depth=F.interpolate(depth,scale_factor=scale,mode='nearest').squeeze()
                        accurate=F.interpolate(accurate,scale_factor=scale,mode='nearest').squeeze()
                        labels_val=(labels_val[...,6*scale:-6*scale,8*scale:-8*scale]).view_as(depth)
                        
                        #accurate=torch.where(accurate>torch.mean(accurate)*4,torch.mean(accurate),accurate)
                        mask=(labels_val>alpha)&(labels_val<beta)
                        mask=mask.float().detach()
                        #lin=torch.sqrt(torch.mean(torch.pow(accurate[mask]-labels_val[mask],2)))
                        #lin=torch.sum(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                        #lin=torch.sum(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                        lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                        lin_d=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(depth-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))

                        #log_d=torch.sqrt(torch.mean(torch.pow(torch.log10(accurate[mask])-torch.log10(labels_val[mask]),2)))
                        #print(torch.min(depth),torch.max(depth),torch.mean(depth))
                        log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels_val)),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1))

                        #print(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                        #log_d=torch.sum(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels_val)),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1))
                        #print(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(torch.log10(accurate)-torch.log10(labels_val),2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                        # if (lin<0.5) & (log_d>0.1):
                        #     np.save('/home/lidong/Documents/RSCFN/analysis.npy',[labels_val.data.cpu().numpy().astype('float32'),accurate.data.cpu().numpy().astype('float32')])
                        #     exit()
                        #accurate=torch.where(accurate>torch.mean(accurate)*4,torch.mean(accurate)*4,accurate)
                        #depth=torch.where(depth>torch.mean(depth)*4,torch.mean(accurate)*4,depth)
                        # if accurate.shape[0]==4:
                        #     a=torch.sqrt(torch.mean(torch.pow(accurate[0,...]-labels_val[0,...],2)))
                        #     b=torch.sqrt(torch.mean(torch.pow(accurate[1,...]-labels_val[1,...],2)))
                        #     c=torch.sqrt(torch.mean(torch.pow(accurate[2,...]-labels_val[2,...],2)))
                        #     d=torch.sqrt(torch.mean(torch.pow(accurate[3,...]-labels_val[3,...],2)))
                        #     lin=(a+b+c+d)/4
                        # else:
                        #     a=torch.sqrt(torch.mean(torch.pow(accurate[0,...]-labels_val[0,...],2)))
                        #     b=torch.sqrt(torch.mean(torch.pow(accurate[1,...]-labels_val[1,...],2)))
                        #     lin=(a+b)/2
                        loss_ave.append(lin.data.cpu().numpy())
                        loss_d_ave.append(lin_d.data.cpu().numpy())
                        loss_log_ave.append(log_d.data.cpu().numpy())
                        #print('error:')
                        #print(loss_ave[-1])
                        #print(torch.max(torch.abs(accurate[mask]-labels_val[mask])).item(),torch.min(torch.abs(accurate[mask]-labels_val[mask])).item())
                        print("error=%.4f,error_d=%.4f,error_log=%.4f"%(lin.item(),lin_d.item(),log_d.item()))
                        # print("loss_d=%.4f loss_var=%.4f loss_dis=%.4f loss_reg=%.4f"%(torch.sum(lin).item()/4,torch.sum(loss_var).item()/4, \
                        #             torch.sum(loss_dis).item()/4,0.001*torch.sum(loss_reg).item()/4))
                    if args.visdom:
                        vis.line(
                            X=torch.ones(1).cpu() * i_val+torch.ones(1).cpu() *test*654/args.batch_size,
                            Y=lin.item()*torch.ones(1).cpu(),
                            win=error_window,
                            update='append')
                        labels_val=labels_val.unsqueeze(1)
                        labels_val=F.interpolate(labels_val,scale_factor=1/2,mode='nearest').squeeze()
                        accurate=accurate.unsqueeze(1)
                        accurate=F.interpolate(accurate,scale_factor=1/2,mode='nearest').squeeze()
                        depth=depth.unsqueeze(1)
                        depth=F.interpolate(depth,scale_factor=1/2,mode='nearest').squeeze()
                        ground=labels_val.data.cpu().numpy().astype('float32')
                        ground = ground[0, :, :]
                        ground = (np.reshape(ground, [228, 304]).astype('float32'))/(np.max(ground)+0.001)
                        vis.image(
                            ground,
                            opts=dict(title='ground!', caption='ground.'),
                            win=ground_window,
                        )
                    

                        depth = accurate.data.cpu().numpy().astype('float32')
                        depth = depth[0, :, :]
                        #depth=np.where(depth>np.max(ground),np.max(ground),depth)
                        depth =np.where(ground>0, np.abs((np.reshape(depth, [228, 304]).astype('float32'))/(np.max(depth)+0.001)-ground),0)
                        depth=depth/(np.max(depth)+0.001)
                        vis.image(
                            depth,
                            opts=dict(title='depth!', caption='depth.'),
                            win=memory_retrain_window,
                        )

                        accurate = accurate.data.cpu().numpy().astype('float32')
                        accurate = accurate[0,...]
                        accurate = (np.reshape(accurate, [228, 304]).astype('float32'))

                        accurate=accurate/(np.max(accurate)+0.001)
                        vis.image(
                            accurate,
                            opts=dict(title='accurate!', caption='accurate.'),
                            win=accurate_window,
                        ) 
                        image=images.data.cpu().numpy().astype('float32')
                        image = image[0,...]
                        #image=image[0,...]
                        #print(image.shape,np.min(image))
                        image = np.reshape(image, [3,228, 304]).astype('float32')
                        vis.image(
                            image,
                            opts=dict(title='image!', caption='image.'),
                            win=image_window,
                        )
                model.train()
                #error=np.mean(loss_ave)
                error=np.mean(loss_ave)
                #error_d=np.mean(loss_d_ave)
                #error_lin=np.mean(loss_lin_ave)
                #error_rate=np.mean(error_rate)
                print("error_r=%.4f,error_d=%.4f,log_error=%.4f"%(error,np.mean(loss_d_ave),np.mean(loss_log_ave)))
                test+=1

                if error<= best_error:
                    best_error = error
                    state = {'epoch': epoch+1,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict(),
                             'error': error,
                             'mean_loss':mean_loss,
                             }
                    torch.save(state, "/home/lidong/Documents/RSCFN/memory/memory_retrain_{}_{}_{}_{}_coarse_best_model.pkl".format(
                        args.arch, args.dataset,str(epoch),str(error)))
                    print('save success')
                np.save('/home/lidong/Documents/RSCFN/loss.npy',loss_rec)
            

        mean_loss=np.mean(mean_loss_ave)
        mean_loss_ave=[]
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
                images_val = Variable(images_val.cuda(1), requires_grad=False)
                labels_val = Variable(labels_val.cuda(1), requires_grad=False)
                segments_val = Variable(segments.cuda(1), requires_grad=False)
                regions_val = Variable(regions.cuda(1), requires_grad=False)

                with torch.no_grad():
                    #depth,loss_var,loss_dis,loss_reg = model(images_val,segments_val,1,'test')
                    depth,accurate,loss_var,loss_dis,loss_reg = model(images_val,regions_val,labels_val,0,'eval')
                    # loss_d=berhu(depth,labels_val)
                    # loss=torch.sum(loss_var)+torch.sum(loss_dis)+0.001*torch.sum(loss_reg)
                    # loss=loss+loss_d
                    accurate=torch.where(accurate>beta,beta*one,accurate)
                    accurate=torch.where(accurate<alpha,alpha*one,accurate)
                    labels_val=torch.where(labels_val>beta,beta*one,labels_val)
                    labels_val=torch.where(labels_val<alpha,alpha*one,labels_val)
                    depth=torch.where(depth>beta,beta*one,depth)
                    depth=torch.where(depth<alpha,alpha*one,depth)
                    depth=F.interpolate(depth,scale_factor=scale,mode='nearest').squeeze()
                    accurate=F.interpolate(accurate,scale_factor=scale,mode='nearest').squeeze()
                    labels_val=(labels_val[...,6*scale:-6*scale,8*scale:-8*scale]).view_as(depth)
                    
                    #accurate=torch.where(accurate>torch.mean(accurate)*4,torch.mean(accurate),accurate)
                    mask=(labels_val>alpha)&(labels_val<beta)
                    mask=mask.float().detach()
                    #lin=torch.sqrt(torch.mean(torch.pow(accurate[mask]-labels_val[mask],2)))
                    #lin=torch.sum(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    #lin=torch.sum(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    lin=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    lin_d=torch.mean(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(depth-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    error_sum+=torch.sum(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    #log_d=torch.sqrt(torch.mean(torch.pow(torch.log10(accurate[mask])-torch.log10(labels_val[mask]),2)))
                    #print(torch.min(depth),torch.max(depth),torch.mean(depth))
                    log_d=torch.mean(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels_val)),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1))

                    #print(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(accurate-labels_val,2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    #log_d=torch.sum(torch.sum(torch.where(mask>0,torch.abs(torch.log10(accurate)-torch.log10(labels_val)),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1))
                    #print(torch.sqrt(torch.sum(torch.where(mask>0,torch.pow(torch.log10(accurate)-torch.log10(labels_val),2),mask).view(labels_val.shape[0],-1),dim=-1)/torch.sum(mask.view(labels_val.shape[0],-1),dim=-1)))
                    # if (lin<0.5) & (log_d>0.1):
                    #     np.save('/home/lidong/Documents/RSCFN/analysis.npy',[labels_val.data.cpu().numpy().astype('float32'),accurate.data.cpu().numpy().astype('float32')])
                    #     exit()
                    #accurate=torch.where(accurate>torch.mean(accurate)*4,torch.mean(accurate)*4,accurate)
                    #depth=torch.where(depth>torch.mean(depth)*4,torch.mean(accurate)*4,depth)
                    # if accurate.shape[0]==4:
                    #     a=torch.sqrt(torch.mean(torch.pow(accurate[0,...]-labels_val[0,...],2)))
                    #     b=torch.sqrt(torch.mean(torch.pow(accurate[1,...]-labels_val[1,...],2)))
                    #     c=torch.sqrt(torch.mean(torch.pow(accurate[2,...]-labels_val[2,...],2)))
                    #     d=torch.sqrt(torch.mean(torch.pow(accurate[3,...]-labels_val[3,...],2)))
                    #     lin=(a+b+c+d)/4
                    # else:
                    #     a=torch.sqrt(torch.mean(torch.pow(accurate[0,...]-labels_val[0,...],2)))
                    #     b=torch.sqrt(torch.mean(torch.pow(accurate[1,...]-labels_val[1,...],2)))
                    #     lin=(a+b)/2
                    loss_ave.append(lin.data.cpu().numpy())
                    loss_d_ave.append(lin_d.data.cpu().numpy())
                    loss_log_ave.append(log_d.data.cpu().numpy())
                    #print('error:')
                    #print(loss_ave[-1])
                    #print(torch.max(torch.abs(accurate[mask]-labels_val[mask])).item(),torch.min(torch.abs(accurate[mask]-labels_val[mask])).item())
                    print("error=%.4f,error_d=%.4f,error_log=%.4f"%(lin.item(),lin_d.item(),log_d.item()))
                    # print("loss_d=%.4f loss_var=%.4f loss_dis=%.4f loss_reg=%.4f"%(torch.sum(lin).item()/4,torch.sum(loss_var).item()/4, \
                    #             torch.sum(loss_dis).item()/4,0.001*torch.sum(loss_reg).item()/4))
                if args.visdom:
                    vis.line(
                        X=torch.ones(1).cpu() * i_val+torch.ones(1).cpu() *test*654/args.batch_size,
                        Y=lin.item()*torch.ones(1).cpu(),
                        win=error_window,
                        update='append')
                    labels_val=labels_val.unsqueeze(1)
                    labels_val=F.interpolate(labels_val,scale_factor=1/2,mode='nearest').squeeze()
                    accurate=accurate.unsqueeze(1)
                    accurate=F.interpolate(accurate,scale_factor=1/2,mode='nearest').squeeze()
                    depth=depth.unsqueeze(1)
                    depth=F.interpolate(depth,scale_factor=1/2,mode='nearest').squeeze()
                    ground=labels_val.data.cpu().numpy().astype('float32')
                    ground = ground[0, :, :]
                    ground = (np.reshape(ground, [228, 304]).astype('float32'))/(np.max(ground)+0.001)
                    vis.image(
                        ground,
                        opts=dict(title='ground!', caption='ground.'),
                        win=ground_window,
                    )
                

                    depth = accurate.data.cpu().numpy().astype('float32')
                    depth = depth[0, :, :]
                    #depth=np.where(depth>np.max(ground),np.max(ground),depth)
                    depth =np.where(ground>0, np.abs((np.reshape(depth, [228, 304]).astype('float32'))/(np.max(depth)+0.001)-ground),0)
                    depth=depth/(np.max(depth)+0.001)
                    vis.image(
                        depth,
                        opts=dict(title='depth!', caption='depth.'),
                        win=memory_retrain_window,
                    )

                    accurate = accurate.data.cpu().numpy().astype('float32')
                    accurate = accurate[0,...]
                    accurate = (np.reshape(accurate, [228, 304]).astype('float32'))

                    accurate=accurate/(np.max(accurate)+0.001)
                    vis.image(
                        accurate,
                        opts=dict(title='accurate!', caption='accurate.'),
                        win=accurate_window,
                    ) 
                    image=images.data.cpu().numpy().astype('float32')
                    image = image[0,...]
                    #image=image[0,...]
                    #print(image.shape,np.min(image))
                    image = np.reshape(image, [3,228, 304]).astype('float32')
                    vis.image(
                        image,
                        opts=dict(title='image!', caption='image.'),
                        win=image_window,
                    ) 
            #error=np.mean(loss_ave)
            error=np.mean(loss_ave)
            #error_d=np.mean(loss_d_ave)
            #error_lin=np.mean(loss_lin_ave)
            #error_rate=np.mean(error_rate)
            print("error_r=%.4f,error_d=%.4f,error_log=%.4f"%(error,np.mean(loss_d_ave),np.mean(loss_log_ave)))
            test+=1
            print(error_sum/654)
            if error<= best_error:
                best_error = error
                state = {'epoch': epoch+1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error,
                         'mean_loss':mean_loss,
                         }
                torch.save(state, "/home/lidong/Documents/RSCFN/memory/memory_retrain_{}_{}_{}_{}_coarse_best_model.pkl".format(
                    args.arch, args.dataset,str(epoch),str(error)))
                print('save success')
            np.save('/home/lidong/Documents/RSCFN/loss.npy',loss_rec)
            #exit()

        if epoch%30==0:
            #best_error = error
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), 
                     'error': error,
                     'mean_loss':mean_loss,}
            torch.save(state, "/home/lidong/Documents/RSCFN/memory/memory_retrain_{}_{}_{}_coarse_model.pkl".format(
                args.arch, args.dataset,str(epoch)))
            print('save success')



if __name__ == '__main__':
    scale=2
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='rsn_cluster',
                        help='Architecture to use [\'region support network\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='nyu',
                        help='Dataset to use [\'sceneflow and kitti etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=228,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=304,
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=4000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/RSCFN/memory/memory_retrain_rsn_cluster_nyu_27_1.0387137_coarse_best_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSCFN/memory_retrain_rsn_cluster_nyu_27_1.0387137_coarse_best_model.pkl.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=False,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
