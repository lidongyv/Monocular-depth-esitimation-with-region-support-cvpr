# -*- coding: utf-8 -*-
# @Author: lidong
# @Date:   2018-03-18 13:41:34
# @Last Modified by:   yulidong
# @Last Modified time: 2018-05-11 20:44:24
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
                           split='train_region', img_size=(args.img_rows, args.img_cols),task='region')
    v_loader = data_loader(data_path, is_transform=True,
                           split='test_region', img_size=(args.img_rows, args.img_cols),task='region')

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
        old_window = vis.line(X=torch.zeros((1,)).cpu(),
                               Y=torch.zeros((1)).cpu(),
                               opts=dict(xlabel='minibatches',
                                         ylabel='Loss',
                                         title='Trained Loss',
                                         legend=['Loss']))
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
        # optimizer = torch.optim.Adam(
        #     model.parameters(), lr=args.l_rate,weight_decay=5e-4,betas=(0.9,0.999))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.l_rate,momentum=0.99, weight_decay=5e-4)
    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = region_log
    trained=0
    scale=100

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #model_dict=model.state_dict()  
            #opt=torch.load('/home/lidong/Documents/RSDEN/RSDEN/exp1/l2/sgd/log/83/rsnet_nyu_best_model.pkl')
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            #opt=None
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            trained=checkpoint['epoch']
            best_error=checkpoint['error']
            #best_error=5
            #print('load success!')
            loss_rec=np.load('/home/lidong/Documents/RSDEN/RSDEN/loss.npy')
            loss_rec=list(loss_rec)
            loss_rec=loss_rec[:816*trained]
            # for i in range(300):
            #     loss_rec[i][1]=loss_rec[i+300][1]
            for l in range(int(len(loss_rec)/816)):
                if args.visdom:
                    #print(loss_rec[l])
                    vis.line(
                        X=torch.ones(1).cpu() * loss_rec[l*816][0],
                        Y=np.mean(np.array(loss_rec[l*816:(l+1)*816])[:,1])*torch.ones(1).cpu(),
                        win=old_window,
                        update='append')
            
    else:

        print("No checkpoint found at '{}'".format(args.resume))
        print('Initialize from resnet34!')
        resnet34=torch.load('/home/lidong/Documents/RSDEN/RSDEN/resnet34-333f7ec4.pth')
        model_dict=model.state_dict()            
        pre_dict={k: v for k, v in resnet34.items() if k in model_dict}
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
        print('load success!')
        best_error=1
        trained=0


    #best_error=5
    # it should be range(checkpoint[''epoch],args.n_epoch)
    for epoch in range(trained, args.n_epoch):
    #for epoch in range(0, args.n_epoch):
        
        #trained
        print('training!')
        model.train()
        for i, (images, labels,segments) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            segments = Variable(segments.cuda())
            optimizer.zero_grad()
            outputs = model(images)
            #outputs=outputs
            loss = loss_fn(input=outputs, target=labels,instance=segments)
            # print('training:'+str(i)+':learning_rate'+str(loss.data.cpu().numpy()))
            loss.backward()
            optimizer.step()
            # print(torch.Tensor([loss.data[0]]).unsqueeze(0).cpu())
            #print(loss.item()*torch.ones(1).cpu())
            #nyu2_train:246,nyu2_all:816
            if args.visdom:
                vis.line(
                    X=torch.ones(1).cpu() * i+torch.ones(1).cpu() *(epoch-trained)*816,
                    Y=loss.item()*torch.ones(1).cpu(),
                    win=loss_window,
                    update='append')
                pre = outputs.data.cpu().numpy().astype('float32')
                pre = pre[0, :, :, :]
                #pre = np.argmax(pre, 0)
                pre = (np.reshape(pre, [480, 640]).astype('float32')-np.min(pre))/(np.max(pre)-np.min(pre))
                #pre = pre/np.max(pre)
                # print(type(pre[0,0]))
                vis.image(
                    pre,
                    opts=dict(title='predict!', caption='predict.'),
                    win=pre_window,
                )
                ground=labels.data.cpu().numpy().astype('float32')
                #print(ground.shape)
                ground = ground[0, :, :]
                ground = (np.reshape(ground, [480, 640]).astype('float32')-np.min(ground))/(np.max(ground)-np.min(ground))
                vis.image(
                    ground,
                    opts=dict(title='ground!', caption='ground.'),
                    win=ground_window,
                )
            
            loss_rec.append([i+epoch*816,torch.Tensor([loss.item()]).unsqueeze(0).cpu()])
            print("data [%d/816/%d/%d] Loss: %.4f" % (i, epoch, args.n_epoch,loss.item()))
        
        if epoch>50:
            check=3
        else:
            check=5
        if epoch>70:
            check=2
        if epoch>85:
            check=1                 
        if epoch%check==0:  
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
            variance=[]
            for i_val, (images_val, labels_val,segments) in tqdm(enumerate(valloader)):
                print(r'\n')
                images_val = Variable(images_val.cuda(), requires_grad=False)
                labels_val = Variable(labels_val.cuda(), requires_grad=False)
                segments = Variable(segments.cuda(), requires_grad=False)
                with torch.no_grad():
                    outputs = model(images_val)
                    pred = outputs.data.cpu().numpy()
                    gt = labels_val.data.cpu().numpy()
                    instance = segments.data.cpu().numpy()
                    ones=np.ones((gt.shape))
                    zeros=np.zeros((gt.shape))
                    pred=np.reshape(pred,(gt.shape))
                    instance=np.reshape(instance,(gt.shape))
                    #gt=np.reshape(gt,[4,480,640])
                    # dis=np.square(gt-pred)
                    # error_lin.append(np.sqrt(np.mean(dis)))
                    # dis=np.square(np.log(gt)-np.log(pred))
                    # error_log.append(np.sqrt(np.mean(dis)))
                    var=0
                    linear=0
                    log_dis=0
                    for i in range(1,int(np.max(instance)+1)):
                        pre_region=np.where(instance==i,pred,0)
                        dis=np.where(instance==i,np.abs(gt-pred),0)
                        num=np.sum(np.where(instance==i,1,0))
                        m=np.sum(pre_region)/num
                        pre_region=np.where(instance==i,pred-m,0)
                        pre_region=np.sum(np.square(pre_region))/num
                        log_region=np.where(instance==i,np.abs(np.log(gt+1e-6)-np.log(pred+1e-6)),0)
                        var+=pre_region
                        linear+=np.sum(dis)/num
                        log_dis+=np.sum(log_region)/num
                    error_log.append(log_dis/np.max(instance))
                    error_lin.append(linear/np.max(instance))
                    variance.append(var/np.max(instance))    
                    print("error_lin=%.4f,error_log=%.4f,variance=%.4f"%(
                        error_lin[i_val],
                        error_log[i_val],
                        variance[i_val]))                   
                    # alpha=np.mean(np.log(gt)-np.log(pred))
                    # dis=np.square(np.log(pred)-np.log(gt)+alpha)
                    # error_va.append(np.mean(dis)/2)
                    # dis=np.mean(np.abs(gt-pred))/gt
                    # error_absrd.append(np.mean(dis))
                    # dis=np.square(gt-pred)/gt
                    # error_squrd.append(np.mean(dis))
                    # thelt=np.where(pred/gt>gt/pred,pred/gt,gt/pred)
                    # thres1=1.25

                    # thre1.append(np.mean(np.where(thelt<thres1,ones,zeros)))
                    # thre2.append(np.mean(np.where(thelt<thres1*thres1,ones,zeros)))
                    # thre3.append(np.mean(np.where(thelt<thres1*thres1*thres1,ones,zeros)))
                    # #a=thre1[i_val]
                    # #error_rate.append(np.mean(np.where(dis<0.6,ones,zeros)))
                    # print("error_lin=%.4f,error_log=%.4f,error_va=%.4f,error_absrd=%.4f,error_squrd=%.4f,thre1=%.4f,thre2=%.4f,thre3=%.4f"%(
                    #     error_lin[i_val],
                    #     error_log[i_val],
                    #     error_va[i_val],
                    #     error_absrd[i_val],
                    #     error_squrd[i_val],
                    #     thre1[i_val],
                    #     thre2[i_val],
                    #     thre3[i_val]))
            error=np.mean(error_lin)
            variance=np.mean(variance)
            #error_rate=np.mean(error_rate)
            print("error=%.4f,variance=%.4f"%(error,variance))

            if error<= best_error:
                best_error = error
                state = {'epoch': epoch+1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(),
                         'error': error,}
                torch.save(state, "{}_{}_best_model.pkl".format(
                    args.arch, args.dataset))
                print('save success')
            np.save('/home/lidong/Documents/RSDEN/RSDEN//loss.npy',loss_rec)
        if epoch%5==0:
            #best_error = error
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict(), 
                     'error': error,}
            torch.save(state, "{}_{}_{}_model.pkl".format(
                args.arch, args.dataset,str(epoch)))
            print('save success')





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
    parser.add_argument('--resume', nargs='?', type=str, default='/home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_95_model.pkl',
                        help='Path to previous saved model to restart from /home/lidong/Documents/RSDEN/RSDEN/rsnet_nyu_30_model.pkl')
    parser.add_argument('--visdom', nargs='?', type=bool, default=True,
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)
