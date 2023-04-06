#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018
@author: maximov
"""

import torch
import torch.optim as optim
import torch.utils.data
import argparse
from dataloaders import nyuloader
from model import defnet

parser = argparse.ArgumentParser(description='camIndDefocus')
parser.add_argument('--bs', type=int,default=12, help='training batch size')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--fscale', default=1.9,help='divide all focal distances by this value')
parser.add_argument('--blurclip', default=25,help='Clip blur by this value : only applicable for camind model. Default=10')
parser.add_argument('--blurweight', default=1.0,help='weight for blur loss')
parser.add_argument('--checkpt', default='C:\\Users\\lahir\\models\\handblur\\best_ep0.pth', help='path to the saved model')
parser.add_argument('--s2limits', nargs='+', default=[0.1,3.],  help='the interval of depth where the errors are calculated')
parser.add_argument('--dataset', default='defocusnet', help='blender data path')
parser.add_argument('--camind', type=bool,default=False, help='True: use camera independent model. False: use defocusnet model')
parser.add_argument('--seg', type=bool,default=True, help='Use hand segmentation to train and eval')
parser.add_argument('--epochs', type=int,default=1000, help='training batch size')
parser.add_argument('--savemodel', default='C:\\Users\\lahir\\models\\handblur\\', help='path to the saved model')



f=25e-3
N=1.9
px=6*1e-6*6
s1=0.7
kcam=f**2/(N*px)

args = parser.parse_args()

'''
load data
'''
datapath='D:\\handsdata\\nyu_out\\train\\'
loaders, total_steps = nyuloader.load_data(datapath,blur=1,train_split=0.8,WORKERS_NUM=0,
    BATCH_SIZE=args.bs,MAX_DPT=args.depthscale,blurclip=args.blurclip)
'''
load model
'''
ch_inp_num = 3
ch_out_num = 1
model = defnet.AENet(ch_inp_num, 1, 16, flag_step2=True)
model = model.to('cuda')
model_params = model.parameters()
# model_params += list(model.parameters())

# loading weights of the first step
if args.checkpt:
    print('loading model....')
    print('model path :'+args.checkpt)
    pretrained_dict = torch.load(args.checkpt)
    model_dict = model.state_dict()
    for param_tensor in model_dict:
        for param_pre in pretrained_dict:
            if param_tensor == param_pre:
                model_dict.update({param_tensor: pretrained_dict[param_pre]})
    model.load_state_dict(model_dict)

# ============ init ===============
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

def eval():
    model.eval()
    total_d_loss=0
    iter=0
    mean_depth=0
    for st_iter, sample_batch in enumerate(loaders[1]):
        # Setting up input and output data
        X = sample_batch['input'][:,0,:,:,:].float().to('cuda')
        Y = sample_batch['output'].float().to('cuda')
        seg=sample_batch['seg'].to('cuda')
        

        # import matplotlib.pyplot as plt
        # seg=sample_batch['seg']
        # seg=seg.detach().cpu().squeeze().numpy()
        # plt.imshow(seg)
        # plt.show()

        # img=X.cpu().squeeze().numpy()
        # plt.imshow(img[0,:,:])
        # plt.show()

        
        #blur (|s2-s1|/(s2*(s1-f)))
        gt_step1 = Y[:, :-1, :, :]
        #depth in m
        gt_step2 = Y[:, -1:, :, :]
        
        # we only use focal stacks with a single image
        stacknum = 1
        if(args.seg):
            mask=(seg>0).int()
        else:
            mask=torch.ones_like(seg)
        if(torch.sum(mask).item()==0):
            continue

        X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
        for t in range(stacknum):
            #iterate through the batch
            for i in range(X.shape[0]):
                X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]/kcam*(0.7-25e-3)*(100)
                s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(0.7)
        X2_fcs = X2_fcs.float().to('cuda')
        s1_fcs = s1_fcs.float().to('cuda')
        # Forward and compute loss
        output_step1,output_step2,_ = model(X,camparam=X2_fcs,foc_dist=s1_fcs)
        depth_loss=torch.sum(torch.square(output_step2*args.depthscale*mask-gt_step2*args.depthscale*mask))/torch.sum(mask)
        # mean_depth+=(torch.sum(gt_step2*args.depthscale*mask)/torch.sum(mask)).item()
        total_d_loss+=depth_loss.item()
        iter+=1
    # print('mean depth='+str(mean_depth/iter))
    return total_d_loss/iter



def train_model():
    model.train()
    criterion = torch.nn.MSELoss()
    #criterion=F.smooth_l1_loss(reduction='none')
    optimizer = optim.Adam(model_params, lr=0.0001)
    min_depth_loss_eval=1000
    ##### Training
    print("Total number of epochs:", args.epochs)
    for e_iter in range(args.epochs):
        epoch_iter = e_iter
        loss_sum, iter_count,absloss_sum= 0,0,0
        depthloss_sum,blurloss_sum=0,0
        blur_sum=0
        mean_blur=0
        for st_iter, sample_batch in enumerate(loaders[0]):
            # Setting up input and output data
            X = sample_batch['input'][:,0,:,:,:].float().to('cuda')
            Y = sample_batch['output'].float().to('cuda')
            seg=sample_batch['seg'].to('cuda')

            if(args.seg):
                mask=(seg>0).int()
            else:
                mask=torch.ones_like(seg)
            if(torch.sum(mask).item()==0):
                continue

            optimizer.zero_grad()
            
            #blur (|s2-s1|/(s2*(s1-f)))
            gt_step1 = Y[:, :-1, :, :]
            #depth in m
            gt_step2 = Y[:, -1:, :, :]
            
            # we only use focal stacks with a single image
            stacknum = 1

            X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
            for t in range(stacknum):
                #iterate through the batch
                for i in range(X.shape[0]):
                    X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]/kcam*(0.7-25e-3)*(100)
                    s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(0.7)
            X2_fcs = X2_fcs.float().to('cuda')
            s1_fcs = s1_fcs.float().to('cuda')
            # Forward and compute loss
            output_step1,output_step2,_ = model(X,camparam=X2_fcs,foc_dist=s1_fcs)
            blur_sum+=torch.sum(output_step1*mask).item()/torch.sum(mask)
            #blurpred=output_step1*(0.1-2.9e-3)*1.4398*7
            depth_loss=criterion(output_step2*mask, gt_step2*mask)
            blur_loss=criterion(output_step1*mask, gt_step1*mask)
            loss=depth_loss+blur_loss*args.blurweight

            absloss=torch.sum(torch.abs(output_step1-gt_step1)*mask)/torch.sum(mask)
            absloss_sum+=absloss.item()

            loss.backward()
            optimizer.step()

            # Training log
            loss_sum += loss.item()
            iter_count += 1.
            blurloss_sum+=blur_loss.item()
            depthloss_sum+=depth_loss.item()

            if (st_iter + 1) % 10 == 0:
                print('handblur ', 'Epoch [{}/{}], Step [{}/{}], blur loss: {:.4f},  depth loss: {:.4f}'
                      .format(epoch_iter + 1, args.epochs, st_iter + 1, total_steps, blurloss_sum / iter_count, depthloss_sum / iter_count))
    
                absloss_sum=0
                depth_sum,blur_sum=0,0
                depthloss_sum,blurloss_sum=0,0

                total_iter = total_steps * epoch_iter + st_iter
                loss_sum, iter_count = 0,0
        # Evaluate
        depth_loss_eval=eval()
        print('eval depth loss = %.5f'%depth_loss_eval)
        model.train()
        #save model if better than the previous best model
        if(depth_loss_eval<min_depth_loss_eval):
            print('saving model')
            torch.save(model.state_dict(),args.savemodel+'best_seg_ep' + str(0) + '.pth')
            min_depth_loss_eval=depth_loss_eval

        # # Save model
        # if (epoch_iter+1) % 10 == 0:
        #     print('saving model')
        #     torch.save(model.state_dict(), model_info['model_dir'] + model_info['model_name'] + '_ep' + str(0) + '.pth')
        #     s2loss1,s2loss2,blurloss,meanblur,gtmeanblur,minblur,maxblur=util_func.eval(loaders[1],model_info,dataset=args.dataset,camind=args.camind,
        #     depthscale=args.depthscale,fscale=args.fscale,s2limits=args.s2limits,aif=args.aif)
        #     print('s2 loss2: '+str(s2loss2))
        #     print('blur loss = '+str(blurloss))
        #     print('mean blur = '+str(meanblur))

def main():
    # Run training
    train_model()

if __name__ == "__main__":
    main()

#datapath='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
#focalblender.get_data_stats(datapath,50)
'''
fdist of DDFF 
tensor([[0.2800, 0.2511, 0.2222, 0.1933, 0.1644, 0.1356, 0.1067, 0.0778, 0.0489,
         0.0200]])
'''

'''
#plotting distribution of blur
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

f=3e-3
p=3.1e-3/256
s=1
N=1
for f in [3e-3,4e-3,5e-3]:
    s1 = np.random.uniform(0.1,1.5,1000)
    s2 = np.random.uniform(0.0,2.0,1000)
    s2=np.random.normal(loc=1.0,scale=0.1,size=1000)
    blur=np.abs(s1-s2)/s2*1/(s1-f) * f**2/N *1/p * s
    density = stats.gaussian_kde(blur)
    bins = np.linspace(0.1, 2.0, 1000)
    n,bins = np.histogram(np.array(blur), bins)
    plt.plot(bins, density(bins),label='f=%1.0fmm'%(f*1000))

ax = plt.gca()
# Hide X and Y axes label marks
ax.yaxis.set_tick_params(labelleft=False)
# Hide Y axes tick marks
ax.set_yticks([])
plt.legend()
plt.xlabel('Blur in pixles')
plt.ylabel('Density')
plt.savefig('blur_distF.png', dpi=500)
plt.show()
'''




