import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, utils

from os import listdir, mkdir
from os.path import isfile, join, isdir
from visdom import Visdom
import numpy as np
import random
import OpenEXR
from PIL import Image
from skimage import img_as_float
import matplotlib.pyplot as plt
import cv2

# img=cv2.imread('D://handsdata//nyu_out//train//depth//depth_1_0000001.png',cv2.IMREAD_UNCHANGED)
# plt.imshow(img)
# plt.show()
# reading depth files
def read_dpt(img_dpt_path): 
    # dpt_img = OpenEXR.InputFile(img_dpt_path)
    dpt_img=cv2.imread(img_dpt_path,cv2.IMREAD_UNCHANGED)
    #converting from mm to m
    dpt_img=dpt_img/1000.
    return dpt_img

# to calculate circle of confusion
def get_blur(s1,s2,f,kcam):
    blur=abs(s2-s1)/s2 * 1/(s1-f)*kcam
    return blur

'''
All in-focus image is attached to the input matrix after the RGB image

input matrix channles :
[batch,image,rgb_channel,256,256]

output: [blur,depth]
'''

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, transform_fnc=None, max_dpt = 3.,blurclip=10.):

        self.root_dir = root_dir
        print("image data root dir : " +str(self.root_dir))
        self.transform_fnc = transform_fnc
        self.blurclip=blurclip

        ##### Load and sort all images
        self.rgbpath=root_dir+"blur\\rgb\\"
        self.depthpath=root_dir+"depth\\"
        self.segpath=root_dir+"seg\\"
        self.imglist_rgb = [f for f in listdir(self.rgbpath) if isfile(join(self.rgbpath, f)) and f[-3:] == "png"]
        self.imglist_dpt = [f for f in listdir(self.depthpath) if isfile(join(self.depthpath, f)) and f[-3:] == "png"]
        self.imglist_seg = [f for f in listdir(self.segpath) if isfile(join(self.segpath, f)) and f[-3:] == "png"]

        print("Total number of rgb files", len(self.imglist_rgb))
        print("Total number of depth files", len(self.imglist_dpt))
        print("Total number of segmentation mask files", len(self.imglist_dpt))

        self.imglist_rgb.sort()
        self.imglist_dpt.sort()
        self.imglist_seg.sort()

        self.max_dpt = max_dpt
        #focal length in m
        self.f=25e-3
        self.N=1.9
        self.px=6*1e-6*6
        self.s1=0.7
        self.kcam=self.f**2/(self.N*self.px)

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        # add RGB, CoC, Depth inputs
        mats_input = np.zeros((256, 256, 3,0))
        mats_output = np.zeros((256, 256, 0))

        ##### Read and process an image
        ind = int(idx)
        img_dpt = read_dpt(self.depthpath + self.imglist_dpt[ind])

        #img_dpt_scaled = np.clip(img_dpt, 0., 1.9)
        #mat_dpt_scaled = img_dpt_scaled / 1.9
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]
        im=cv2.imread(self.rgbpath + self.imglist_rgb[ind],cv2.IMREAD_UNCHANGED)
        # im = Image.open(self.rgbpath + self.imglist_rgb[ind])
        img_all = np.array(im)
        mat_all = img_all.copy() / 255.
        mat_all=np.expand_dims(mat_all,axis=-1)
        print(self.imglist_rgb[ind])
        print('mats_input:'+str(mats_input.shape))
        print('mat_all:'+str(mat_all.shape))
        mats_input = np.concatenate((mats_input, mat_all), axis=3)
        img_msk = get_blur(self.s1,img_dpt,self.f,self.kcam)

        img_msk = img_msk / self.blurclip
        #img_msk = np.clip(img_msk, 0, 1.0e-4) / 1.0e-4
        mat_msk = img_msk.copy()[:, :, np.newaxis]

        #append blur to the output
        # print('blur:'+str(mat_msk.shape))
        mats_output = np.concatenate((mats_output, mat_msk), axis=2)
        #append depth to the output
        # print('depth:'+str(mat_dpt.shape))
        print(self.imglist_dpt[ind])
        mats_output = np.concatenate((mats_output, mat_dpt), axis=2)
        
        sample = {'input': mats_input, 'output': mats_output}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        sample = {'input': sample['input'], 'output': sample['output']}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output = sample['input'], sample['output']

        mats_input = mats_input.transpose((3,2, 0, 1))
        mats_output = mats_output.transpose((2, 0, 1))
        return {'input': torch.from_numpy(mats_input),
                'output': torch.from_numpy(mats_output),}


def load_data(data_dir, blur,train_split,
              WORKERS_NUM, BATCH_SIZE, MAX_DPT,blurclip=10.0):
    img_dataset = ImageDataset(root_dir=data_dir,transform_fnc=transforms.Compose([ToTensor()]),max_dpt=MAX_DPT,
                               blurclip=blurclip)
    
    indices = list(range(len(img_dataset)))
    split = int(len(img_dataset) * train_split)

    indices_train = indices[:split]
    indices_valid = indices[split:]

    dataset_train = torch.utils.data.Subset(img_dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(img_dataset, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=1, batch_size=1, shuffle=False)

    total_steps = int(len(dataset_train) / BATCH_SIZE)
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(dataset_train))
    print("Total number of validataion sample:", len(dataset_valid))

    return [loader_train, loader_valid], total_steps

# print('starting...')
# datapath='D:\\handsdata\\nyu_out\\train\\'
# loaders, total_steps = load_data(datapath,blur=1,train_split=0.8,WORKERS_NUM=0,
#         BATCH_SIZE=1,MAX_DPT=1.0,blurclip=10)
# for st_iter, sample_batch in enumerate(loaders[0]):
#     break

# img=sample_batch['input'].squeeze().numpy()[0,:,:]
# plt.imshow(img)
# plt.show()

# img=sample_batch['output'].squeeze().numpy()[0,:,:]
# plt.imshow(img)
# plt.show()

# import os

# rgbpath='D:\\handsdata\\nyu_out\\train\\blur\\rgb\\'
# depthpath='D:\\handsdata\\nyu_out\\train\\depth\\'
# segpath='D:\\handsdata\\nyu_out\\train\\seg\\'
# imglist_rgb = [f for f in listdir(rgbpath) if isfile(join(rgbpath, f)) and f[-3:] == "png"]
# imglist_dpt = [f for f in listdir(depthpath) if isfile(join(depthpath, f)) and f[-3:] == "png"]
# imglist_seg = [f for f in listdir(segpath) if isfile(join(segpath, f)) and f[-3:] == "png"]
# imglist_rgb.sort()
# imglist_dpt.sort()
# imglist_seg.sort()

# sum=0
# for i,img in enumerate(imglist_rgb):
#     im=cv2.imread(rgbpath+img)
#     s=im.shape
#     if(s!=(256,256,3)):
#         sum+=1
#         print(img)
#         os.remove(rgbpath+img)
#         os.remove(depthpath+imglist_dpt[i])
#         os.remove(segpath+imglist_seg[i])



datapath='D:\\handsdata\\nyu_out\\train\\'
blurclip=1
def get_data_stats(datapath,blurclip):
    loaders, total_steps = load_data(datapath,blur=1,train_split=0.8,WORKERS_NUM=0,
        BATCH_SIZE=1,MAX_DPT=1.0,blurclip=blurclip)
    print('stats of train data')
    get_loader_stats(loaders[0])
    print('______')

#data statistics of the input images
def get_loader_stats(loader):
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    for st_iter, sample_batch in enumerate(loader):
        # Setting up input and output data
        X = sample_batch['input'][:,0,:,:,:].float()
        Y = sample_batch['output'].float()

        xmin_=torch.min(X).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(X).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(X).cpu().item()
        count+=1
    
        #blur (|s2-s1|/(s2*(s1-f)))
        gt_step1 = Y[:, 0, :, :]
        #depth in m
        gt_step2 = Y[:, 1, :, :]
        
        depthmin_=torch.min(gt_step2).cpu().item()
        if(depthmin_<depthmin):
            depthmin=depthmin_
        depthmax_=torch.max(gt_step2).cpu().item()
        if(depthmax_>depthmax):
            depthmax=depthmax_
        depthmean+=torch.mean(gt_step2).cpu().item()

        blurmin_=torch.min(gt_step1).cpu().item()
        if(blurmin_<blurmin):
            blurmin=blurmin_
        blurmax_=torch.max(gt_step1).cpu().item()
        if(blurmax_>blurmax):
            blurmax=blurmax_
        blurmean+=torch.mean(gt_step1).cpu().item()

    print('X min='+str(xmin))
    print('X max='+str(xmax))
    print('X mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))

    print('blur min='+str(blurmin))
    print('blur max='+str(blurmax))
    print('blur mean='+str(blurmean/count))

get_data_stats(datapath,blurclip)

'''
blur_thres=7.0
p=3.1/256*1e-3 # pixel width in m
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,1.5]

get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres)
'''







