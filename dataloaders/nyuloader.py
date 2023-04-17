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

# d=read_dpt('C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\depth\\00000.png')
# import matplotlib.pyplot as plt
# plt.imshow(d)
# plt.show()
# to calculate circle of confusion
def get_blur(s1,s2,f):
    blur=abs(s2-s1)/s2 * 1/(s1-f)
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
        self.rgbpath=root_dir+"blur\\"
        self.depthpath=root_dir+"depth\\"
        self.segpath=root_dir+"seg_resized\\"
        self.imglist_rgb = [f for f in listdir(self.rgbpath) if isfile(join(self.rgbpath, f)) and f[-3:] == "png"]
        self.imglist_dpt = [f for f in listdir(self.depthpath) if isfile(join(self.depthpath, f)) and f[-3:] == "png"]
        self.imglist_seg = [f for f in listdir(self.segpath) if isfile(join(self.segpath, f)) and f[-3:] == "jpg"]

        print("Total number of rgb files", len(self.imglist_rgb))
        print("Total number of depth files", len(self.imglist_dpt))
        print("Total number of segmentation mask files", len(self.imglist_dpt))

        self.imglist_rgb.sort()
        self.imglist_dpt.sort()
        self.imglist_seg.sort()

        self.max_dpt = max_dpt
        #focal length in m
        self.f=60e-3
        self.N=2.0
        self.px=36*1e-6
        self.s1=2.0
        self.kcam=self.f**2/(self.N*self.px)

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        ##### Read depth image
        ind = int(idx)
        # print('dept:'+str(self.imglist_dpt[ind]))
        # print('rgb:'+str(self.imglist_rgb[ind]))
        img_dpt = read_dpt(self.depthpath + self.imglist_dpt[ind])
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]
        mat_dpt=self.s1/mat_dpt

        #read rgb image
        im=cv2.imread(self.rgbpath + self.imglist_rgb[ind],cv2.IMREAD_UNCHANGED)
        im = np.array(im)
        mat_rgb = im.copy() / 255.

        #read hand segmentation image
        seg=cv2.imread(self.segpath + self.imglist_seg[ind],cv2.IMREAD_UNCHANGED)
        seg=np.expand_dims(seg,axis=2)

        #get blur
        img_msk = get_blur(self.s1,img_dpt,self.f)
        img_msk = img_msk / self.blurclip
        mat_blur = img_msk.copy()[:, :, np.newaxis]

        #create a single matrix with rgb,depth,blur and seg
        data=np.concatenate([mat_rgb,mat_dpt,mat_blur,seg],axis=2)
        
        if self.transform_fnc:
            sample = self.transform_fnc(data)
        sample = {'rgb': sample[:3,:,:], 'depth': sample[3,:,:],'blur':sample[4,:,:],'seg':sample[5,:,:]}
        return sample

class Transform(object):
    def __call__(self, image):
        image=torch.from_numpy(image)
        image=torch.permute(image,(2,0,1))
        return image


def load_data(data_dir, blur,train_split,
              WORKERS_NUM, BATCH_SIZE, MAX_DPT,blurclip=10.0):
    tr=transforms.Compose([
        Transform(),
        transforms.RandomCrop((256,256),pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    
    img_dataset = ImageDataset(root_dir=data_dir,transform_fnc=tr,max_dpt=MAX_DPT,
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

# datapath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\'
# loaders, total_steps = load_data(datapath,blur=1,train_split=0.8,WORKERS_NUM=0,
#         BATCH_SIZE=10,MAX_DPT=1.0,blurclip=10)

# for st_iter, sample_batch in enumerate(loaders[0]):
#     X=sample_batch['rgb'].float().to('cuda')
#     depth=sample_batch['depth']
#     blur=sample_batch['blur']
#     seg=sample_batch['seg']
#     gt_step1=blur.float().to('cuda')
#     gt_step2=depth.float().to('cuda')
#     gt_step1=torch.unsqueeze(gt_step1,dim=1)
#     gt_step2=torch.unsqueeze(gt_step2,dim=1)
#     break

# import matplotlib.pyplot as plt
# img=X[9,:,:,:].detach().cpu().permute(1,2,0)
# d=gt_step1[9,0,:,:].detach().cpu()
# f, axarr = plt.subplots(1,2)
# axarr[0].imshow(img)
# axarr[1].imshow(d)
# plt.show()


# img=sample_batch['input'].squeeze().numpy()[0,:,:]
# plt.imshow(seg[0,:,:])
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



datapath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\'
blurclip=1

'''
stats:
depth
min 0.1  max 13.73

blur 
min 0.0  max  74.20
'''

def get_data_stats(datapath,blurclip):
    loaders, total_steps = load_data(datapath,blur=1,train_split=0.8,WORKERS_NUM=0,
        BATCH_SIZE=1,MAX_DPT=1.0,blurclip=1.0)
    print('stats of train data')
    get_loader_stats(loaders[0])
    print('______')

# import matplotlib.pyplot as plt
# plt.imshow(gt_step2[0,:,:])
# plt.show()

# img=torch.permute(X[0,:,:,:],(1,2,0))
# plt.imshow(img)
# plt.show()

# mask=(seg>100)*(gt_step2>0)
# plt.imshow(mask[0,:,:])
# plt.show()

# b =  gt_step2 > 10.
# indices = b.nonzero()

# mask[0,30,247]

# f, axarr = plt.subplots(1,2)
# axarr[0].imshow(gt_step2[0,:,:])
# axarr[1].imshow(mask[0,:,:])
# plt.show()

# depthmax_=torch.max(gt_step2[mask>0]).cpu().item()

#data statistics of the input images
def get_loader_stats(loader):
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    for st_iter, sample_batch in enumerate(loader):
        X=sample_batch['rgb'].float()
        depth=sample_batch['depth']
        blur=sample_batch['blur']
        seg=sample_batch['seg']
        gt_step1=blur.float()
        gt_step2=depth.float()
        mask=(seg>100)*(gt_step2>0)*(gt_step2<1.0)
        m=torch.sum(mask).item()
        if(m<20000):
            continue

        xmin_=torch.min(X).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(X).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(X).cpu().item()
        count+=1
        
        depthmin_=torch.min(gt_step2[mask>0]).cpu().item()
        if(depthmin_<depthmin):
            depthmin=depthmin_
        depthmax_=torch.max(gt_step2[mask>0]).cpu().item()

        if(depthmax_>depthmax):
            depthmax=depthmax_
        depthmean+=torch.mean(gt_step2[mask>0]).cpu().item()

        blurmin_=torch.min(gt_step1[mask>0]).cpu().item()
        if(blurmin_<blurmin):
            blurmin=blurmin_
        blurmax_=torch.max(gt_step1[mask>0]).cpu().item()
        if(blurmax_>blurmax):
            blurmax=blurmax_
        blurmean+=torch.mean(gt_step1[mask>0]).cpu().item()

    print('X min='+str(xmin))
    print('X max='+str(xmax))
    print('X mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))

    print('blur min='+str(blurmin))
    print('blur max='+str(blurmax))
    print('blur mean='+str(blurmean/count))

# get_data_stats(datapath,blurclip)

'''
blur_thres=7.0
p=3.1/256*1e-3 # pixel width in m
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,1.5]

get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres)
'''







