import torch
import torch.optim as optim
import torch.utils.data
import argparse
from dataloaders import nyuloader
from model import defnet
import cv2
import numpy as np
import mediapipe as mp
import handdetection
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='predict hand depth')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--checkpt', default='C:\\Users\\lahir\\models\\handblur\\best_ep0.pth', help='path to the saved model')

f=25e-3
N=1.9
px=6*1e-6*6
s1=0.2
kcam=f**2/(N*px)

args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


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

imgpath='C:\\Users\\lahir\\Downloads\\handres.jpg'

#read image with cv2 and give this function the np array of the image
#this will return the depth prediction
def predict_depth_map(im):
    # im=cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
    im = np.array(im)
    img = im.copy() / 255.
    img=np.expand_dims(img,axis=-1)
    img=img.transpose((3,2, 0, 1))
    X=torch.from_numpy(img).to('cuda').float()

    stacknum=1
    X2_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
    s1_fcs = torch.ones([X.shape[0], 1 * stacknum, X.shape[2], X.shape[3]])
    for t in range(stacknum):
        #iterate through the batch
        for i in range(X.shape[0]):
            X2_fcs[i, t:(t + 1), :, :] = X2_fcs[i, t:(t + 1), :, :]/kcam*(0.7-25e-3)*(100)
            s1_fcs[i, t:(t + 1), :, :] = s1_fcs[i, t:(t + 1), :, :]*(0.7)
    X2_fcs = X2_fcs.float().to('cuda')
    s1_fcs = s1_fcs.float().to('cuda')
    with torch.no_grad():
        output_step1,output_step2,_ = model(X,camparam=X2_fcs,foc_dist=s1_fcs)
    return output_step2*args.depthscale


imgpath='C:\\Users\\lahir\\Downloads\\near.jpg'
imgpath=
im=cv2.imread(imgpath)
show_img_depth(im)
def show_img_depth(im):
    if(not im.shape==(1920,1080,3)):
        print('cannot proccess')
    resized = cv2.resize(im, (550,978), interpolation = cv2.INTER_AREA)
    #detect hands from this image
    res=handdetection.get_cropped_image(resized)

    cropped=res[0]
    coords=res[1]
    joints=res[2]
    x=[j[0] for j in joints]
    y=[j[1] for j in joints]
    #these are flipped in x dimention
    middle_x=np.mean(np.array(x))
    middle_y=np.mean(np.array(y))
    ry,rx,_=resized.shape
    middle_x=rx-middle_x

    cropped_x=middle_x-coords[0]
    cropped_y=middle_y-coords[2]

    dptmap=predict_depth_map(cropped)
    dptmap=dptmap.cpu().squeeze().numpy()
    plt.imshow(dptmap)
    plt.show()

    #get mean depth of middle part of hand
    middle_depth=np.mean(dptmap[int(cropped_y)-10:int(cropped_y)+10 , int(cropped_x)-10:int(cropped_x)+10])

    #show the depth in the resized image as overlay
    fig, axs = plt.subplots(1, 1, figsize=(15,15))
    axs.imshow(resized)
    axs.plot(middle_x, middle_y, "og", markersize=10)
    axs.annotate(text = str(int(middle_depth*100))+" cm", xy = (middle_x+10,middle_y+5), fontsize = 18, color = 'white')
    plt.show()  


