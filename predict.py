import torch
import torch.optim as optim
import torch.utils.data
import argparse
from dataloaders import nyuloader
from model import defnet
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='predict hand depth')
parser.add_argument('--depthscale', default=1.9,help='divide all depths by this value')
parser.add_argument('--checkpt', default='C:\\Users\\lahir\\models\\handblur\\best_ep0.pth', help='path to the saved model')

f=25e-3
N=1.9
px=6*1e-6*6
s1=0.7
kcam=f**2/(N*px)

args = parser.parse_args()


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

def predict_depth_map(imgpath):
    im=cv2.imread(imgpath,cv2.IMREAD_UNCHANGED)
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
    output_step1,output_step2,_ = model(X,camparam=X2_fcs,foc_dist=s1_fcs)
    return output_step2

output_step2=predict_depth_map(imgpath)
pred=output_step2.detach().cpu().squeeze().numpy()
pred*=args.depthscale
import matplotlib.pyplot as plt
plt.imshow(pred)
plt.show()

