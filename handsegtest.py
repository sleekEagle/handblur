import os
import sys
import cv2
sys.path.insert(0, "C:\\Users\\lahir\\code\\Monk_Object_Detection\\9_segmentation_models\\lib")
from infer_segmentation import Infer

gtf = Infer();


classes_dict = {
    'background': 0, 
    'hand': 1
};
classes_to_train = ['hand'];
gtf.Data_Params(classes_dict, classes_to_train, image_shape=[716,1024])
gtf.Model_Params(model="Unet", backbone="efficientnetb3", path_to_model='C://Users//lahir//code//models//seg_hand_trained//best_model.h5')
gtf.Setup();
gtf.Predict("C://Users//lahir//Downloads//handre.jpg", vis=True); #path of the image in the traine model folder should be used.

cv2.imread("C://Users//lahir//Downloads//handre.jpg")




# import module0
# import scipy
# from scipy import signal

# from scipy.signal import savgol_filter, medfilt
# from infer_segmentation import Infer


import torch
import torch.hub
import matplotlib.pyplot as plt

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)


# Inference
model.eval()
img_rnd = torch.randn(1, 3, 256, 256) # [B, C, H, W]
img=cv2.imread('C://Users//lahir//Downloads//handalla.jpg')
img=torch.from_numpy(img).permute(2,0,1).unsqueeze(dim=0).float()
preds = model(img).argmax(1) # [B, H, W]
preds=torch.squeeze(preds)

plt.imshow(preds)
plt.imshow(preds)
plt.show()





# Imports
import torch
import torch.hub

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch',
    model='hand_segmentor',
    pretrained=True
)





