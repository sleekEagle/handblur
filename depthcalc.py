#calculate real depth in mm from depth png files

import matplotlib.pyplot as plt
import cv2
import numpy as np

d=cv2.imread('D:\\handsdata\\nyu_hand_dataset_v2(1)\\dataset\\train\\depth_1_0000012.png',cv2.IMREAD_UNCHANGED)

d0=d[:,:,0].astype(np.uint16)
d1=d[:,:,1].astype(np.uint16)
d1=np.left_shift(d1,8)
r=d0+d1
plt.imshow(r)
plt.show()