import cv2
import mediapipe as mp
import numpy as np
from os import listdir
from os.path import isfile, join

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#image heigth and width both must be even numbers
IMG_WIDTH,IMG_HEIGTH=256,256

#get cropped image and coordinates around the hand
imagepath='D://handsdata//nyu_hand_dataset_v2(1)//dataset//train//rgb_1_0000701.png'
def get_cropped_image(imagepath):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(imagepath), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #no hand landmarks detected
        if not results.multi_hand_landmarks:
            return -1
        image_height, image_width, _ = image.shape

        # annotated_image = image.copy()
        
        
        #get all landmarks
        x,y=[],[]
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
            # annotated_image,
            # hand_landmarks,
            # mp_hands.HAND_CONNECTIONS,
            # mp_drawing_styles.get_default_hand_landmarks_style(),
            # mp_drawing_styles.get_default_hand_connections_style())
            # cv2.imwrite('D:\\handsdata\\nyu_out\\train\\test.png', cv2.flip(annotated_image, 1))
            lm=hand_landmarks.landmark
        for l in lm:
            x.append(l.x*image_width)
            y.append(l.y*image_height)
        
        #create a boundingbox for hand
        meanx,meany=np.mean(x),np.mean(y)
        minx_=int(meanx-IMG_WIDTH/2)
        maxx_=int(meanx+IMG_WIDTH/2)
        miny_=int(meany-IMG_HEIGTH/2)
        maxy_=int(meany+IMG_HEIGTH/2)

        # revert the flipping of the image (needed for media pipe)
        flipimg=cv2.flip(image, 1)
        maxx=image_width-minx_
        minx=image_width-maxx_
        maxy=maxy_
        miny=miny_

        if(minx<0):
            minx=0
            maxx=IMG_WIDTH
        if(miny<0):
            miny=0
            maxy=IMG_HEIGTH

        cropped=flipimg[miny:maxy,minx:maxx,:]
        return cropped,(minx,maxx,miny,maxy)
        # cv2.imwrite("C:\\Users\\lahir\\Downloads\\annot.png", cropped)
       
# path='D://handsdata//nyu_hand_dataset_v2(1)//dataset//train//rgb_1_0062605.png'

path='D:\\handsdata\\nyu_hand_dataset_v2(1)\\dataset\\train\\'
outpath='D:\\handsdata\\nyu_out\\train\\'
print('reading files...')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

rgbfiles=[file for file in onlyfiles if(file.split('_')[0]=='rgb')]
depthfiles=[file for file in onlyfiles if(file.split('_')[0]=='depth')]
rgbfiles.sort()
depthfiles.sort()

print('starting data prep...')
L=len(rgbfiles)
for i,rgb in enumerate(rgbfiles):
    perce=i/L*100
    print("%.2f %% finished"%perce,end='\r')
    #check if rgb and depth file are for the same scene
    if(not rgb[4:]==depthfiles[i][6:]):
        continue
    ret=get_cropped_image(path+rgb)
    if(ret==-1):
        continue
    cropped_rgb,coords=ret[0],ret[1]
    d=cv2.imread(path+depthfiles[i],cv2.IMREAD_UNCHANGED)
    #convert depth image to 16bit
    d0=d[:,:,0].astype(np.uint16)
    d1=d[:,:,1].astype(np.uint16)
    d1=np.left_shift(d1,8)
    depthimg=d0+d1
    cropped_depth=depthimg[coords[2]:coords[3],coords[0]:coords[1]]
    cv2.imwrite(outpath+"depth\\"+depthfiles[i], cropped_depth)
    cv2.imwrite(outpath+"rgb\\"+rgb, cropped_rgb)

# rgbimg=cv2.imread(path+rgb)
# dimg=cv2.imread(path+depthfiles[i])
# f, axarr = plt.subplots(2,1) 
# axarr[0].imshow(cropped_rgb)
# axarr[1].imshow(cropped_depth)
# plt.show()   

# import matplotlib.pyplot as plt

# print("%.2f percent finished" %12/4)

# ,end='\r')

# print('%.2f' % 1.23)



