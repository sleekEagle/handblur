import cv2
import mediapipe as mp
import numpy as np
from os import listdir
from os.path import isfile, join

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#image heigth and width both must be even numbers
IMG_WIDTH,IMG_HIGHT=256,256
PAD=40

#get cropped image and coordinates around the hand
#gives a fixed size crop of IMG_WIDTH,IMG_HIGHT
imagepath='C:\Users\lahir\kinect_hand_data\extracted\lahiru1\color\\'
def get_cropped_image_fixed(imagepath):
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
        miny_=int(meany-IMG_HIGHT/2)
        maxy_=int(meany+IMG_HIGHT/2)

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
            maxy=IMG_HIGHT

        cropped=flipimg[miny:maxy,minx:maxx,:]
        return cropped,(minx,maxx,miny,maxy)


#get cropped image and coordinates around the hand
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
        minx_,maxx_=int(np.min(x)),int(np.max(x))
        miny,maxy=int(np.min(y)),int(np.max(y))

        #padding
        minx_=max(0,minx_-PAD)
        maxx_=min(image_width,maxx_+PAD)
        miny=max(0,miny-PAD)
        maxy=min(image_height,maxy+PAD)

        # revert the flipping of the image (needed for media pipe)
        flipimg=cv2.flip(image, 1)
        maxx=image_width-minx_
        minx=image_width-maxx_

        cropped=flipimg[miny:maxy,minx:maxx,:]
        return cropped,(minx,maxx,miny,maxy)


'''
dataprep for our kinect dataset
'''

rgbpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\color\\'
depthpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\depth\\'
outpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\'
print('reading files...')
rgbfiles = [f for f in listdir(rgbpath) if (isfile(join(rgbpath, f)) and f.split('.')[-1]=='jpg')]
depthfiles = [f for f in listdir(depthpath) if (isfile(join(depthpath, f)) and f.split('.')[-1]=='png')]
rgbfiles.sort()
depthfiles.sort()

print('starting data prep...')
L=len(rgbfiles)
for i,rgb in enumerate(rgbfiles):
    perce=i/L*100
    print("%.2f %% finished"%perce,end='\r')
    #check if rgb and depth file are for the same scene
    if(not rgb.split('.')[0]==depthfiles[i].split('.')[0]):
        continue
    ret=get_cropped_image(rgbpath+rgb)
    if(ret==-1):
        continue
    cropped_rgb,coords=ret[0],ret[1]
    d=cv2.imread(depthpath+depthfiles[i],cv2.IMREAD_UNCHANGED)
    cropped_depth=d[coords[2]:coords[3],coords[0]:coords[1]]
    cv2.imwrite(outpath+"depth\\"+depthfiles[i], cropped_depth)
    cv2.imwrite(outpath+"rgb\\"+rgb, cropped_rgb)

#resize segmentation maps from 256x256 to the correct size
rgbpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\rgb\\'
segpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\seg\\'
segout='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\seg_resized\\'

rgbfiles = [f for f in listdir(rgbpath) if (isfile(join(rgbpath, f)) and f.split('.')[-1]=='jpg')]
segfiles = [f for f in listdir(segpath) if (isfile(join(segpath, f)) and f.split('.')[-1]=='jpg')]

rgbfiles.sort()
segfiles.sort()

for i,seg in enumerate(segfiles):
    #read the size of rgb file
    rgbimg=cv2.imread(rgbpath+rgbfiles[i],cv2.IMREAD_UNCHANGED)
    rgb_size=rgbimg.shape
    if(not seg==rgbfiles[i]):
        continue
    segimg=cv2.imread(segpath+seg,cv2.IMREAD_UNCHANGED)
    output = cv2.resize(segimg, (rgb_size[1],rgb_size[0]))
    res=cv2.imwrite(segout+seg,output)

    










   
'''
data prep for NUY hands
'''
path='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\color\\'
outpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\cropped\\'
print('reading files...')
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

rgbfiles=[file for file in onlyfiles if(file.split('.')[-1]=='jpg')]
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

'''
resize images to 256 x 256 so the segmentation can work on it
'''
path='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\color\\'
outpath='C:\\Users\\lahir\\kinect_hand_data\\extracted\\lahiru1\\seg_resized\\'
onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and f.split('.')[-1]=='jpg')]
for file in onlyfiles:
    src = cv2.imread(path+file, cv2.IMREAD_UNCHANGED)
    output = cv2.resize(src, (1920,1080))
    cv2.imwrite(outpath+file,output)


src = cv2.imread(r'C:\Users\lahir\kinect_hand_data\extracted\lahiru1\test\seg\res.jpg', cv2.IMREAD_UNCHANGED)
output = cv2.resize(src, (1128,1080))
cv2.imwrite(r'C:\Users\lahir\kinect_hand_data\extracted\lahiru1\test\\seg\res_scaledup.jpg',output)







