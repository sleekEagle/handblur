import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#image heigth and width both must be even numbers
IMG_WIDTH,IMG_HEIGTH=256,256

#get cropped image and coordinates around the hand
#input: np array from cv2.imread
imgpt='C:\\Users\\lahir\\Downloads\\IMG_20230401_025011.jpg'
def get_cropped_image(img):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        image = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #no hand landmarks detected
        if not results.multi_hand_landmarks:
            return -1
        image_height, image_width, _ = image.shape

        # annotated_image = image.copy()
        
        #get all landmarks
        x,y=[],[]
        coords=[]
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
            # annotated_image,
            # hand_landmarks,
            # mp_hands.HAND_CONNECTIONS,
            # mp_drawing_styles.get_default_hand_landmarks_style(),
            # mp_drawing_styles.get_default_hand_connections_style())
            # cv2.imwrite('D:\\handsdata\\nyu_out\\train\\test.png', cv2.flip(annotated_image, 1))
            lm=hand_landmarks.landmark
            #get important landmark coordinates to get middle of the hand
            coords.append([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width,
                           hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height])
            coords.append([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
                           hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height])
            coords.append([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width,
                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height])
            coords.append([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width,
                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height])
            coords.append([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width,
                           hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height])
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
        return cropped,(minx,maxx,miny,maxy),coords