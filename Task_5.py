import cv2
import numpy as np

def remove_background(self):
    image = cv2.imread(self)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    b = image_hsv[:,:,1]
    b = np.where(b < 127, 0, 1) #exclude values below 127
    
    w = (image_hsv[:,:,2] + 127) 
    w = np.where(w > 127, 1, 0)  #accept values above 127
    
    b_and_w = np.where(b+w > 0, 0, 1).astype(np.uint8)  
    mask = np.where(b_and_w==0,255,0).astype(np.uint8) 
    
    #result = cv2.bitwise_and(image,image,mask=mask)
    return mask
