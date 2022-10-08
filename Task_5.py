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
    
    #output = cv2.bitwise_and(image,image,mask=mask)
    return mask


def remove_background_2(self):
    image = cv2.imread(self)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    boundaries = [
	([7, 5, 150], [70, 70, 200])
    ]

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        mask_inv=255-mask

        #output = cv2.bitwise_and(image, image, mask=mask_inv)
        #output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        return mask_inv


def remove_background_3(self):
    image = cv2.imread(self)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    ret, mask = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

    output = cv2.bitwise_and(image, image, mask=mask)

    return mask