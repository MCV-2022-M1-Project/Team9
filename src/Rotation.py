import numpy as np
import cv2

from scipy.ndimage import rotate as rotate_image

class Rotation:   

    @staticmethod
    def fix_image_rotation(padded_image,image, painting):
        """
        Given an image, estimates the rotation of the paintings in it and returns the image after correcting the rotation. 
        padded_image: cropped image that only includes 1 painting with an amount of padding
        image: full query image
        painting: current painting object (has its mask as an attribute)
        """
        img = padded_image
        #estimate rotation angle 
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append((area, cnt))
            rect = cv2.minAreaRect(cnt)

        areas.sort(key=lambda x: x[0], reverse=True)
        #areas.pop(0) # remove biggest contour
        x, y, w, h = cv2.boundingRect(areas[0][1]) # get bounding rectangle around biggest contour to crop to
        img = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255,0,0), 2)
        crop = thresh[y:y+h, x:x+w] # crop to size

        edges = cv2.Canny(crop, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200) # Find lines in image

        if lines is None:
            angle = 0
            angle_ini = 0
            delta_x = 0
            delta_y = 0
            pass
        else:
            img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) # Convert cropped black and white image to color to draw the red line
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 5) # draw line

                delta_x = x2 - x1
                delta_y = y2 - y1
                
                if delta_x !=0 and delta_y!=0:
                    angle = np.arctan(delta_y/delta_x) * 180 / np.pi
                    angle_ini = angle
                    #set angle to -45,45 degree range if necessary
                    if angle > 45 and angle < 135:
                        angle = angle-90
                    elif angle >135 and angle < 180:
                        angle = angle-180
                    elif angle <-45 and angle>-135:
                        angle = angle+90
                    elif angle <-135 and angle>-180:
                        angle = angle+180
                    
                else:
                    angle = 0
                    angle_ini = 0
            
        #rotate image and mask
        rotated_mask = rotate_image(painting.mask,angle)
        rotated_image = rotate_image(image,angle, mode='mirror')

        #trim image and mask into mask size
        ret,rotated_mask = cv2.threshold(rotated_mask,127,255,cv2.THRESH_BINARY)    
        rotated_image,_,_ = painting.crop_image_with_mask_bbox(img = rotated_image, mask = rotated_mask, margins = 0)
        rotated_mask,_,_ = painting.crop_image_with_mask_bbox(img = rotated_mask, mask = rotated_mask, margins = 0)


        #Obtain edges of the rotated bounding box on the mask (coordinates)
        cnts = cv2.findContours(painting.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        rect = cv2.minAreaRect(cnts[0])
        box = np.int0(cv2.boxPoints(rect))
        coord1 = box[0]
        coord2 = box[1]
        coord3 = box[2]
        coord4 = box[3]
        coordinates_original_domain = [coord1, coord2, coord3, coord4]

        #ensure that coordinates are within the bounds of the image
        img_h = image.shape[0]
        img_w = image.shape[1]
        for coordinates in coordinates_original_domain:
            coordinates[0] = max(coordinates[0],0)
            coordinates[1] = max(coordinates[1],0)
            
            coordinates[0] = min(coordinates[0],img_w-1)
            coordinates[1] = min(coordinates[1],img_h-1)
        
        #convert to proper format (0-180)
        if abs(delta_y)>abs(delta_x): #if hough line is vertical 
            if angle_ini<0 and angle<0:
                angle = angle + 180
            elif angle_ini<0 and angle>0:
                angle = 180 - angle
            elif angle_ini>0:
                angle = -angle          
        elif abs(delta_y)<abs(delta_x): #if hough line is horizontal
            if angle_ini<0 and angle<0:
                angle = -angle
            elif angle_ini<0 and angle>0:
                angle = angle + 180
            elif angle_ini>0:
                angle = 180 - angle
        print("ANGLE ", angle)
        return rotated_image, rotated_mask, angle, coordinates_original_domain