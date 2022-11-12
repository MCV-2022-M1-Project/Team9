import numpy as np
import cv2

from scipy.ndimage import rotate as rotate_image

class Rotation:   

    @staticmethod
    def fix_image_rotation(padded_image_to_rotate,image, mask, painting, offsets,img_dims):
        """
        Given an image, estimates the rotation of the paintings in it and returns the image after correcting the rotation. 

        """
        img = padded_image_to_rotate
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
                #print(delta_x,delta_y)
                
                if delta_x !=0 and delta_y!=0:
                    angle = np.arctan(delta_y/delta_x) * 180 / np.pi
                    print(angle)
                    if angle > 45 and angle < 135:
                        angle = angle-90
                    elif angle >135 and angle < 180:
                        angle = angle-180
                    elif angle <-45 and angle>-135:
                        angle = angle+90
                    elif angle <-135 and angle>-180:
                        angle = angle+180
                    
                    rotated_img = rotate_image(img,angle)
                else:
                    rotated_img = img
                    angle = 0
                print("ANGLE AFTER ",angle)
                

            #cv2.imwrite(root_folder + 'tests/'+'_hline.jpg', img)
            #cv2.imwrite(root_folder + os.path.basename(file).split('.')[0]+'_rotated.png', rotated_img)
            
        #rotate image -angle degrees to correct the rotation
        rotated_mask = rotate_image(mask,angle)
        rotated_image = rotate_image(image,angle, mode='mirror')
        
        ret,rotated_mask = cv2.threshold(rotated_mask,127,255,cv2.THRESH_BINARY)
        cv2.imwrite("./tests/"+'rotated_mask.png', rotated_mask)
        
        cv2.imwrite("./tests/"+'rotated_before.png', rotated_image)
        print("SHAPE ", rotated_mask.shape)
        rotated_image,top_left_coordinate_offset,bottom_right_coordinate_offset = painting.crop_image_with_mask_bbox(img = rotated_image, mask = rotated_mask, margins = 0)
        rotated_mask,_,_ = painting.crop_image_with_mask_bbox(img = rotated_mask, mask = rotated_mask, margins = 0)
        cv2.imwrite("./tests/"+'rotated.png', rotated_image)

        x1 = top_left_coordinate_offset[1]
        x2 = bottom_right_coordinate_offset[1]
        y1 = top_left_coordinate_offset[0]
        y2 = bottom_right_coordinate_offset[0]
        
        img_h = img_dims[0]
        img_w = img_dims[1]

        #Find contours, find rotated rectangle, obtain four verticies, and draw 
        cnts = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        rect = cv2.minAreaRect(cnts[0])
        box = np.int0(cv2.boxPoints(rect))
        print("BOX ", box)
        coord1 = box[0]
        coord2 = box[1]
        coord3 = box[2]
        coord4 = box[3]
        coordinates_original_domain = [coord1, coord2, coord3, coord4]
        print("COORDS OG ", coordinates_original_domain)
        for coordinates in coordinates_original_domain:
            coordinates[0] = max(coordinates[0],0)
            coordinates[1] = max(coordinates[1],0)
            
            coordinates[0] = min(coordinates[0],img_w-1)
            coordinates[1] = min(coordinates[1],img_h-1)
        

        return rotated_image, rotated_mask,  angle, coordinates_original_domain
    
    @staticmethod
    def rotate_coordinates(top_left, top_right, bottom_right, bottom_left, angle,origin, img_dims):
        """Given 4 coordinates and an angle, returns the same coordinates rotated by angle degrees
            top_left, top_right, bottom_right, bottom_left: Coordinates in the format (x,y)
            angle: angle in degrees between 0 and 180
        """
        ###TODO 
        rotated_points = Rotation.rotate([top_left, top_right,bottom_right,bottom_left], origin = origin,degrees = angle)
        top_left_rotated = rotated_points[0]
        print("TL ROTATED ", top_left_rotated)
        top_right_rotated = rotated_points[1]
        print("TE ROTATED ", top_right_rotated)
        bottom_right_rotated = rotated_points[2]
        print("BR ROTATED ", bottom_right_rotated)
        bottom_left_rotated = rotated_points[3]
        print("BL ROTATED ", bottom_left_rotated)

        return top_left_rotated, top_right_rotated, bottom_right_rotated, bottom_left_rotated
        
    def rotate(p, origin=(0, 0), degrees=0):
        """
        From https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
        """
        angle = np.deg2rad(degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T-o.T) + o.T).T)