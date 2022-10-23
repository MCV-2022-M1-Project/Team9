from tkinter import N
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import glob
import skimage
import os
class TextDetection :

    def mask_from_contours(ref_im, contours):
        mask = np.zeros(ref_im.shape, np.uint8)
        mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
        cv2.fillPoly(mask, pts =[contours], color=(255,255,255))
        
        # Kernel size should vary with mask size
        kernel  = np.ones((41,41), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        new_im = ref_im * dilated
        
        new_im = (new_im[:,:,0] + new_im[:,:,1] + new_im[:,:,2]).astype(np.int16)
        new_im_flatten = new_im.flatten()
        new_im_flatten = new_im_flatten[new_im_flatten != 0]
        
        values, counts = np.unique(new_im_flatten, return_counts=True)
        ind = np.argmax(counts)
        mask_im = np.array((new_im == values[ind]) * 255).astype(np.uint8)
        mask_im = cv2.erode(mask_im, np.ones((5,5), np.uint8), iterations=1)
        # Kernel size should vary with mask size
        mask_im = cv2.dilate(mask_im, np.ones((20,20), np.uint8), iterations=1)
        
        # plt.imshow(mask_im)
        # plt.show()
        # sys.exit()
        
        # Find the contours of the image and if each bounding box is a rectangle or not
        contours, hierarchy = cv2.findContours(mask_im, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        new_contours = []
        for contour in contours:
            epsilon = 0.04*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            contour_area = cv2.contourArea(contour)
            
            x, y, w, h = cv2.boundingRect(contour)
            rectanglessness = contour_area / (w * h)
            # print(rectanglessness)
            # print(contour_area)
            if contour_area > 1500 and contour_area < 80000:
                if rectanglessness > 0.8:
                    new_contours.append(contour)
                    # print(rectanglessness)
        
        # plt.imshow(mask_im)
        # plt.show()
        
        # print(new_contours)
        
        # sys.exit()    
        return new_contours

    def get_text_bounding_box(im, im_gray, im_hsv, get_negative=False, morph_open=3, extra_open=0, laplacian_kernel_center_weight=4, debug=False):
        # We save pixels that are not saturated on the image to remove the saturated later. We do that because the text bounding boxes are somewhat black and white
        pixels_saturated = abs((im_hsv[:,:,1] > 60) - 1)
        
        # Not a laplacian kernel perse, a modification
        laplacian_kernel = np.array([[0,-1,0],
                    [-1,laplacian_kernel_center_weight,-1],
                    [0,-1,0]])
        
        # You may need to use abs(255-im_gray) instead of abs(255) to find the text bounding box on images with different contrast
        # Find image edges with a "somewhat" Laplacian filter
        if get_negative == True:
            src = 255-im_gray
        else:
            src = im_gray
        
        laplacian_image = cv2.filter2D(src=abs(src), ddepth=-1, kernel=laplacian_kernel)
        laplacian_image = cv2.morphologyEx(laplacian_image, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        laplacian_image = cv2.morphologyEx(laplacian_image, cv2.MORPH_OPEN, np.ones((morph_open,morph_open),np.uint8))
        if extra_open != 0:
            laplacian_image = cv2.morphologyEx(laplacian_image, cv2.MORPH_OPEN, np.ones((extra_open,extra_open),np.uint8))
        # laplacian_image = cv2.morphologyEx(laplacian_image, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        # laplacian_image = cv2.morphologyEx(laplacian_image, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        # Threshold low "gradients"
        laplacian_image = ((laplacian_image < 210) * 255).astype(np.uint8)

        laplacian_without_saturation = (laplacian_image * pixels_saturated).astype(np.uint8)

        # Find the contours of the image and if each bounding box is a rectangle or not
        contours, hierarchy = cv2.findContours(laplacian_without_saturation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # https://stackoverflow.com/questions/61166180/detect-rectangles-in-opencv-4-2-0-using-python-3-7
        new_contours = []
        for contour in contours:
            epsilon = 0.04*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            contour_area = cv2.contourArea(contour)
            if contour_area > 2000 and contour_area < 80000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w/h
                rectanglessness = contour_area / (w * h)
                if rectanglessness > 0.8 and aspect_ratio > 2.5:
                    new_contours.append(contour)
                # else:
                #     mask_contours = mask_from_contours(im_hsv, contour)
                #     for mask_contour in mask_contours:
                #         new_contours.append(mask_contour)
        cv2.drawContours(im,new_contours,-1,(0,255,0),3)
        
        if debug == True:
            # Plot everything
            plt.subplot(1,2,1)
            plt.imshow(im)
            # plt.subplot(1,2,2)
            # plt.imshow(laplacian_image, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(laplacian_without_saturation, cmap='gray')
            plt.show()
            
        return new_contours
                

    laplacian_kernel_center_weights = [4,8,6,10,12,16]
    morph_opens = [3, 12]
    extra_opens = [0, 3, 12]

    # From https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    def order_points_old(pts):
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

    def detect_text(img):
        #for file in glob.glob('./Datasets/qsd2_w2/*.jpg'):
        # file = './Datasets/qsd2_w2/00015.jpg'
        im = img
        
        laplacian_kernel_center_weights = [4,8,6,10,12,16]
        morph_opens = [3, 12]
        extra_opens = [0, 3, 12]
        orig_im = im
        
        bounding_boxes = []
        bounding_box_im = np.ones((im.shape[0], im.shape[1]))
        
        
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        for weight in laplacian_kernel_center_weights:
            for morph_open in morph_opens:
                for extra_open in extra_opens:
                    bounding_box = TextDetection.get_text_bounding_box(im, im_gray, im_hsv, get_negative=False, morph_open=morph_open,
                                                        extra_open=extra_open, laplacian_kernel_center_weight=weight, debug=False)
                    for box in bounding_box:
                        rot_rect = cv2.minAreaRect(box)
                        box_points = cv2.boxPoints(rot_rect).astype(np.uint16)
                        bounding_box_im = cv2.rectangle(bounding_box_im, (box_points[0][0],box_points[0][1]), (box_points[2][0],box_points[2][1]), (0,0,0), -1)
                        
                    bounding_box = TextDetection.get_text_bounding_box(im, im_gray, im_hsv, get_negative=True, morph_open=morph_open,
                                                        extra_open=extra_open, laplacian_kernel_center_weight=weight, debug=False)
                    for box in bounding_box:
                        rot_rect = cv2.minAreaRect(box)
                        box_points = cv2.boxPoints(rot_rect).astype(np.uint16)
                        bounding_box_im = cv2.rectangle(bounding_box_im, (box_points[0][0],box_points[0][1]), (box_points[2][0],box_points[2][1]), (0,0,0), -1)
        
        # Get bounding boxes for test
        final_bounding_boxes = []
        # Only take the bigger boundingbox
        # Find the contours of the image and if each bounding box is a rectangle or not
        contours, _ = cv2.findContours((bounding_box_im).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # We dont want the first contour since it is all the image
        contours = contours[1:]
        for contour in contours:
            rot_rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rot_rect).astype(np.uint16)
            ordered_box = TextDetection.order_points_old(box_points).astype(np.uint16)
            
            # print('Ordered')
            # print(ordered_box)
            # print('Not ordered')
            # print(box_points)
            
            final_bounding_boxes.append([ordered_box[0][0], ordered_box[0][1], ordered_box[2][0], ordered_box[2][1]])

        bounding_box_im = bounding_box_im*255
        # plt.imshow(bounding_box_im, cmap='gray')
        # plt.show()
        
        if final_bounding_boxes ==[]:
            bounding_box_im =  np.ones((im.shape[0], im.shape[1]))*255
            return [0,20,0,20], bounding_box_im
        
        final_bounding_boxes = final_bounding_boxes[0]
        

        tlx1 = final_bounding_boxes[0]
        tly1 = final_bounding_boxes[1]
        brx1 = final_bounding_boxes[2]
        bry1 = final_bounding_boxes[3]
        tlx1, tly1, brx1, bry1
        return [tlx1, tly1, brx1, bry1], bounding_box_im
    
    def read_text(img):
        pass