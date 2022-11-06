from tkinter import N
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import glob
import pytesseract
import skimage
import os
import math

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

    def get_text_bounding_box_laplace(im, im_gray, im_hsv, get_negative=False, morph_open=3, extra_open=0, laplacian_kernel_center_weight=4, debug=False):
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
    
    def get_bounding_boxes_laplace(im, im_gray, im_hsv, bounding_box_im, laplacian_kernel_center_weights=[4,8,6,10,12,16], morph_opens=[3,12], extra_opens=[0,3,12]):
        for weight in laplacian_kernel_center_weights:
            for morph_open in morph_opens:
                for extra_open in extra_opens:
                    bounding_box = TextDetection.get_text_bounding_box_laplace(im, im_gray, im_hsv, get_negative=False, morph_open=morph_open,
                                                        extra_open=extra_open, laplacian_kernel_center_weight=weight, debug=False)
                    for box in bounding_box:
                        rot_rect = cv2.minAreaRect(box)
                        box_points = cv2.boxPoints(rot_rect).astype(np.uint16)
                        bounding_box_im = cv2.rectangle(bounding_box_im, (box_points[0][0],box_points[0][1]), (box_points[2][0],box_points[2][1]), (0,0,0), -1)
                        
                    bounding_box = TextDetection.get_text_bounding_box_laplace(im, im_gray, im_hsv, get_negative=True, morph_open=morph_open,
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
            
            final_bounding_boxes.append([ordered_box[0][0], ordered_box[0][1], ordered_box[2][0], ordered_box[2][1]])
        
        return bounding_box_im, final_bounding_boxes
    
    def resize_im(im, max_height=800, max_width=800):
        height, width, _ = im.shape
        factor = min(max_width / width, max_height / height)
        im = cv2.resize(im, (int(width * factor), int(height * factor)))
        return im

    # https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # Modified for the bounding boxes of mser, maximum supression
    #  Felzenszwalb et al.
    def non_max_suppression_slow(boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # print(x1)
        # sys.exit()
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list, add the index
            # value to the list of picked indexes, then initialize
            # the suppression list (i.e. indexes that will be deleted)
            # using the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
        # loop over all indexes in the indexes list
            for pos in range(0, last):
                # grab the current index
                j = idxs[pos]
                # find the largest (x, y) coordinates for the start of
                # the bounding box and the smallest (x, y) coordinates
                # for the end of the bounding box
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                # compute the width and height of the bounding box
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                # compute the ratio of overlap between the computed
                # bounding box and the bounding box in the area list
                overlap = float(w * h) / area[j]
                # if there is sufficient overlap, suppress the
                # current bounding box
                if overlap > overlapThresh:
                    suppress.append(pos)
            # delete all indexes from the index list that are in the
            # suppression list
            idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
        return boxes[pick]

    # Merge boxes (to create a big rectangular box, in case of text)


    def merge_boxes_mser(boxes):
        box = []

        min = np.min(boxes, axis=0)
        max = np.max(boxes, axis=0)

        x1 = min[0]
        y1 = min[1]
        x2 = max[2]
        y2 = max[3]

        box = [x1, y1, x2, y2]
        return box

    # Get 3 channel histogram to compare letters


    def get_3_channel_hist(im, nbins=8):
        chan1_hist, bin_edges = np.histogram(im[:, :, 0], bins=nbins, density=True)
        chan2_hist, bin_edges = np.histogram(im[:, :, 1], bins=nbins, density=True)
        chan3_hist, bin_edges = np.histogram(im[:, :, 2], bins=nbins, density=True)

        hist = np.concatenate([chan1_hist, chan2_hist, chan3_hist])
        hist = hist.astype(np.float32)
        return hist


    def group_bounding_boxes_mser(im, bounding_boxes_parsed, vis):
        regions_merged = []

        # We group bounding boxes based on their color and their slope, we want horizontal slope and similar colors
        for box in bounding_boxes_parsed:
            x1, y1, x2, y2 = box
            # plt.imshow(im[y1:y2,x1:x2], cmap='gray')
            # plt.show()

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

            hist = TextDetection.get_3_channel_hist(im[y1:y2, x1:x2])

            region_exist = False

            # plt.imshow(im[y1:y2,x1:x2])
            # plt.show()

            slope_threshold = 0.15
            hist_threshold = 3
            dist_threshold = 450

            # Merge regions based on some parameters
            for region in regions_merged:
                hist_comparison = cv2.compareHist(
                    hist, region[0]['hist'], cv2.HISTCMP_CHISQR)

                if (x1-region[0]['pos'][0]) == 0:
                    slope_top = 0
                else:
                    slope_top = (y1-region[0]['pos'][1])/(x1-region[0]['pos'][0])

                if (x2-region[0]['pos'][2] == 0):
                    slope_bottom = 0
                else:
                    slope_bottom = (y2-region[0]['pos'][3]) / \
                        (x2-region[0]['pos'][2])

                dist = math.dist([x1, y1, x2, y2], region[0]['pos'])

                if dist < dist_threshold and hist_comparison < hist_threshold and (abs(slope_top) < slope_threshold or abs(slope_bottom) < slope_threshold):
                    region_exist = True
                    region.append({'pos': [x1, y1, x2, y2], 'hist': hist})

            if region_exist == False:
                regions_merged.append([
                    {'pos': [x1, y1, x2, y2], 'hist': hist}
                ])

        return regions_merged


    def get_bounding_box_im_mser(im, height, width, regions_merged):
        final_bounding_boxes = []
        final_mask = np.zeros((height, width, 1), dtype=np.uint8)

        regions_merged = np.array(regions_merged)
        num_regions = regions_merged.shape[0]

        for i in range(num_regions):
            bounding_boxes_parsed = []
            
            for region in regions_merged[i]:
                x1, y1, x2, y2 = region['pos']
                bounding_boxes_parsed.append((x1, y1, x2, y2))
            bounding_boxes_parsed = np.array(bounding_boxes_parsed)

            # print(bounding_boxes_parsed.shape)

            # If less than n boxes
            if bounding_boxes_parsed.shape[0] < 4 or bounding_boxes_parsed.shape[0] > 20:
                continue

            mask = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.uint8)

            # If we want to merge this box
            bounding_boxes_parsed = [
                np.array(TextDetection.merge_boxes_mser(bounding_boxes_parsed))]

            for box in bounding_boxes_parsed:
                x1, y1, x2, y2 = box
                
                # Pad region (to increase the bounding box size)
                box_area = (x2-x1)*(y2-y1)
                pad = int(box_area/1500)
                x1 -= pad
                y1 -= pad
                x2 += pad
                y2 += pad
                
                final_bounding_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(mask, (x1, y1), (x2, y2), (1, 1, 1), -1)


            final_mask = cv2.bitwise_or(final_mask, mask)
            
            
        return final_mask, final_bounding_boxes
    
    def detect_text_mser(im, im_gray):
        delta = 16

        mser = cv2.MSER_create(delta=delta)
        vis = im.copy()
        regions, boundingBoxes = mser.detectRegions(im_gray)

        bounding_boxes_parsed = []

        for box in boundingBoxes:
            x1, y1, h, w = box
            bounding_boxes_parsed.append((x1, y1, x1+h, y1+w))

        bounding_boxes_parsed = np.array(bounding_boxes_parsed)
        
        bounding_boxes_parsed = TextDetection.non_max_suppression_slow(bounding_boxes_parsed, 0.3)
        regions_merged = TextDetection.group_bounding_boxes_mser(im, bounding_boxes_parsed, vis)
        
        bounding_box_im, final_bounding_boxes = TextDetection.get_bounding_box_im_mser(im, im.shape[0], im.shape[1], regions_merged)
        
        bounding_box_im = cv2.bitwise_not(bounding_box_im) / 255
        
        return bounding_box_im, final_bounding_boxes

    def detect_text(img, method='both'):
        im = img.copy()
        
        bounding_box_im = np.ones((im.shape[0], im.shape[1]))
        
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        if method == 'both' or method == 'laplace':
            bounding_box_im_laplace, final_bounding_boxes_laplace = TextDetection.get_bounding_boxes_laplace(im, im_gray, im_hsv, bounding_box_im)
            bounding_box_im_laplace = bounding_box_im_laplace.astype(np.uint8)
            
        
        if method == 'both' or method == 'mser':
            bounding_box_im_mser, final_bounding_boxes_mser = TextDetection.detect_text_mser(im, im_gray)
            bounding_box_im_mser = bounding_box_im_mser.astype(np.uint8)
        
        
        if method == 'both':
            bounding_box_im = cv2.bitwise_and(bounding_box_im_laplace, bounding_box_im_mser)
            
            final_bounding_boxes = final_bounding_boxes_laplace
            for final_bounding_box_mser in final_bounding_boxes_mser:
                final_bounding_boxes.append(final_bounding_box_mser)
            final_bounding_boxes = np.array(final_bounding_boxes)
        elif method == 'laplace':
            bounding_box_im = bounding_box_im_laplace
            final_bounding_boxes = final_bounding_boxes_laplace
        elif method == 'mser':
            bounding_box_im = bounding_box_im_mser
            final_bounding_boxes = final_bounding_boxes_mser
        
        # Get percentage of 1 and 0 in the final bounding box mask, if percentage is too big remove
        masks_percentage = bounding_box_im.sum() / (bounding_box_im.shape[0] * bounding_box_im.shape[1])
        
        if masks_percentage < 0.60:
            bounding_box_im = np.ones((im.shape[0], im.shape[1]))
            final_bounding_boxes = []
        
        bounding_box_im = bounding_box_im*255
        
        if final_bounding_boxes ==[] or len(final_bounding_boxes) == 0:
            bounding_box_im =  np.ones((im.shape[0], im.shape[1]))*255
            return [0,0,20,20], bounding_box_im
        
        final_bounding_boxes = final_bounding_boxes[0]
        

        tlx1 = final_bounding_boxes[0]
        tly1 = final_bounding_boxes[1]
        brx1 = final_bounding_boxes[2]
        bry1 = final_bounding_boxes[3]
        
        return [tlx1, tly1, brx1, bry1], bounding_box_im

    def read_text(textbox_img, j = 0):
        """Given an image containing a text box, returns the string of the text read in it using ocr
            textbox_img: image that ideally contains a textbox. The bounding box used to obtain this image is the one detected with detect_text
        """
        bounding_box_cropped_im =textbox_img
        
        #cv2.imwrite("./text/"+str(j)+".png",bounding_box_cropped_im)
        bounding_box_cropped_im_gray = cv2.cvtColor(bounding_box_cropped_im, cv2.COLOR_BGR2GRAY)
        th, bounding_box_cropped_im_binary = cv2.threshold(bounding_box_cropped_im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        total_black_px =np.sum(bounding_box_cropped_im_binary == 0)
        total_white_px = np.sum(bounding_box_cropped_im_binary == 255)
        if total_black_px>total_white_px:
            bounding_box_cropped_im_binary = 255-bounding_box_cropped_im_binary
        
        #remove small connected components
        bounding_box_cropped_im_binary = 255-bounding_box_cropped_im_binary
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bounding_box_cropped_im_binary, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        heights = stats[1:,3]
        widths = stats[1:,2]
        paintings = []
        min_size = 3
        max_width = output.shape[1]*0.6
        #for each mask
        temp_mask = np.zeros((output.shape))
        for i in range(0, nb_components):
            if(sizes[i]>min_size and widths[i]<max_width):
                temp_mask[output == i + 1] = 255

        bounding_box_cropped_im_binary = (255-temp_mask).astype(np.uint8)
        #cv2.imwrite("test2.png", bounding_box_cropped_im_binary)
        
        #cv2.imwrite("./text/"+"binarised"+str(j)+".png",bounding_box_cropped_im_binary)
        extractedInformation = pytesseract.image_to_string(bounding_box_cropped_im_binary, config="-c 'tessedit_char_whitelist=01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ' --psm 6")

        extractedInformation = extractedInformation.replace("\n", "")
        extractedInformation = extractedInformation.replace("", "")
        extractedInformation = extractedInformation.strip() #remove extra whitespaces
        return extractedInformation