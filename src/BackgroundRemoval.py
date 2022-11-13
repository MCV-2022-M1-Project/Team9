import cv2

import matplotlib.pyplot as plt
import numpy as np
import src.Image

class BackgroundRemoval:

    @staticmethod
    def remove_background_color( im, colorspace='HSV', debug=False):
        """
            Given an image and within a specified colorspace, the background colour is estimated (normalised histogram maxima). Afterwards, the image is thresholded considering background
            the pixels with values close to the predicted histogram maximas (intervals stored in boundaries variable) and foreground everything else
            im: BGR image to remove the background of
            colorspace: colorspace used to compute the histograms in
        """
        # Convert image to specified colorspace in order to create background mask
        if colorspace == 'LAB':
            image_transformed = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        elif colorspace == 'HSV':
            image_transformed = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
       
        # Extract image borders
        border_size = 70   #size in px of the border used to obtain the background colour estimate
        tmp = [image_transformed[:,:border_size,:], image_transformed[:,-border_size:,:],image_transformed[:border_size,:,:], image_transformed[-border_size:,:,:]]
        for border in tmp:
            border = border.reshape(-1, border.shape[-1])
            im_orig_borders = np.stack(border, axis=1)
        im_orig_borders = im_orig_borders.transpose()

        # Search in the image borders, for each channel of the specified colorspace, the predominant values
        channel_1 = im_orig_borders[:,0]
        channel_2 = im_orig_borders[:,1]
        channel_3 = im_orig_borders[:,2]
        nbins = 32
        hist_channel_1, bins_channel_1 = np.histogram(channel_1, bins=nbins)
        hist_channel_2, bins_channel_2 = np.histogram(channel_2, bins=nbins)
        hist_channel_3, bins_channel_3 = np.histogram(channel_3, bins=nbins)

        prob_threshold_channels = [0.1,0.1,0.1]
        channels = {
            'channel_1': {
                'hist_orig': hist_channel_1,
                'hist_norm': [],
                'bins': bins_channel_1,
                'prob_threshold': prob_threshold_channels[0],
                'hist_threshold': [],
                'range': [np.min(image_transformed[:, :, 0]), np.max(image_transformed[:, :, 0])]
            },
            'channel_2': {
                'hist_orig': hist_channel_2,
                'hist_norm': [],
                'bins': bins_channel_2,
                'prob_threshold': prob_threshold_channels[1],
                'hist_threshold': [],
                'range': [np.min(image_transformed[:, :, 1]), np.max(image_transformed[:, :, 1])]
            },
            'channel_3': {
                'hist_orig': hist_channel_3,
                'hist_norm': [],
                'bins': bins_channel_3,
                'prob_threshold': prob_threshold_channels[2],
                'hist_threshold': [],
                'range': [np.min(image_transformed[:, :, 2]), np.max(image_transformed[:, :, 2])]
            }
        }
        
        # Search on the concatenated borders, histogram channel values with the largest amount of bins
        # We do this using a normalized probability histogram of each original histogram
        # Then we find the contiguous regions with most probability to remove them from the original image
        for channel in channels:
            # Normalize histogram
            channels[channel]['hist_norm'] = channels[channel]['hist_orig'] / np.sum(channels[channel]['hist_orig'])
            # Threshold histogram
            channels[channel]['hist_threshold'] = channels[channel]['hist_norm'] > channels[channel]['prob_threshold']
                
            # From https://stackoverflow.com/questions/68514880/finding-contiguous-regions-in-a-1d-boolean-array
            # How to seek for contiguous regions efficiently using numpy
            # Used to find most prob regions
            runs = np.flatnonzero(np.diff(np.r_[np.int8(0), channels[channel]['hist_threshold'].view(np.int8), np.int8(0)])).reshape(-1, 2)

            # Search region of most probability
            most_prob = 0
            most_prob_region = 0
            for i in range(len(runs)):
                run_for_sum = runs[i]
                if runs[i][-1] == len(channels[channel]['hist_norm']):
                    run_for_sum[-1] = run_for_sum[-1] - 1
                else:
                    run_for_sum = runs[i]
                    
                prob_sum = channels[channel]['hist_norm'][run_for_sum].sum()
                
                if prob_sum > most_prob:
                    most_prob = prob_sum
                    most_prob_region = runs[i]
                
            # Save range of most probable values the borders have in this channel 
            if most_prob > 0:
                if most_prob_region[0] == 0:
                    min_i = 0
                else:
                    min_i = most_prob_region[0] - 1
                    
                max_i = most_prob_region[-1]
                
                channels[channel]['range'] = [channels[channel]['bins'][min_i], channels[channel]['bins'][max_i]]
        
        # Remove boundaries
        # Add custom threshold because histogram does not take into account last values
        weights = [20,40,60]    #added custom tolerance to enlarge the found intervals of each channel with different values
        boundaries = [(
            [channels['channel_1']['range'][0]-weights[0], channels['channel_2']['range'][0]-weights[1], channels['channel_3']['range'][0]-weights[2]],
            [channels['channel_1']['range'][1] +weights[0], channels['channel_2']['range'][1] + weights[1]/2, channels['channel_3']['range'][1] + weights[2]/2]
        )]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="int16")
            upper = np.array(upper, dtype="int16")
            mask = 255 - cv2.inRange(image_transformed, lower, upper)
        
        # If debug option is true, show mask and normalized histogram values
        if debug == True:
            plt.subplot(1,4,1)
            # cv2.bitwise_and(im,im, mask=mask)
            plt.imshow(mask,cmap='gray')    
            plt.subplot(1,4,2)
            plt.plot(channels['channel_1']['bins'][:-1], channels['channel_1']['hist_norm'], linestyle='--', marker='o')
            plt.subplot(1,4,3)
            plt.plot(channels['channel_2']['bins'][:-1], channels['channel_2']['hist_norm'], linestyle='--', marker='o')
            plt.subplot(1,4,4)
            plt.plot(channels['channel_3']['bins'][:-1], channels['channel_3']['hist_norm'], linestyle='--', marker='o')
            plt.show()
            
        if(np.sum(mask)==0):
            mask = 255-mask
        cv2.imwrite("HSV.png",mask)
        return mask
    
    @staticmethod
    def remove_background_otsu(im):
        """
            Given an image, computes a mask containing the foreground using otsu thresholding and morphological operations

        """
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((100,100),np.uint8)
        blur = cv2.GaussianBlur(im,(5,5),0)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,thresh1 = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        kernel2 = np.ones((10,10),np.uint8)
        opening1 = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
        closing1 = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel2)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        im_binary=255-closing1
        return im_binary

    @staticmethod
    def remove_background_morph(img):
        """
            Given an image, the gradient of its grayscale version is computed (edges of the image) and they are expanded with morphological operations
        
        """
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_greyscale = cv2.medianBlur(img_greyscale, 5)
        #define kernel sizes and kernels
        kernel_size_close = 20
        kernel_size_close2 = 20
        kernel_size_open = 20
        
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close,kernel_size_close))
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close2,kernel_size_close2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_open,kernel_size_open))

        #obtain gradient of grayscale image
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, gradient_kernel)
    
        #binarise gradient
        temp, gradient_binary = cv2.threshold(gradient,30,255,cv2.THRESH_BINARY)
        mask = gradient_binary[:,:,0]
        

        #add zero padding for morphology tasks 
        padding = 50
        mask = cv2.copyMakeBorder( mask,  padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
        
        #slight closing to increase edge size
        mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        #flood image starting from edge (result will have white background)
        _,mask_flooded,_,_ = cv2.floodFill(mask.copy(), None, (0, 0), 255)
        #get area of interest
        mask = mask_flooded-mask

        #small opening and closing
        
        mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
        mask = mask =cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = BackgroundRemoval.crop_img(mask ,padding,padding,padding,padding)

        #invert it (background -> 0, foreground-> 255)
        mask = 255-mask.astype(np.uint8)

        #remove connected components under a specific size
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        heights = stats[1:,3]
        widths = stats[1:,2]
        paintings = []
        height, width,_ = img.shape
        fraction = 5
        min_height= height/fraction
        min_width = width/fraction

        #set maximum on how many components to check in case there's too many
        max_components = min(nb_components,10)
        temp_mask = np.zeros((output.shape))
        #for each connected component
        for i in range(0, max_components):
            #write it into resultant mask if its big enough
            if(heights[i]>min_height and widths[i]>min_width):
                temp_mask[output == i + 1] = 255
        mask = temp_mask

        return mask
    
    @staticmethod
    def remove_background_canny(img):
        """
            Given an image, obtains the binarised gradient with canny and fills the background
        
        """
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_greyscale = cv2.GaussianBlur(img_greyscale,(5,5),0)
        img_greyscale = cv2.medianBlur(img_greyscale, 5)
        #define kernel sizes and kernels
        kernel_size_close = 30
        kernel_size_close2 = 50
        kernel_size_open = 50
        
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close,kernel_size_close))
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close2,kernel_size_close2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_open,kernel_size_open))

        cv2.imwrite("1_GREYSCALE.png",img_greyscale)
        #obtain binarised gradient of grayscale image
        mask = cv2.Canny(img,20,100)

        cv2.imwrite("canny.png",mask)
        #add zero padding for morphology tasks 
        padding = 50
        mask = cv2.copyMakeBorder( mask,  padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
        
        
        #slight closing to increase edge size
        mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        cv2.imwrite("morph.png",mask)
        #flood image starting from edge (result will have white background)
        _,mask_flooded,_,_ = cv2.floodFill(mask.copy(), None, (0, 0), 255)
        
        cv2.imwrite("flood.png",mask_flooded)
        #get area of interest
        mask = mask_flooded-mask

        cv2.imwrite("substract.png",mask)
        #cv2.imwrite("fill.png",mask)
        #small opening and closing
        
        mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
        mask = mask =cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = BackgroundRemoval.crop_img(mask ,padding,padding,padding,padding)

        #invert it (background -> 0, foreground-> 255)
        mask = 255-mask.astype(np.uint8)

        #remove connected components under a specific size
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        heights = stats[1:,3]
        widths = stats[1:,2]
        paintings = []
        height, width,_ = img.shape
        fraction = 6
        min_height= height/fraction
        min_width = width/fraction

        #set maximum on how many components to check in case there's too many
        max_components = min(nb_components,10)
        temp_mask = np.zeros((output.shape))
        #for each connected component
        for i in range(0, max_components):
            #write it into resultant mask if its big enough
            if(heights[i]>min_height and widths[i]>min_width):
                temp_mask[output == i + 1] = 255
        mask = temp_mask

        #return full mask if nothing got detected
        if(np.sum(mask)==0):
            mask = 255-mask
        
        cv2.imwrite("final.png",mask)
        return mask

    @staticmethod
    def remove_background_otsu_2( im):
        """
            Given an image, a threshold is found using otsu and the resulting mask of the binarisation is outputted
            im: BGR image to remove the background from
        """
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        n_classes = 2
        otsu_thresholds = skimage.filters.threshold_multiotsu(im, classes=n_classes, nbins=256)
        otsu_thresholds = otsu_thresholds.tolist()

        # Add the last class to sum probability
        otsu_thresholds.append(255)

        # Threshold to remove
        max_threshold = 1
        min_threshold = max_threshold - 1
        
        im_binary = np.asarray(im < otsu_thresholds[min_threshold], dtype='uint8')
        
        im_without_background = im * im_binary
        im_binary_inverted = 1-im_binary
        if(sum(im_binary[1,:]+im_binary[-1,:])+sum(im_binary[:,1]+im_binary[:,-1]))>(sum(im_binary_inverted[1,:]+im_binary_inverted[-1,:])+sum(im_binary_inverted[:,1]+im_binary_inverted[:,-1])):
            im_binary = im_binary_inverted
        
        # Crop background if necessary
        def crop_background(image, threshold=0):
            if len(image.shape) == 3:
                Im = np.max(image, 2)
            else:
                Im = image
            assert len(Im.shape) == 2

            rows = np.where(np.max(Im, 0) > threshold)[0]
            if rows.size:
                cols = np.where(np.max(Im, 1) > threshold)[0]
                image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
            else:
                image = image[:1, :1]
                
            return image
            
        return im_binary*255

    @staticmethod
    def remove_background_morph_old(img):
        """
            Given an image, the gradient of its grayscale version is computed (edges of the image) and they are expanded with morphological operations
        
        """
        
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #define kernels
        kernel_size_close = 20
        kernel_size_close2 = 100
        kernel_size_remove = 1500
        kernel_size_open = 70
        
        
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close,kernel_size_close))
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close2,kernel_size_close2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_open,kernel_size_open))
        
        kernel_close_vert = cv2.getStructuringElement(cv2.MORPH_RECT,(2,kernel_size_remove))
        kernel_close_hor = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_remove,2))

        #obtain gradient of grayscale image
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, gradient_kernel)
        
        #binarise gradient
        temp, gradient_binary = cv2.threshold(gradient,30,255,cv2.THRESH_BINARY)
        mask = gradient_binary[:,:,0]
        

        #add zero padding for morphology tasks 
        padding = 1500
        mask = cv2.copyMakeBorder( mask,  padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
        
        #slight closing to increase edge size
        mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        #really wide closing in horizontal and vertical directions
        temp1 = mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_vert)
        
        temp2 = mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_hor)

        #the mask will be the intersection
        mask = cv2.bitwise_and(temp1, temp2)
        
        #small opening and closing
        mask = mask =cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask =cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close2)
        mask = BackgroundRemoval.crop_img(mask ,padding,padding,padding,padding)

        mask = mask.astype(np.uint8)

        return mask
        
    @staticmethod 
    def crop_img(image_array,top,bottom,left,right):
        """ Cuts off the specified amount of pixels of an image
            top,bottom,keft,right: amount of px to crop in each direction
            
        """
        height = image_array.shape[0]
        width = image_array.shape[1]
        cropped_image = image_array[int(top):int(height-bottom),int(left):int(width-right)]
        return cropped_image