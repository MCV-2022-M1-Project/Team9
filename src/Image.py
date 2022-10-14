import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import math

class Image:
    def __init__(self, file_directory: str, id:int) -> None:
        self.file_directory = file_directory    
        self.id = id    #numerical id of the image in str format
        self.mask = []  #initialise mask as empty, if the mask is [] it will be ignored. In cases where background will be removed, this mask will get updated with the foreground estimate mask
        #image information is not saved into image objects (ironically). If the image information is needed, it can be read with file_directory (image path). This is to avoid storing all images into the DB and only use images when needed
        self.image_filename = file_directory.split("/")[-1]


    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
        
    def convert_image_grey_scale(self, BGR_image) -> np.ndarray:
        return cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)
        
    def plot_histogram_grey_scale(self):
        plt.plot(self.histogram_grey_scale_image)
        plt.show()
    
    def plot_histogram_RGB(self):
        plt.plot(self.histogram_rgb_image)
        plt.show()
    

    
    def compute_descriptor(self, descriptorConfig:dict):
        """
        Given the descriptor configuration, it computes it and stores it into the descriptor property
            descriptorConfig: dictionary containing information on how to generate the descriptor
        """
        #generate descriptor
        descriptorType = descriptorConfig.get("descriptorType")

        #if it's a 1D histogram
        if descriptorType=="1Dhistogram":
            nbins = descriptorConfig.get("nbins")
            histogramType = descriptorConfig.get("histogramType")
            self.descriptor= self.compute_histogram(histogramType, nbins)

        #TODO: multiresolution histogram

    ### HISTOGRAM-RELATED FUNCTIONS
    def compute_histogram(self, histogram_type:str, nbins:int):
        """Computes the histogram of a given image. The histogram type (grayscale, concatenated histograms,...) can be selected with histogram_type
            histogram_type: if GRAYSCALE is selected it will compute the 1d grayscale histogram. If histogram_type contains "HSV", "BGR", "YCBCR" or "LAB" it will
                compute the concatenated HSV/BGR/YCBCR/LAB histogram
            nbins: # of bins of the resultant histogram

        """
        #read image
        BGR_image =self.read_image_BGR()
        if len(self.mask)>0:
            #set  foreground pixels to 1
            self.mask = self.mask/255

        if histogram_type=="GRAYSCALE":
            histogram = self.compute_histogram_grey_scale(BGR_image, nbins)
        #by default, compute 1d concatenated histogram if its not the grayscale one
        else:
            histogram = self.compute_histogram_3channel(BGR_image,nbins, histogram_type)

        #normalise histogram to not take into account the amount of pixels/how big the picture is into the similarity comparison
        norm_histogram = histogram/sum(histogram)
        
        #cast to float64 just in case
        norm_histogram = np.float64(norm_histogram)
        return norm_histogram
    
    def compute_histogram_grey_scale(self, image,nbins:int):
        """
        Computes the histogram greyscale histogram. If the image has a mask, it will not take into account the pixels marked as background
            image: Image to obtain the histogram from
            nbins: #of bins of the histogram

        """
        #convert to gray
        grey_scale_image = self.convert_image_grey_scale(image)
        
        if len(self.mask)>0:
            
            hist, bin_edges = np.histogram(grey_scale_image, bins=nbins, weights = self.mask)
        else:
            hist, bin_edges = np.histogram(grey_scale_image, bins=nbins)
        return hist
    
    def compute_histogram_3channel(self, BGR_image,nbins:int, colourSpace:str):
        """Computes the concatenated 3 channel histogram of an image, aka an array containing sequentially all of the 1D histograms of each channel
            BGR_image: Image to compute the histogram of
            nbins: # of bins of the resultant histogram
            colourSpace: colour space where the histogram will be computed in

        """
        if colourSpace=="BGR":
            image = BGR_image
        elif colourSpace=="HSV":
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
        elif colourSpace=="YCRCB":
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2YCrCb)
        elif colourSpace=="LAB":
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2Lab)
        
        if len(self.mask)>0:
        
            chan1_hist, bin_edges = np.histogram(image[:,:,0], bins=nbins, weights = self.mask)
            chan2_hist, bin_edges = np.histogram(image[:,:,1], bins=nbins, weights = self.mask)
            chan3_hist, bin_edges = np.histogram(image[:,:,2], bins=nbins, weights = self.mask)
        
        else:
            chan1_hist, bin_edges = np.histogram(image[:,:,0], bins=nbins)
            chan2_hist, bin_edges = np.histogram(image[:,:,1], bins=nbins)
            chan3_hist, bin_edges = np.histogram(image[:,:,2], bins=nbins)
        hist = np.concatenate([chan1_hist, chan2_hist,chan3_hist])
        return hist



    ### BACKGROUND REMOVAL FUNCTIONS
    def remove_background(self, method:str):
        """Removes the background of the image and saves it in a path. If computeGT is set to True, it will also compute the precision/recall of the mas compared to the GT
            method: method used to remove the background
        """
        im = cv2.imread(self.file_directory)
        im = cv2.medianBlur(im,3)
        #remove background using otsu
        if method =="OTSU":
            print("Removing background with OTSU")
            mask = self.remove_background_otsu(im = im)
        #remove background using colour thresholds
        elif(method =="LAB" or method=="HSV"):
            mask = self.remove_background_color(im = im, colorspace=method)
        elif(method=="MORPHOLOGY"):
            mask = self.remove_background_morph()

        return mask

    def remove_background_color(self, im, colorspace='HSV', debug=False):
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
                
                # print(channels[channel]['hist_threshold'])
                # print(channels[channel]['bins'])
                channels[channel]['range'] = [channels[channel]['bins'][min_i], channels[channel]['bins'][max_i]]
                # channels[channel]['range'] = [channels[channel]['bins'][min_i], 255]
                # print(channels[channel]['range'])
        
        # Remove boundaries
        # Add custom threshold because histogram does not take into account last values
        weights = [20,40,60]    #added custom tolerance to enlarge the found intervals of each channel with different values
        boundaries = [(
            [channels['channel_1']['range'][0]-weights[0], channels['channel_2']['range'][0]-weights[1], channels['channel_3']['range'][0]-weights[2]],
            [channels['channel_1']['range'][1] +weights[0], channels['channel_2']['range'][1] + weights[1]/2, channels['channel_3']['range'][1] + weights[2]/2]
        )]
        # boundaries = [(
        #     [128, 0, 0],
        #     [255, 255, 255]
        # )]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="int16")
            upper = np.array(upper, dtype="int16")
            mask = 255 - cv2.inRange(image_transformed, lower, upper)
            # print(mask)
            # plt.imshow(mask,cmap='gray')
            # plt.show()
        
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
            
        return mask
    def remove_background_otsu(self, im):
        """
            Given an image, a threshold is found using otsu and the resulting mask of the binarisation is outputted
            im: BGR image to remove the background from
        """
        im = self.convert_image_grey_scale(im)  #grayscale image
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
        
            # im_merged = cv2.bitwise_and(im, im, mask=im_binary)
            # im_cropped = crop_background(im_merged)
            
            
        return im_binary*255

    def remove_background_morph(self):
        #define kernels
        kernel_size_close = 30
        kernel_size_close2 = 70
        kernel_size_remove = 1500
        kernel_size_open = 200
        
        gradient_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close,kernel_size_close))
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_close2,kernel_size_close2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_open,kernel_size_open))
        
        kernel_close_vert = cv2.getStructuringElement(cv2.MORPH_RECT,(2,kernel_size_remove))
        kernel_close_hor = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size_remove,2))

        #obtain gradient of grayscale image
        img = self.read_image_BGR()
        img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        mask = Image.crop_img(mask ,padding,padding,padding,padding)
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

    ### GIVEN A MASK, COUNT PAINTINGS
    def count_paintings(self, max_paintings: int):
        self.mask = self.mask.astype(np.uint8)
        #count mask connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        heights = stats[1:,3]
        widths = stats[1:,2]
        paintings = []
        #for each mask
        for i in range(0, nb_components):
            temp_mask = np.zeros((output.shape))
            temp_mask[output == i + 1] = 1
            possible_painting = Image(self.file_directory,self.id)
            possible_painting.mask = temp_mask
            paintings.append(possible_painting)
        
        if len(paintings)>max_paintings:
            #if necessary, obtain the most possible mask
            paintings = paintings[:max_paintings]

        return paintings