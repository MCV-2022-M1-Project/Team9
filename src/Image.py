import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage

from evaluation.evaluation_funcs import performance_accumulation_pixel, performance_evaluation_pixel
class Image:
    def __init__(self, file_directory: str, id:int, descriptorConfig:dict) -> None:
        self.file_directory = file_directory
        self.id = id    #numerical id of the image in str format
        self.mask = []  #initialise mask as empty, if the mask is [] it will be ignored. In cases where background will be removed, this mask will get updated with the foreground estimate mask
        #image information is not saved into image objects (ironically). If the image information is needed, it can be read with file_directory (image path). This is to avoid storing all images into the DB and only use images when needed
        self.image_filename = file_directory.split("/")[-1]

    def convert_image_grey_scale(self, BGR_image) -> np.ndarray:
        return cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)

    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
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
    
    
    def compute_descriptor(self, descriptorConfig:dict):
        """
        Given the descriptor configuration, it computes it and stores it into the descriptor property
        """
        #generate descriptor
        descriptorType = descriptorConfig.get("descriptorType")
        if descriptorType=="1Dhistogram":
            nbins = descriptorConfig.get("nbins")
            histogramType = descriptorConfig.get("histogramType")
            self.descriptor= self.compute_histogram(histogramType, nbins)

    def compute_histogram(self, histogram_type:str, nbins:int):
        """Computes the histogram of a given image. The histogram type (grayscale, concatenated histograms,...) can be selected with histogram_type
            histogram_type:

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
    
    def compute_histogram_3channel(self, BGR_image,nbins:int, colourSpace:str):
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

    
    #Task 1
    def plot_histogram_grey_scale(self):
        plt.plot(self.histogram_grey_scale_image)
        plt.show()
    
    #Task 1
    def plot_histogram_RGB(self):
        plt.plot(self.histogram_rgb_image)
        plt.show()


    def remove_background(self, save_masks_path:str, computeGT:str, method:str):
        im = cv2.imread(self.file_directory)
        if method =="OTSU":
            print("Removing background with OTSU")
            mask = self.remove_background_otsu(im = im, color = False)
        else:
            mask = self.remove_background_color(im = im, colorspace=method)
        #save mask into inputted path
        cv2.imwrite(str(save_masks_path+str(self.id).zfill(5)+".png"), mask)

        if (computeGT =='True'):
            #load gt mask
            mask_gt_path = str(self.file_directory.split(".jpg")[0]+".png")
            mask_gt = cv2.imread(mask_gt_path,0)
            #compute metrics
            [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(mask,mask_gt)
            print([pixelTP, pixelFP, pixelFN, pixelTN])
            [pixel_precision, pixel_accuracy, pixel_specificity, pixel_recall] = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
            if(pixel_precision==0 or pixel_recall==0):
                pixel_F1_score = 0
            else:
                pixel_F1_score = 2*float(pixel_precision) *float(pixel_recall)/ float(pixel_recall+pixel_precision)
            print("PRECISION: ", pixel_precision)
            print("RECALL: ", pixel_recall)
            print("F1 SCORE: ", pixel_F1_score)
        else:
            pixel_precision = -1
            pixel_recall = -1
            pixel_F1_score = -1
        
        return mask, pixel_precision, pixel_recall, pixel_F1_score
    # Remove background with a specified colorspace
    def remove_background_color(self, im, colorspace='HSV', debug=False):
        # Convert image to specified colorspace in order to create background mask
        if colorspace == 'LAB':
            image_transformed = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        elif colorspace == 'HSV':
            image_transformed = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        # Extract image borders
        tmp = [image_transformed[:,:100,:], image_transformed[:,-100:,:],image_transformed[:100,:,:], image_transformed[-100:,:,:]]
        for border in tmp:
            border = border.reshape(-1, border.shape[-1])
            im_orig_borders = np.stack(border, axis=1)
        im_orig_borders = im_orig_borders.transpose()

        # Search in the image borders, for each channel of the specified colorspace, the predominant values
        channel_1 = im_orig_borders[:,0]
        channel_2 = im_orig_borders[:,1]
        channel_3 = im_orig_borders[:,2]
        
        hist_channel_1, bins_channel_1 = np.histogram(channel_1, bins=4)
        hist_channel_2, bins_channel_2 = np.histogram(channel_2, bins=4)
        hist_channel_3, bins_channel_3 = np.histogram(channel_3, bins=4)
        
        channels = {
            'channel_1': {
                'hist_orig': hist_channel_1,
                'hist_norm': [],
                'bins': bins_channel_1,
                'prob_threshold': 0.1,
                'hist_threshold': [],
                'range': [np.min(image_transformed[:, :, 0]), np.max(image_transformed[:, :, 0])]
            },
            'channel_2': {
                'hist_orig': hist_channel_2,
                'hist_norm': [],
                'bins': bins_channel_2,
                'prob_threshold': 0.1,
                'hist_threshold': [],
                'range': [np.min(image_transformed[:, :, 1]), np.max(image_transformed[:, :, 1])]
            },
            'channel_3': {
                'hist_orig': hist_channel_3,
                'hist_norm': [],
                'bins': bins_channel_3,
                'prob_threshold': 0.1,
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
        boundaries = [(
            [channels['channel_1']['range'][0], channels['channel_2']['range'][0], channels['channel_3']['range'][0]],
            [channels['channel_1']['range'][1] + 30, channels['channel_2']['range'][1] + 20, channels['channel_3']['range'][1] + 20]
        )]
        # boundaries = [(
        #     [128, 0, 0],
        #     [255, 255, 255]
        # )]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
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
    
    def remove_background_otsu(self, im, color=False):
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