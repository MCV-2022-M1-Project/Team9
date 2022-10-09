import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage

class Image:
    def __init__(self, file_directory: str, id:int) -> None:
        self.file_directory = file_directory
        self.id = id    #numerical id of the image in str format
        self.BGR_image = self.read_image_BGR()
        self.grey_scale_image = self.convert_image_grey_scale()
        self.histogram= self.compute_histogram("GRAYSCALE")
    
    def convert_image_grey_scale(self) -> np.ndarray:
        return cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)

    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
    #Task 1
    def compute_histogram_grey_scale(self):
        #temporary, update once its done
        hist, bin_edges = np.histogram(self.grey_scale_image, bins=16)
        return hist
        #return np.bincount((cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
    
    #Task 1
    
    def compute_histogram(self, histogram_type):
        
        if histogram_type=="GRAYSCALE":
            histogram = self.compute_histogram_grey_scale()
        
        elif histogram_type=="BGR":
            histogram = self.compute_histogram_BGR()

        #normalise histogram to not take into account the amount of pixels/how big the picture is into the similarity comparison
        norm_histogram = histogram/sum(histogram)
        return norm_histogram
    
    def compute_histogram_BGR(self):
        return np.bincount((cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)

    
    #Task 1
    def plot_histogram_grey_scale(self):
        plt.plot(self.histogram_grey_scale_image)
        plt.show()
    
    #Task 1
    def plot_histogram_RGB(self):
        plt.plot(self.histogram_rgb_image)
        plt.show()
    
    #Task 5
    # Remove background with a specified colorspace
    def remove_background_color(self, im, colorspace='lab', debug=True):
        # Convert image to specified colorspace in order to create background mask
        if colorspace == 'lab':
            image_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        elif colorspace == 'hsv':
            image_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        # Extract image borders
        tmp = [image_hsv[:,:100,:], image_hsv[:,-100:,:],image_hsv[:100,:,:], image_hsv[-100:,:,:]]
        for border in tmp:
            border = border.reshape(-1, border.shape[-1])
            im_orig_borders = np.stack(border, axis=1)
        im_orig_borders = im_orig_borders.transpose()

        # Search in the image borders, for each channel of the specified colorspace, the predominant values
        h = im_orig_borders[:,0]
        s = im_orig_borders[:,1]
        v = im_orig_borders[:,2]
        
        hist_h, bins_h = np.histogram(h, bins=2)
        hist_s, bins_s = np.histogram(s, bins=2)
        hist_v, bins_v = np.histogram(v, bins=2)
        
        channels = {
            'hue': {
                'hist_orig': hist_h,
                'hist_norm': [],
                'bins': bins_h,
                'prob_threshold': 0.1,
                'hist_threshold': [],
                'range': [np.min(image_hsv[:, :, 0]), np.max(image_hsv[:, :, 0])]
            },
            'saturation': {
                'hist_orig': hist_s,
                'hist_norm': [],
                'bins': bins_s,
                'prob_threshold': 0.1,
                'hist_threshold': [],
                'range': [np.min(image_hsv[:, :, 1]), np.max(image_hsv[:, :, 1])]
            },
            'value': {
                'hist_orig': hist_v,
                'hist_norm': [],
                'bins': bins_v,
                'prob_threshold': 0.1,
                'hist_threshold': [],
                'range': [np.min(image_hsv[:, :, 2]), np.max(image_hsv[:, :, 2])]
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
        boundaries = [(
            [channels['hue']['range'][0], channels['saturation']['range'][0], channels['value']['range'][0]],
            [channels['hue']['range'][1], channels['saturation']['range'][1], channels['value']['range'][1]]
        )]
        # boundaries = [(
        #     [128, 0, 0],
        #     [255, 255, 255]
        # )]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            mask = cv2.inRange(image_hsv, lower, upper)
            # print(mask)
            # plt.imshow(mask,cmap='gray')
            # plt.show()
        
        # If debug option is true, show mask and normalized histogram values
        if debug == True:
            plt.subplot(1,4,1)
            # cv2.bitwise_and(im,im, mask=mask)
            plt.imshow(mask,cmap='gray')    
            plt.subplot(1,4,2)
            plt.plot(channels['hue']['bins'][:-1], channels['hue']['hist_norm'], linestyle='--', marker='o')
            plt.subplot(1,4,3)
            plt.plot(channels['saturation']['bins'][:-1], channels['saturation']['hist_norm'], linestyle='--', marker='o')
            plt.subplot(1,4,4)
            plt.plot(channels['value']['bins'][:-1], channels['value']['hist_norm'], linestyle='--', marker='o')
            plt.show()
            
        return mask
    
    def remove_background(self, color=False):
        if color:
            hist = self.compute_histogram_RGB()
            im = self.BGR_image
            mask_im = self.remove_background_color(im, 'lab', False)
        else:
            hist = self.compute_histogram_grey_scale()
            im = self.grey_scale_image
        
            n_classes = 2
            otsu_thresholds = skimage.filters.threshold_multiotsu(im, classes=n_classes, nbins=256)
            otsu_thresholds = otsu_thresholds.tolist()

            # Add the last class to sum probability
            otsu_thresholds.append(255)

            # Threshold to remove
            max_threshold = 2
            min_threshold = max_threshold - 1
            
            im_binary = cv2.bitwise_or(np.asarray(im < otsu_thresholds[min_threshold], dtype='uint8'), np.asarray(im > otsu_thresholds[max_threshold], dtype='uint8'))
            
            im_without_background = im * im_binary
        
        return im_binary