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
    # Remove background with the border of an HSV image
    def remove_background_hsv_border(self, im, debug=False):
        image_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
        tmp = [image_hsv[:,:200,:], image_hsv[:,-200:,:],image_hsv[:100,:,:], image_hsv[-100:,:,:]]
        for border in tmp:
            border = border.reshape(-1, border.shape[-1])
            im_orig_borders = np.stack(border, axis=1)
        im_orig_borders = im_orig_borders.transpose()

        h = im_orig_borders[:,0]
        s = im_orig_borders[:,1]
        v = im_orig_borders[:,2]
        
        n_bins = 16
        hist_h, bins_h = np.histogram(h, bins=n_bins)
        hist_s, bins_s = np.histogram(s, bins=n_bins)
        hist_v, bins_v = np.histogram(v, bins=n_bins)
        
        channels = {
            'hue': {
                'hist': hist_h,
                'bins': bins_h,
                'hist_threshold': []
            },
            'saturation': {
                'hist': hist_s,
                'bins': bins_s,
                'hist_threshold': []
            },
            'value': {
                'hist': hist_v,
                'bins': bins_v,
                'hist_threshold': []
            }
        }
            
        prob_threshold = 0.15
        
        cannot_remove = 0
        
        for channel in channels:
            # Normalize histogram
            hist = channels[channel]['hist']
            channels[channel]['hist'] = hist / np.sum(hist)
            hist = channels[channel]['hist']
            # Threshold histogram
            channels[channel]['hist_threshold'] = hist > prob_threshold
            
            # Index positions for min and max indices of histogram
            max_args = channels[channel]['hist_threshold'].nonzero()[0]
            
            if len(max_args) >= 1:
                if max_args[0] == 0:
                    min_i = 0
                else:
                    min_i = max_args[0] - 1
                if max_args[-1] == (n_bins-1):
                    max_i = n_bins-1
                else:
                    max_i = max_args[-1] + 1
                
                # print(channels[channel]['hist_threshold'])
                # print(channels[channel]['bins'])
                channels[channel]['range'] = [channels[channel]['bins'][min_i], channels[channel]['bins'][max_i]]
                # print(channels[channel]['range'])
            else:
                cannot_remove = 1
            
        
        if cannot_remove != 1:
            boundaries = [(
                [channels['hue']['range'][0], channels['saturation']['range'][0], channels['value']['range'][0]],
                [channels['hue']['range'][1], channels['saturation']['range'][1], channels['value']['range'][1]]
            )]

            for (lower, upper) in boundaries:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask = 255 - cv2.inRange(image_hsv, lower, upper)
            
            if debug == True:
                plt.subplot(1,4,1)
                plt.imshow(mask,cmap='gray')    
                plt.subplot(1,4,2)
                plt.plot(channels['hue']['hist'], linestyle='--', marker='o')
                plt.subplot(1,4,3)
                plt.plot(channels['saturation']['hist'], linestyle='--', marker='o')
                plt.subplot(1,4,4)
                plt.plot(channels['value']['hist'], linestyle='--', marker='o')
                plt.show()
        else:
            mask = np.ones((im.shape[0], im.shape[1])) * 255
            
        return mask
    
    def remove_background(self, color=False):
        if color:
            hist = self.compute_histogram_RGB()
            im = self.BGR_image
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