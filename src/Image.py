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