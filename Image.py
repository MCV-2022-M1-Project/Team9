import pandas as pd 
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Image:
    def __init__(self, file_directory: str) -> None:
        self.file_directory = file_directory
        self.grey_scale_image = self.read_image_grey_scale()
        self.RGB_image = self.read_image_RGB()
        self.histogram_grey_scale_image = self.compute_histogram_grey_scale()
        self.histogram_rgb_image = self.compute_histogram_RGB()
    
    def read_image_grey_scale(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_GRAYSCALE)
    
    # Assuming entry color is RGB
    def convert_image_color(self, color='LAB') -> np.ndarray:
        im_BGR = cv2.cvtColor(self.RGB_image, cv2.COLOR_RGB2BGR)
        if color == 'LAB':
            color_code = cv2.COLOR_BGR2LAB
        elif color == 'HSV':
            color_code = cv2.COLOR_BGR2HSV
        elif color == 'YCBCR':
            color_code = cv2.COLOR_BGR2YCR_CB            
        return cv2.cvtColor(im_BGR, color_code)

    def read_image_RGB(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
    def transform_RGB_image(self) -> np.ndarray:
        # Pad the RGB image for testing purposes
        top = 200
        bottom = 200
        left = 200
        right = 300
        mode = 'constant'
        constant_values = (0)
        im_transform = np.pad(self.RGB_image, ((top, bottom), (left, right), (0, 0)), mode, constant_values=constant_values)
        return im_transform

    
    #Task 1
    def compute_histogram_grey_scale(self):
        self.histogram_grey_scale_image = cv2.calcHist([self.grey_scale_image], [0], None, [256], [0, 256])
    
    #Task 1
    def compute_histogram_RGB(self):
        histograms = []
        n_histograms = 1
        if len(self.RGB_image.shape) > 2:
            n_histograms = self.RGB_image.shape[2]

        for i in range(0, n_histograms):
            histogram = cv2.calcHist([self.RGB_image], [i], None, [256], [0, 256])
            histograms.append(histogram)

        histograms = np.array(histograms)
        self.histogram_rgb_image = np.sum(histograms, 0)
        
    
    #Task 1
    def plot_histogram_grey_scale(self):
        plt.plot(self.histogram_grey_scale_image)
        plt.show()
    
    #Task 1
    def plot_histogram_RGB(self):
        plt.plot(self.histogram_rgb_image)
        plt.show()
    
    #Task 5
    def remove_background(self):
        pass
    
    
