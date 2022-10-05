import pandas as pd 
import cv2
import numpy as np

class Image:
    def __init__(self, file_directory: str, id:int) -> None:
        self.file_directory = file_directory
        self.id = id    #numerical id of the image in str format
        self.grey_scale_image = self.read_image_grey_scale()
        self.BGR_image = self.read_image_BGR()
        self.histogram_grey_scale_image = self.compute_histogram_grey_scale()
    
    def read_image_grey_scale(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_GRAYSCALE)

    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
    #Task 1
    def compute_histogram_grey_scale(self):
        #temporary, update once its done
        return np.bincount((cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
    
    #Task 1
    
    def compute_histogram(self, histogram_type):
        
        if histogram_type=="GRAYSCALE":
            histogram = self.compute_histogram_grey_scale()
        
        elif histogram_type=="BGR":
            histogram = self.compute_histogram_BGR()

        return histogram
    
    def compute_histogram_BGR(self):
        np.bincount((cv2.cvtColor(self.BGR_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
        pass
    
    #Task 1
    def plot_histogram_grey_scale(self):
        pass
    
    #Task 1
    def plot_histogram_RGB(self):
        pass
    
    #Task 5
    def remove_background(self):
        pass
    
    
