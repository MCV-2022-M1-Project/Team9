import pandas as pd 
import cv2
import numpy as np

class Image:
    def __init__(self, file_directory: str, id:int) -> None:
        self.file_directory = file_directory
        self.id = id    #numerical id of the image in str format
        self.grey_scale_image = self.read_image_grey_scale()
        self.RGB_image = self.read_image_RGB()
        self.histogram_grey_scale_image = self.compute_histogram_grey_scale()
    
    def read_image_grey_scale(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_GRAYSCALE)

    def read_image_RGB(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
    #Task 1
    def compute_histogram_grey_scale(self):
        pass
    
    #Task 1
    def compute_histogram_RGB(self):
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
    
    
