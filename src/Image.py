import cv2
import numpy as np

class Image:
    def __init__(self, file_directory: str, id:int, descriptorConfig:dict) -> None:
        self.file_directory = file_directory
        self.id = id    #numerical id of the image in str format
        #generate descriptor
        
        descriptorType = descriptorConfig.get("descriptorType")
        if descriptorType=="1Dhistogram":
            nbins = descriptorConfig.get("nbins")
            histogramType = descriptorConfig.get("histogramType")
            self.descriptor= self.compute_histogram(histogramType, nbins)
    
    def convert_image_grey_scale(self, BGR_image) -> np.ndarray:
        return cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)

    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
    #Task 1
    def compute_histogram_grey_scale(self, BGR_image,nbins:int):
        #temporary, update once its done
        grey_scale_image = self.convert_image_grey_scale(BGR_image)
        hist, bin_edges = np.histogram(grey_scale_image, bins=nbins)
        return hist
    
    #Task 1
    
    def compute_histogram(self, histogram_type:str, nbins:int):
        """Computes the histogram of a given image. The histogram type (grayscale, concatenated histograms,...) can be selected with histogram_type


        """
        #read image
        BGR_image = self.read_image_BGR()
        if histogram_type=="GRAYSCALE":
            histogram = self.compute_histogram_grey_scale(BGR_image, nbins)
        
        elif histogram_type=="BGR":
            histogram = self.compute_histogram_BGR(BGR_image,nbins)

        #normalise histogram to not take into account the amount of pixels/how big the picture is into the similarity comparison
        norm_histogram = histogram/sum(histogram)
        
        #cast to float64 just in case
        norm_histogram = np.float64(norm_histogram)
        return norm_histogram
    
    def compute_histogram_BGR(self, BGR_image,nbins:int):
        #TODO (output is wrong)
        return np.bincount((cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)

    
    #Task 1
    def plot_histogram_grey_scale(self):
        pass
    
    #Task 1
    def plot_histogram_RGB(self):
        pass
    
    #Task 5
    def remove_background(self):
        pass
    