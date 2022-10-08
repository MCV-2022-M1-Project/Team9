import cv2
import numpy as np

class Image:
    def __init__(self, file_directory: str, id:int, descriptorConfig:dict) -> None:
        self.file_directory = file_directory
        self.id = id    #numerical id of the image in str format
        self.mask = []  #initialise mask as empty, if the mask is [] it will be ignored. In cases where background will be removed, this mask will get updated with the foreground estimate mask
        #image information is not saved into image objects (ironically). If the image information is needed, it can be read with file_directory (image path). This is to avoid storing all images into the DB and only use images when needed
    
    def convert_image_grey_scale(self, BGR_image) -> np.ndarray:
        return cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)

    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
    
    def compute_histogram_grey_scale(self, BGR_image,nbins:int):
        #temporary, update once its done
        grey_scale_image = self.convert_image_grey_scale(BGR_image)
        
        if len(self.mask)>0:
            
            hist, bin_edges = np.histogram(grey_scale_image, bins=nbins, weights = self.mask)
        else:
            hist, bin_edges = np.histogram(grey_scale_image, bins=nbins)
        return hist
    
    
    def compute_descriptor(self, descriptorConfig:dict):
        #generate descriptor
        descriptorType = descriptorConfig.get("descriptorType")
        if descriptorType=="1Dhistogram":
            nbins = descriptorConfig.get("nbins")
            histogramType = descriptorConfig.get("histogramType")
            self.descriptor= self.compute_histogram(histogramType, nbins)

    def compute_histogram(self, histogram_type:str, nbins:int):
        """Computes the histogram of a given image. The histogram type (grayscale, concatenated histograms,...) can be selected with histogram_type


        """
        #read image
        BGR_image =self.read_image_BGR()
        if len(self.mask)>0:
            #set  foreground pixels to 1
            self.mask = self.mask/255
            print("AFTER",BGR_image)
            print("MASK",self.mask)
            cv2.imwrite(str(str(self.id)+"maskedImg.png"), BGR_image)

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
    #temporarily pasted code from Task.py to test fscore metrics!! 
        image = cv2.imread(self.file_directory)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        boundaries = [
        ([7, 5, 150], [70, 70, 200])
        ]

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            mask = cv2.inRange(image, lower, upper)
            mask_inv=255-mask

        masked_image = cv2.bitwise_and(image, image, mask=mask_inv)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR) 

        cv2.imwrite(str(str(self.id)+"mask.png"), mask_inv)
        
        cv2.imwrite(str(str(self.id)+"maskedImg.png"), masked_image)

        return mask_inv
            