import cv2
import numpy as np

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

        if histogram_type=="GRAYSCALE":
            histogram = self.compute_histogram_grey_scale(BGR_image, nbins)
        
        elif histogram_type=="BGR":
            histogram = self.compute_histogram_3channel(BGR_image,nbins, "BGR")
        elif histogram_type=="HSV":
            histogram = self.compute_histogram_3channel(BGR_image,nbins, "HSV")
        elif histogram_type=="YCRCB":
            histogram = self.compute_histogram_3channel(BGR_image,nbins, "YCRCB")
        elif histogram_type=="LAB":
            histogram = self.compute_histogram_3channel(BGR_image,nbins, "YCRCB")

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
        
            B_hist, bin_edges = np.histogram(image[:,:,0], bins=nbins, weights = self.mask)
            G_hist, bin_edges = np.histogram(image[:,:,1], bins=nbins, weights = self.mask)
            R_hist, bin_edges = np.histogram(image[:,:,2], bins=nbins, weights = self.mask)
        
        else:
            B_hist, bin_edges = np.histogram(image[:,:,0], bins=nbins)
            G_hist, bin_edges = np.histogram(image[:,:,1], bins=nbins)
            R_hist, bin_edges = np.histogram(image[:,:,2], bins=nbins)
        hist = np.concatenate([B_hist, G_hist,R_hist])
        return hist

    
    #Task 1
    def plot_histogram_grey_scale(self):
        pass
    
    #Task 1
    def plot_histogram_RGB(self):
        pass
    
    #Task 5
    def remove_background(self, save_masks_path:str):
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
        filename_without_extension =  self.image_filename.split(".")[0]
        #save into inputted path
        cv2.imwrite(str(save_masks_path+str(self.id)+".png"), mask_inv)

        #load gt mask
        mask_gt_path = str(self.file_directory.split(".jpg")[0]+".png")
        mask_gt = cv2.imread(mask_gt_path,0)
        
        cv2.imwrite(str("test.png"), mask_gt)
        [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(mask_inv,mask_gt)
        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_recall] = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
        pixel_F1_score = 2*float(pixel_precision) *float(pixel_recall)/ float(pixel_recall+pixel_precision)
        return mask_inv, pixel_precision, pixel_recall, pixel_F1_score
            