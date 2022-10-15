import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from src.Histograms import Histograms
from src.BackgroundRemoval import BackgroundRemoval
import math

class Image:
    def __init__(self, file_directory: str, id:int) -> None:
        self.file_directory = file_directory    
        self.id = id    #numerical id of the image in str format
        self.mask = []  #initialise mask as empty, if the mask is [] it will be ignored. In cases where background will be removed, this mask will get updated with the foreground estimate mask
        #image information is not saved into image objects (ironically). If the image information is needed, it can be read with file_directory (image path). This is to avoid storing all images into the DB and only use images when needed
        self.image_filename = file_directory.split("/")[-1]


    def read_image_BGR(self) -> np.ndarray:
        return cv2.imread(self.file_directory, cv2.IMREAD_COLOR)
        
    def convert_image_grey_scale(self, BGR_image) -> np.ndarray:
        return cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)
        
    def plot_histogram_grey_scale(self):
        plt.plot(self.histogram_grey_scale_image)
        plt.show()
    
    def plot_histogram_RGB(self):
        plt.plot(self.histogram_rgb_image)
        plt.show()
    
    def compute_descriptor(self, descriptorConfig:dict):
        """
        Given the descriptor configuration, it computes it and stores it into the descriptor property
            descriptorConfig: dictionary containing information on how to generate the descriptor
        """
        #generate descriptor
        descriptorType = descriptorConfig.get("descriptorType")

        #if it's a 1D histogram
        if descriptorType=="1Dhistogram":
            nbins = descriptorConfig.get("nbins")
            histogramType = descriptorConfig.get("histogramType")
            self.descriptor= self.compute_histogram(descriptorType,histogramType, nbins)

        #multiresolution histogram
        elif descriptorType=="mult_res_histogram":
            nbins = descriptorConfig.get("nbins")
            histogramType = descriptorConfig.get("histogramType")
            level = descriptorConfig.get("level")
            max_level = descriptorConfig.get("max_level")
            self.descriptor= self.compute_histogram(descriptorType,histogramType, nbins,max_level= max_level)


    def compute_histogram(self,descriptor_type:str, histogram_type:str, nbins:int,max_level=None):
        """Computes the histogram of a given image. The histogram type (grayscale, concatenated histograms,...) can be selected with histogram_type
            histogram_type: if GRAYSCALE is selected it will compute the 1d grayscale histogram. If histogram_type contains "HSV", "BGR", "YCBCR" or "LAB" it will
                compute the concatenated HSV/BGR/YCBCR/LAB histogram
            nbins: # of bins of the resultant histogram

        """
        #read image
        BGR_image =self.read_image_BGR()
        if descriptor_type =="1Dhistogram":
            if histogram_type=="GRAYSCALE":
                histogram = Histograms.compute_histogram_grey_scale(BGR_image, nbins,mask = self.mask)
            #by default, compute 1d concatenated histogram if its not the grayscale one
            else:
                histogram = Histograms.compute_histogram_3channel(BGR_image,nbins, histogram_type, mask=self.mask)
        elif descriptor_type=="mult_res_histogram":
            histogram = Histograms.compute_spatial_pyramid_representation(image=BGR_image, mask=self.mask, color_space=histogram_type, nbins=nbins, max_level=max_level)
        #normalise histogram to not take into account the amount of pixels/how big the picture is into the similarity comparison
        norm_histogram = histogram/sum(histogram)
        #cast to float64 just in case
        norm_histogram = np.float64(norm_histogram)
        return norm_histogram

    def remove_background(self, method:str):
        """Removes the background of the image and saves it in a path. If computeGT is set to True, it will also compute the precision/recall of the mas compared to the GT
            method: method used to remove the background
        """
        im = cv2.imread(self.file_directory)

        #remove background using otsu
        if method =="OTSU":
            print("Removing background with OTSU")
            mask = BackgroundRemoval.remove_background_otsu(im = im)
        #remove background using colour thresholds
        elif(method =="LAB" or method=="HSV"):
            mask = BackgroundRemoval.remove_background_color(im = im, colorspace=method)
        elif(method=="MORPHOLOGY"):
            im = cv2.medianBlur(im,5)
            mask = BackgroundRemoval.remove_background_morph(img=im)

        return mask



    ### GIVEN A MASK, COUNT PAINTINGS
    def count_paintings(self, max_paintings: int):
        self.mask = self.mask.astype(np.uint8)
        #count mask connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(self.mask, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        heights = stats[1:,3]
        widths = stats[1:,2]
        paintings = []
        min_size = 400
        #set maximum on how many components to check
        max_components = min(nb_components,10)
        #for each mask
        for i in range(0, max_components):
            temp_mask = np.zeros((output.shape))
            temp_mask[output == i + 1] = 255
            possible_painting = Image(self.file_directory,self.id)
            possible_painting.mask = temp_mask
            if(np.sum(temp_mask)>min_size):
                paintings.append(possible_painting)
        if len(paintings)>max_paintings:
            #if necessary, obtain the most possible mask
            paintings = paintings[:max_paintings]
        if len(paintings)>1:
            #sort them from left to right (temporary, only case ==2 )

            white_pixels1 = np.array(np.where(paintings[0].mask == 255))
            white_pixels1 = np.sort(white_pixels1)
            #get coordinates of the first and last white pixels (useful to set a mask bounding box)
            first_white_pixel1 = white_pixels1[:,0]

            white_pixels2 = np.array(np.where(paintings[1].mask == 255))
            white_pixels2 = np.sort(white_pixels2)

            #get coordinates of the first and last white pixels (useful to set a mask bounding box)
            first_white_pixel2 = white_pixels2[:,0]
            if first_white_pixel2[1]<first_white_pixel1[1]:
                temp = paintings[0]
                paintings[0] = paintings[1]
                paintings[1] =temp

        return paintings
    
    def crop_image_with_mask_bbox(self):
        white_pixels = np.array(np.where(self.mask == 255))
        white_pixels = np.sort(white_pixels)
        #get coordinates of the first and last white pixels (useful to set a mask bounding box)
        first_white_pixel = white_pixels[:,0]
        last_white_pixel = white_pixels[:,-1]
        img = self.read_image_BGR()
        #crop image with np slicing
        img_cropped = img[first_white_pixel[0]:last_white_pixel[0],first_white_pixel[1]:last_white_pixel[1]]
        return img_cropped,first_white_pixel
    
    ##move to another .py file if possible
    def detect_text(img):
        ##fill with proper code

        ####
        
        #return the bbox coordinates (TODO)
        tlx1 = 0   #top left
        tly1 = 0   #top right
        brx1 = 1   #bottom left
        bry1 = 1   #bottom right
        return [tlx1, tly1, brx1, bry1]