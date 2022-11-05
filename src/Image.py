import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from src.Histograms import Histograms
from src.TextureDescriptors import TextureDescriptors
from src.KeypointDescriptors import KeypointDescriptors
from src.BackgroundRemoval import BackgroundRemoval
import math

class Image:
    def __init__(self, file_directory: str, id:int, artist:str=None, title:str=None) -> None:
        self.file_directory = file_directory    
        self.id = id    #numerical id of the image in str format
        self.mask = []  #initialise mask as empty, if the mask is [] it will be ignored. In cases where background will be removed, this mask will get updated with the foreground estimate mask
        #image information is not saved into image objects (ironically). If the image information is needed, it can be read with file_directory (image path). This is to avoid storing all images into the DB and only use images when needed
        self.image_filename = file_directory.split("/")[-1]

        #if available, store the artist and title of the painting
        self.artist = artist
        self.title = title

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
    
    def compute_descriptor(self, image, descriptor_config_array:dict, cropped_img = None):
        """
        Given the descriptor configuration, it computes it and stores it into the descriptor property
            descriptor_config_array: array of dictionaries containing information on how to generate the descriptor
            cropped_img: cropped image in case there's multiple paintings
        """
        concatenated_descriptors = []
        descriptor = None
        #cropped_img is used for the texture descriptors in case the image has been cropped to separate it from the background. if it's empty, the entire image will be used
        if len(self.mask)==0:
            cropped_img = image


        for descriptor_config in descriptor_config_array:
            #generate descriptor
            descriptorType = descriptor_config.get("descriptorType")
            weight = descriptor_config.get("weight")
            
            #1D histogram
            if descriptorType=="1Dhistogram":
                nbins = descriptor_config.get("nbins")
                histogramType = descriptor_config.get("histogramType")
                descriptor= self.compute_histogram(image,descriptorType,histogramType, nbins)

            #multiresolution histogram
            elif descriptorType=="mult_res_histogram":
                nbins = descriptor_config.get("nbins")
                histogramType = descriptor_config.get("histogramType")
                max_level = descriptor_config.get("max_level")
                descriptor= self.compute_histogram(image,descriptorType,histogramType, nbins,max_level= max_level)
            
            #block histogram
            elif descriptorType=="block_histogram":
                nbins = descriptor_config.get("nbins")
                histogramType = descriptor_config.get("histogramType")
                level = descriptor_config.get("level")
                descriptor= self.compute_histogram(image,descriptorType,histogramType, nbins,level= level)

            #hog histogram
            elif descriptorType=="HOG":
                descriptor= TextureDescriptors.compute_HOG(cropped_img)

            #lbp histogram
            elif descriptorType=="LBP":
                lbp_radius = descriptor_config.get("lbp_radius")
                descriptor= TextureDescriptors.compute_LBP_histogram(self.convert_image_grey_scale(cropped_img), radius = lbp_radius)

            #dct coefficients
            elif descriptorType=="DCT":
                block_size = descriptor_config.get("dct_block_size")
                descriptor = TextureDescriptors.compute_DCT_histogram(self.convert_image_grey_scale(cropped_img))

            #save keypoints instead of the descriptors to match them later
            elif descriptorType=="SIFT":
                self.keypoints = KeypointDescriptors.compute_SIFT_descriptor(self.convert_image_grey_scale(cropped_img))
            elif descriptorType=="SURF":
                self.keypoints = KeypointDescriptors.compute_SURF_descriptor(self.convert_image_grey_scale(cropped_img))
            elif descriptorType=="ORB":
                self.keypoints = KeypointDescriptors.compute_ORB_descriptor(self.convert_image_grey_scale(cropped_img))
            elif descriptorType=="DAISY":
                self.keypoints = KeypointDescriptors.compute_DAISY_descriptor(self.convert_image_grey_scale(cropped_img))

            if descriptor is not None:
                descriptor = descriptor * weight
                concatenated_descriptors = np.concatenate([descriptor,concatenated_descriptors])
            
        if len(concatenated_descriptors)>0:
            self.descriptor = concatenated_descriptors

    def compute_histogram(self,BGR_image, descriptor_type:str, histogram_type:str, nbins:int,max_level=None, level = None):
        """Computes the histogram of a given image. The histogram type (grayscale, concatenated histograms,...) can be selected with histogram_type
            histogram_type: if GRAYSCALE is selected it will compute the 1d grayscale histogram. If histogram_type contains "HSV", "BGR", "YCBCR" or "LAB" it will
                compute the concatenated HSV/BGR/YCBCR/LAB histogram
            nbins: # of bins of the resultant histogram
            descriptor_type: type of descriptor (1D histogram, multi resolution histogram, ...)
            histogram_type: colour space of the histogram
        """
        if descriptor_type =="1Dhistogram":
            if histogram_type=="GRAYSCALE":
                histogram = Histograms.compute_histogram_grey_scale(BGR_image, nbins,mask = self.mask)
            #by default, compute 1d concatenated histogram if its not the grayscale one
            else:
                histogram = Histograms.compute_histogram_3channel(BGR_image,nbins, histogram_type, mask=self.mask)
        elif descriptor_type=="mult_res_histogram":
            histogram = Histograms.compute_spatial_pyramid_representation(image=BGR_image, mask=self.mask, color_space=histogram_type, nbins=nbins, max_level=max_level)
        elif descriptor_type=="block_histogram":
            histogram = Histograms.compute_block_based_histogram(image=BGR_image, mask=self.mask, color_space=histogram_type, nbins=nbins, level=level)
    
        #normalise histogram to not take into account the amount of pixels/how big the picture is into the similarity comparison
        norm_histogram = histogram/sum(histogram)
        #cast to float64 just in case
        norm_histogram = np.float64(norm_histogram)
        return norm_histogram

    def remove_background(self, image,method:str):
        """Removes the background of the image and saves it in a path. If computeGT is set to True, it will also compute the precision/recall of the mas compared to the GT
            method: method used to remove the background
        """
        #remove background using otsu
        if method =="OTSU":
            print("Removing background with OTSU")
            mask = BackgroundRemoval.remove_background_otsu(im = image)
        #remove background using colour thresholds
        elif(method =="LAB" or method=="HSV"):
            print("Removing background with ", method)
            mask = BackgroundRemoval.remove_background_color(im = image, colorspace=method)
        elif(method=="MORPHOLOGY"):
            print("Removing background using CONTOURS+MORPHOLOGY")
            mask = BackgroundRemoval.remove_background_morph(img=image)
        elif(method=="CANNY"):
            print("Removing background using CANNY")
            mask = BackgroundRemoval.remove_background_canny(img=image)

        return mask



    def count_paintings(self, max_paintings: int):
        """Counts how many connected components are in the image and returns a list where each position is an Image object with the mask information of the corresponding
        connected component
        
        """
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
        #sort list from left to right
        if len(paintings)>1:
            paintings.sort(key = self.find_first_white_px)
            #sort them from left to right (only case ==2 )
        return paintings
    
    def crop_image_with_mask_bbox(self, img):
        """Crops an image with the shape of the bounding box of its mask
            img: image to crop
        """
        white_pixels = np.array(np.where(self.mask == 255))
        white_pixels = np.sort(white_pixels)
        #get coordinates of the first and last white pixels (useful to set a mask bounding box)
        first_white_pixel = white_pixels[:,0]
        last_white_pixel = white_pixels[:,-1]
        #crop image with np slicing
        img_cropped = img[first_white_pixel[0]:last_white_pixel[0],first_white_pixel[1]:last_white_pixel[1]]
        return img_cropped,first_white_pixel

    def find_first_white_px(self,painting):
        """Finds the x coordinate of the mask of a painting (useful to sort them from left to right)

        """
        white_px_coordinates = np.array(np.where(painting.mask == 255))
        white_px_coordinates = np.sort(white_px_coordinates)
        first_white_pixel_coord = white_px_coordinates[:,0]
        first_white_pixel_xcoord = first_white_pixel_coord[1]
        return first_white_pixel_xcoord