import numpy as np
import cv2

class Histograms:   

    @staticmethod
    def compute_histogram_grey_scale(image,nbins:int, mask):
        """
        Computes the histogram greyscale histogram. If the image has a mask, it will not take into account the pixels marked as background
            image: Image to obtain the histogram from
            nbins: #of bins of the histogram

        """
        #convert to gray
        grey_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if len(mask)>0:
            
            hist, bin_edges = np.histogram(grey_scale_image, bins=nbins, weights = mask)
        else:
            hist, bin_edges = np.histogram(grey_scale_image, bins=nbins)
        return hist

    @staticmethod
    def compute_histogram_3channel(BGR_image,nbins:int, colourSpace:str, mask):
        """Computes the concatenated 3 channel histogram of an image, aka an array containing sequentially all of the 1D histograms of each channel
            BGR_image: Image to compute the histogram of
            nbins: # of bins of the resultant histogram
            colourSpace: colour space where the histogram will be computed in

        """
        if colourSpace=="BGR":
            image = BGR_image
        elif colourSpace=="HSV":
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
        elif colourSpace=="YCRCB":
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2YCrCb)
        elif colourSpace=="LAB":
            image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2Lab)
        
        if len(mask)>0:
            chan1_hist, bin_edges = np.histogram(image[:,:,0], bins=nbins, weights = mask/255)
            chan2_hist, bin_edges = np.histogram(image[:,:,1], bins=nbins, weights = mask/255)
            chan3_hist, bin_edges = np.histogram(image[:,:,2], bins=nbins, weights = mask/255)
        
        else:
            chan1_hist, bin_edges = np.histogram(image[:,:,0], bins=nbins)
            chan2_hist, bin_edges = np.histogram(image[:,:,1], bins=nbins)
            chan3_hist, bin_edges = np.histogram(image[:,:,2], bins=nbins)
        hist = np.concatenate([chan1_hist, chan2_hist,chan3_hist])
        return hist


    @staticmethod
    def compute_spatial_pyramid_representation(image: np.ndarray, mask: np.ndarray, color_space:str, nbins: int, max_level: int = 2) -> np.ndarray:
        """Recursively calculates block based histogram based on its level, and then concatenates them
           
            color_space: if GRAYSCALE is selected it will compute the 1d grayscale histogram. If color_space contains "HSV", "BGR", "YCBCR" or "LAB" it will
                compute the concatenated HSV/BGR/YCBCR/LAB histogram
            nbins: # of bins of the resultant histogram
            level: defines in how many blocks will be splitted the image. For example, setting level 4 will split the image in 16 blocks
        """
        histogram_level_list = []
        
        histogram_level_list.append(Histograms.compute_histogram_3channel(image, mask = mask, colourSpace = color_space, nbins = nbins))
        
        sequence = 2
        while sequence <= max_level:
            histogram_level_list.append(Histograms.compute_block_based_histogram(image, mask, color_space, nbins, sequence))
            sequence *=2
            
        return np.concatenate(histogram_level_list, axis = 0)
    
    @staticmethod
    def compute_block_based_histogram(image: np.ndarray, mask: np.ndarray, color_space:str, nbins: int,  level: int = 4) -> np.ndarray:
        """Divides image in non-overlapping blocks, first splits image vertically, and then horizontally. For each block, computes its histogram, and then concatenates them.
            color_space: if GRAYSCALE is selected it will compute the 1d grayscale histogram. If color_space contains "HSV", "BGR", "YCBCR" or "LAB" it will
                compute the concatenated HSV/BGR/YCBCR/LAB histogram
            nbins: # of bins of the resultant histogram
            level: defines in how many blocks will be splitted the image. For example, setting level 4 will split the image in 16 blocks
        """
        
        image_blocks = []
        histogram_block_list = []
        #Split image vertically
        image_vertically_splitted = np.array_split(image, level, axis = 0)
        
        #Split each block of the image horizontally
        for i in image_vertically_splitted:
            image_blocks.extend(np.array_split(i, level,axis=1))
        
        
        
        if len(mask) > 0:
            #Split also mask
            mask_blocks = []
            mask_vertically_splitted = np.array_split(mask, level, axis = 0)
            
            for i in mask_vertically_splitted:
                mask_blocks.extend(np.array_split(i, level, axis=1))
            
            for image_block, mask_block in zip(image_blocks, mask_blocks):
                histogram_block_list.append(Histograms.compute_histogram_3channel(image_block, mask = mask_block, colourSpace = color_space, nbins = nbins))
        else:
            #Compute each histogram's block image without mask
            for image_block in image_blocks:
                histogram_block_list.append(Histograms.compute_histogram_3channel(image_block, mask = mask, colourSpace = color_space, nbins = nbins))        
        
        #Concatenate histograms
        return np.concatenate(histogram_block_list, axis=0)