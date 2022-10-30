import numpy as np
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern
import cv2
from scipy.fftpack import fft, dct

class TextureDescriptors:
    @staticmethod
    def compute_LBP_histogram(image: np.ndarray, radius: int = 2):
        """Given an image, computes the lbp histogram
            image: grayscale target image to compute the histogram of
        """
        #resized_img = resize(image, (128*4, 64*4))
        image = resize(image, (128*4, 64*4))
        n_points = radius * 8
        lbp = local_binary_pattern(image, n_points, radius,method= 'default')
        
        
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        hist = hist/sum(hist)
        return hist

    @staticmethod
    def compute_HOG(image: np.ndarray):
        """Given an image, computes the hog histogram
            image: target image to compute the histogram of
        """
        #resize proposed https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f to be able to compare between two images of different sizes
        resized_img = resize(image, (128*4, 64*4))
        hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),multichannel=True)
        return hog_image

    @staticmethod
    
    def compute_DCT_histogram(image: np.ndarray):
        image = np.float32(image) / 255.0
        histograms = []
        block_size = 16
        # number of coefficients
        n = 1
        for i in range(block_size):
            for j in range(block_size):
                im_block = image[int(i * (image.shape[0] / block_size)):int((i + 1) * (image.shape[0] / block_size)), int(j * (image.shape[1] / block_size)):int((j + 1) * (image.shape[1] / block_size))]
                #compute DCT
                dct_im = cv2.dct(im_block)
                #obtain coefficients doing a zigzag scan (most relevanT to least relevant)
                zig_zag_scan = np.concatenate([np.diagonal(dct_im[::-1, :], i)[::(2 * (i % 2) - 1)] for i in range(1 - dct_im.shape[0], dct_im.shape[0])])
                histograms.append(zig_zag_scan[:n])

        histogram = np.concatenate(histograms, axis=0)
        histogram = (histogram - np.min(histogram))/(np.max(histogram) - np.min(histogram)) #normalized
        return histogram