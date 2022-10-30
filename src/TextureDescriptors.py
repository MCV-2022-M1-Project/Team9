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
    def compute_DCT_histogram(image: np.ndarray, block_size:int=32):
        
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        DCT_histogram = []
        for i in range(3):
            image = ycbcr_image[:,:,i]

            image = resize(image, (128*4, 128*4))
            #add padding to image to convert it to square
            """left_padding = 0
            right_padding = 0
            top_padding = 0
            bottom_padding = 0
            
            if image.shape[0] > image.shape[1]:
                left_padding = int((image.shape[0] - image.shape[1]) / 2)
                right_padding = int((image.shape[0] - image.shape[1]) / 2)
            elif image.shape[1] > image.shape[0]:
                top_padding = int((image.shape[1] - image.shape[0]) / 2)
                bottom_padding = int((image.shape[1] - image.shape[0]) / 2)
            
            #corner case when difference between axis 0 and axis 1 is odd
            if (image.shape[0] + top_padding + bottom_padding) < image.shape[1]:
                top_padding += 1
            elif (image.shape[1] + left_padding + right_padding) < image.shape[0]:
                left_padding += 1
            
            #make the image even to have all the blocks with the same size
            if (image.shape[0] + top_padding + bottom_padding) % 2 == 1:
                bottom_padding += 1
                right_padding += 1
            
            #add padding to the image
            image_square = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT)
            """
            image_square = image
            
            method = None
            
            ncoefs = 512
            method = 'ortho'
            block_image_dct2d = dct(dct(image_square.T, norm=method).T, norm=method, n = ncoefs)

            block_image_dct2d_zigzag = np.concatenate([np.diagonal(block_image_dct2d[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-block_image_dct2d.shape[0], block_image_dct2d.shape[0])])
            #ncoefs = block_image_dct2d_zigzag[:n]
             
            DCT_histogram.extend(block_image_dct2d_zigzag)     
        return DCT_histogram
        #divide it in 8x8 blocks (https://stackoverflow.com/questions/44615166/how-do-you-divide-an-image-array-into-blocks)
        #8x8 suggested here (https://towardsdatascience.com/revisiting-dct-domain-deep-learning-51458fe2e6e4)
        im_h, im_w = image_square.shape[:2]
        bl_h, bl_w = block_size, block_size
        
        for row in np.arange(im_h - bl_h + 1, step=bl_h):
            for col in np.arange(im_w - bl_w + 1, step=bl_w):
                #block of 8x8
                block_image = image_square[row:row+bl_h, col:col+bl_w] 
                #compute dct2d
                #n = block_image_dct2d.shape[0]  # # of coefficients per block
                n = 32
                block_image_dct2d = dct(dct(block_image.T, norm='ortho').T, norm='ortho', n = n)
                #zig zag scan (https://stackoverflow.com/questions/50445847/how-to-zigzag-order-and-concatenate-the-value-every-line-using-python)

                block_image_dct2d_zigzag = np.concatenate([np.diagonal(block_image_dct2d[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-block_image_dct2d.shape[0], block_image_dct2d.shape[0])])
                coefs = block_image_dct2d_zigzag[:n]
                
                #sum_coefs=sum(abs(np.array(coefs)))
                #coefs = coefs/sum_coefs
                DCT_histogram.extend(coefs)
            
        return DCT_histogram    