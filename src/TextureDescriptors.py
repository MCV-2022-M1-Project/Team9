import numpy as np
from skimage.transform import resize
from skimage.feature import hog, local_binary_pattern

class TextureDescriptors:
    @staticmethod
    def compute_LBP_histogram(image: np.ndarray, radius: int = 2):
        """Given an image, computes the lbp histogram
            image: grayscale target image to compute the histogram of
        """
        n_points = radius * 8
        lbp = local_binary_pattern(image, n_points, radius, 'uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
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