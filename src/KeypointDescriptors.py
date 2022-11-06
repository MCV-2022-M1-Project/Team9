import numpy as np
import cv2
class KeypointDescriptors:
    @staticmethod
    def compute_SIFT_descriptor(image: np.ndarray, keypoints=None):
        """Given an image, computes the lbp histogram
            image: grayscale target image to compute the histogram of
        """
        #to decrease matches, increase contrastThreshold and decrease edgeThreshold
        sift = cv2.SIFT_create(nfeatures = 500,	contrastThreshold = 0.010, edgeThreshold = 15, sigma = 1.6)
        # find keypoints
        kp, des = sift.detectAndCompute(image,keypoints)
        if des is not None:
            print("LEN DES", len(des))
        return des

    @staticmethod
    def compute_SURF_descriptor(image : np.ndarray, keypoints=None):
        surf = cv2.xfeatures2d.SURF_create(400,upright = False, nOctaves = 5)
        #surf.setExtended(True)
        kp, des = surf.detectAndCompute(image,keypoints)
        if des is not None:
            print("LEN DES", len(des))
            if len(des)>1500:
                des = des[:1500]
            
            print("LEN DES", len(des))
        return des
        
    @staticmethod
    def compute_ORB_descriptor(image: np.ndarray, keypoints=None):
        orb = cv2.ORB_create(nfeatures=1000, scaleFactor = 1.1)
        kp = orb.detect(image,keypoints)
        kp, des = orb.compute(image, kp)
        
 
        return des

    @staticmethod
    def compute_DAISY_descriptor(image: np.ndarray, keypoints = None):
        daisy = cv2.xfeatures2d.DAISY.create()
        kp, des = daisy.detectAndCompute(image, keypoints)
        return des
    
    @staticmethod
    def compute_harris_laplace_detector(image: np.ndarray):
        harris_laplace = cv2.xfeatures2d.HarrisLaplaceFeatureDetector.create()
        kp, des = harris_laplace.detectAndCompute(image)
        return des