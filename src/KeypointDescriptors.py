import numpy as np
import cv2
class KeypointDescriptors:
    @staticmethod
    def compute_SIFT_descriptor(image: np.ndarray, keypoints=None, nfeat = 500, mask = []):
        """Given an image, computes the SIFT descriptors
            image: grayscale target image to compute the histogram of
            nfeat: max amount of descriptors to return
        """
        #to decrease matches, increase contrastThreshold and decrease edgeThreshold
        sift = cv2.SIFT_create(nfeatures = nfeat,	contrastThreshold = 0.010, edgeThreshold = 15, sigma = 1.6)
        # find keypoints
        #kp, des = sift.detectAndCompute(image,keypoints)
        if len(mask)>0:
            kp,des = sift.detectAndCompute(image, mask = mask)
        else:
            kp = sift.detect(image,keypoints)
            kp, des = sift.compute(image, kp)
        return kp,des

    @staticmethod
    def compute_SURF_descriptor(image : np.ndarray, keypoints=None, nfeat = 1500, n_octaves = 5, mask = []):
        """Given an image, computes the SURF descriptors
            image: grayscale target image to compute the histogram of
            nfeat: max amount of descriptors to return
        """
        surf = cv2.xfeatures2d.SURF_create(400,upright = False, nOctaves = n_octaves)
        #kp, des = surf.detectAndCompute(image,keypoints)
        
        if len(mask)>0:
            kp,des = surf.detectAndCompute(image, mask = mask)
                
        else:
            kp = surf.detect(image,keypoints)
            kp, des = surf.compute(image, kp)

        if des is not None:
            if len(des)>nfeat:
                des = des[:nfeat]
            
        return kp, des
        
    @staticmethod
    def compute_ORB_descriptor(image: np.ndarray, keypoints=None, nfeat = 1000, nbins = 32, mask = []):
        """Given an image, computes the SURF descriptors
            image: grayscale target image to compute the histogram of
            nfeat: max amount of descriptors to return
            mask: mask of the area of interest (optional)
        """
        orb = cv2.ORB_create(nfeatures=nfeat, scaleFactor = 1.2, nlevels=nbins)
        
        if len(mask)>0:
            kp,des = orb.detectAndCompute(image, mask = mask)
                
        else:
            kp = orb.detect(image,keypoints)
            kp, des = orb.compute(image, kp)

        return kp,des

    @staticmethod
    def compute_DAISY_descriptor(image: np.ndarray, keypoints = None):
        sift = cv2.SIFT_create(nfeatures = 1000,	contrastThreshold = 0.012, edgeThreshold = 10, sigma = 1.6)
        daisy = cv2.xfeatures2d.DAISY.create()
        kp = sift.detect(image, None)
        kp, des = daisy.compute(image, kp)
        return des
    
    @staticmethod
    def compute_harris_laplace_detector(image: np.ndarray):
        harris_laplace = cv2.xfeatures2d.HarrisLaplaceFeatureDetector.create()
        kp, des = harris_laplace.compute(image, None)
        return kp