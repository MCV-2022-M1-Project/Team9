import numpy as np
from evaluation.evaluation_funcs import performance_accumulation_pixel, performance_evaluation_pixel

class Measures:
  
    @staticmethod
    def compute_euclidean_distance(histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the euclidean distance between them 
                histogram1, histogram2: histograms of the images to compare the similarities of
            Output: euclidean distance
        """
        sum_sq = np.sum(np.square(histogram2 - histogram1))
        euclidean_distance = np.sqrt(sum_sq)
        return euclidean_distance
    
    
    @staticmethod
    def compute_L1_distance(histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the L1 distance between them. 
                histogram1, histogram2: histograms of the images to compare the similarities of
            Output: L1 distance
        """
        l1_distance = sum(abs(histogram2-histogram1))
        return l1_distance
    
    @staticmethod
    def compute_x2_distance(histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the chi-squared distance between them. 
                histogram1, histogram2: histograms of the images to compare the similarities of
            Output: chi-squared distance
        """
        #if both bins have no pixels with that level (array pos==0), ignore them as the chi distance is 0
        np.seterr(divide='ignore', invalid='ignore')
        before_squared = np.square((histogram2 - histogram1))/(histogram1+histogram2)
        before_squared[np.isnan(before_squared)] = 0    #fix nan values of this case (0/0 division)
        x2_distance = np.sum(before_squared)
        return x2_distance
    
    @staticmethod
    def compute_histogram_intersection(histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the histogram intersection
            histogram1, histogram2: histograms of the images to compare the similarities of
        Output: histogram intersection
            """
        histogram_intersection = np.sum(np.minimum(histogram1, histogram2))
        return histogram_intersection
    
    @staticmethod
    def compute_hellinger_kernel(histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes hellinger kernel distance
            histogram1, histogram2: histograms of the images to compare the similarities of
        Output: hellinger kernel distance
        """
        hellinger_kernel =  np.sum(np.sqrt(np.multiply(histogram1,histogram2)))
        return hellinger_kernel
    
    @staticmethod
    def apk(actual, predicted, k: int):
        """
        Code belongs to the benhammer repo https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        if len(predicted)>k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i,p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)
    
    @staticmethod
    def compute_TP_FP_FN_TN(mask, mask_gt) -> float:
        """Given two masks, it computes the true positives, false positives, false negatives and true negatives of their values
            mask: predicted mask
            mask_gt: ground truth mask
        """

        if mask.shape!=mask_gt.shape:
            #mask is rotated
            mask = np.reshape(mask, (mask.shape[1], mask.shape[0]))
        #obtain TP,FP,FN,TN values
        [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(mask,mask_gt)
  

        return pixelTP,pixelFP,pixelFN, pixelTN

    @staticmethod
    def compute_precision_recall_F1(pixelTP, pixelFP, pixelFN, pixelTN):
        """Given the amount of TP, FP, FN and TN, it computes the precision, recall and Fscore of them
            pixelTP: # of true positives
            pixelFP: # of false positives
            pixelTP: # of false negatives
            pixelTN: # of true negatives
        """
        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_recall] = performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
        #if condition to avoid 0 division if both metrics are 0
        if(pixel_precision==0 or pixel_recall==0):
            pixel_F1_score = 0
        else:
            #compute F1 score
            pixel_F1_score = 2*float(pixel_precision) *float(pixel_recall)/ float(pixel_recall+pixel_precision)
        
        return pixel_precision,pixel_recall,pixel_F1_score