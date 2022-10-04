import pandas as pd 
import sys
import numpy as np
import os
import cv2
from Image import Image


class Museum:
    def __init__(self, dataset_directory: str, query_set_directory: str) -> None:
        self.dataset_directory = dataset_directory
        self.query_set_directory = query_set_directory
        self.relationships = self.read_relationships()
        #self.dataset = self.read_images(self.dataset_directory)
        #self.query_set = self.read_images(self.query_set_directory)
        
    def read_relationships(self):
        print(self.dataset_directory)
        return
        relationships_path = self.dataset_directory + '/relationships.pkl'
        return pd.read_pickle(relationships_path)
    
    def read_images(self, directory: str) -> list:
        images = []
        for file in os.listdir(directory):
            if file.endswith(".jpg"): 
                file_directory = os.path.join(directory, file)
                images.append(Image(file_directory))
                
        return images
    
    #Task 2
    def compute_distances(self,image1:Image, image2:Image, distance_string:str) -> float:
        """
        Given two images and the type of distance, it computes the distance between their histograms
            image1,image2: Image objects to be measured
            distance_string: Contains the label of the distance that will be used to compute it
        """

        #uncomment once proper histograms are in the images
        #histogram1 = image1.histogram_grey_scale_image
        #histogram2 = image2.histogram_grey_scale_image
       
        ###temporary values for testing purposes
        img1 = cv2.imread("bbdd_00000.jpg",0)
        img2 = cv2.imread("bbdd_00004.jpg",0)
        histogram1 = np.bincount(img2.ravel(), minlength = 256)
        histogram2 = np.bincount(img1.ravel(), minlength = 256)
        
        if distance_string == "L2":
            distance = self.compute_euclidean_distance(histogram1, histogram2)
        elif distance_string =="L1":
            distance = self.compute_L1_distance(histogram1, histogram2)
        elif distance_string == "X2":
            distance = self.compute_x2_distance(histogram1, histogram2)
        elif distance_string == "HIST_INTERSECTION":
            distance = self.compute_histogram_intersection(histogram1, histogram2)
        elif distance_string == "HELLINGER_KERNEL":
            distance = self.compute_hellinger_kernel(histogram1, histogram2)

        print(distance)
        return distance
    def compute_euclidean_distance(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        sum_sq = np.sum(np.square(histogram2 - histogram1))
        euclidean_distance = np.sqrt(sum_sq)
        return euclidean_distance
    
    def compute_L1_distance(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        l1_distance = sum(abs(histogram2-histogram1))
        return l1_distance
    
    def compute_x2_distance(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        x2_distance = np.sum(np.square(histogram2 - histogram1)/(histogram1+histogram2))
       
        return x2_distance
    
    def compute_histogram_intersection(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        histogram_intersection = np.sum(np.minimum(histogram1, histogram2))
        return histogram_intersection
    
    def compute_hellinger_kernel(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        hellinger_kernel =  np.sum(np.sqrt(np.multiply(histogram1,histogram2)))
        return hellinger_kernel

    
    #Task 3/4 -> For an image of the query set, retrieve the top K images with the lowest distance or the highest score from the dataset
    def retrieve_top_K_results(self, image: Image, K: int) -> list:
        pass
    
    #Task 4 -> 
    def compute_MAP_at_k(self):
        pass
    

def main():
    dataset_directory = sys.argv[1]
    query_set_directory = sys.argv[2]
    museum = Museum(dataset_directory, query_set_directory)
    museum.compute_distances(None,None)

if __name__ == "__main__":
    main()

