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
        relationships_path = self.dataset_directory + '/relationships.pkl'
        return pd.read_pickle(relationships_path)
    
    def read_images(self, directory: str) -> list:
        images = []
        for file in os.listdir(directory):
            if file.endswith(".jpg"): 
                file_directory = os.path.join(directory, file)
                filename_without_extension = file.split(".")[0]
                filename_id =  int(filename_without_extension.split("_")[-1])
                images.append(Image(file_directory,filename_id))
                
        return images
    
    #Task 2
    def compute_distances(self,image1:Image, image2:Image, distance_string:str) -> float:
        """
        Given two images and the type of distance, it computes the distance between their histograms
            image1,image2: Image objects to analyse their similarities
            distance_string: Contains the label of the distance that will be used to compute it
        """

        #uncomment once proper histograms are in the image
        histogram1 = image1.histogram_grey_scale_image
        histogram2 = image2.histogram_grey_scale_image
       
        ###temporary values for testing purposes
        #img1 = cv2.imread("bbdd_00000.jpg",0)
        #img2 = cv2.imread("bbdd_00004.jpg",0)
        #histogram1 = np.bincount(img2.ravel(), minlength = 256)
        #histogram2 = np.bincount(img1.ravel(), minlength = 256)
        
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

    #Task 3/4 -> F
    def retrieve_top_K_results(self, query_image: Image, K: int, distance_string:str) -> list:
        """Given an image of the query set, retrieve the top K images with the lowest distance speficied by distance_string from the dataset
            query_image: Image to compute the distances against the BBDD
            K: # of most similar images returned 
            distance_string: specifies which distance to use
        """
    
        
        ###temporary values for testing purposes REMOVE AFTER THE SYSTEM GETS JOINED
        img1 = Image("bbdd_00000.jpg",0)
        img2 = Image("bbdd_00002.jpg",1)
        img3 = Image("bbdd_00003.jpg",2)
        img4 = Image("bbdd_00004.jpg",3)

        img1.histogram_grey_scale_image = np.bincount((cv2.cvtColor(img1.RGB_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
        img2.histogram_grey_scale_image =  np.bincount((cv2.cvtColor(img2.RGB_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
        img3.histogram_grey_scale_image =  np.bincount((cv2.cvtColor(img3.RGB_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
        img4.histogram_grey_scale_image = np.bincount((cv2.cvtColor(img4.RGB_image, cv2.COLOR_BGR2GRAY)).ravel(), minlength = 256)
        bbdd_images = [img1, img2, img3, img4]
        query_image = img1
        ####


        distances = []
        ids = []

        #for BBDD_current_image in self.dataset:
        for BBDD_current_image in bbdd_images:        #remove and comment once the system gets joined (proper line is the first one)
            current_distance = self.compute_distances(BBDD_current_image, query_image,distance_string )
            current_id = BBDD_current_image.id
            
            distances.append(current_distance)
            ids.append(current_id)

        list_distance_ids = list(zip(distances, ids))
        list_distance_ids.sort()
        ids_sorted = [distances for ids, distances in list_distance_ids]
        return ids_sorted[:K]

    #Task 4 -> 
    def compute_MAP_at_k(self):
        pass
    

def main():
    dataset_directory = sys.argv[1]
    query_set_directory = sys.argv[2]
    museum = Museum(dataset_directory, query_set_directory)
    museum.retrieve_top_K_results(None,2,"L1")

if __name__ == "__main__":
    main()

