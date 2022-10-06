"""
Generate similarity results given a query folder
Usage:
  Museum.py <inputDir> <queryDir> [--distance=<dist>] [--K=<k>] [--picklePath=<ppath>] [--generateDB=<gendb>]
  Museum.py -h | --help
  -
  <inputDir>               Directory with database data 
  <queryDir>               Directory with query data
Options:
  --distance=<dist>        Distance to compute image similarity (L1, L2, X2, HIST_INTERSECTION, HELLINGER_KERNEL) [default: L1]
  --K=<k>                  Number of similar results to output [default: 3]
  --picklePath=<ppath>     Filename/path to save the pkl results file [default: result.pkl]
  --generateDB=<gendb>     Regenerate database (True/False) [default: True]
"""

import pandas as pd 
import sys
import numpy as np
import os
import cv2
import pickle
from Image import Image
from docopt import docopt


class Museum:
    def __init__(self, dataset_directory: str, query_set_directory: str, generateDB:bool) -> None:
        self.dataset_directory = dataset_directory
        self.query_set_directory = query_set_directory
        self.relationships = self.read_relationships()
        self.query_gt = self.read_query_gt()
        if generateDB == "True":
            print("Generating database pkl file")
            self.dataset = self.read_images(self.dataset_directory)
        else:
            with open("database.pkl", 'rb') as f:
                self.dataset = pickle.load(f)
                f.close()

        print("Computing query image descriptors")
        self.query_set = self.read_images(self.query_set_directory)
        
    def read_relationships(self):
        print(self.dataset_directory)
        relationships_path = self.dataset_directory + '/relationships.pkl'
        return pd.read_pickle(relationships_path)

    def read_query_gt(self):
        relationships_path = self.query_set_directory + '/gt_corresps.pkl'
        return pd.read_pickle(relationships_path)
    
    def read_images(self, directory: str) -> list:
        images = []
        for file in os.listdir(directory):
            if file.endswith(".jpg"): 
                file_directory = os.path.join(directory, file)
                filename_without_extension = file.split(".")[0]
                filename_id =  int(filename_without_extension.split("_")[-1])
                print(file)
                images.append(Image(file_directory,filename_id))
                
                
        return images
    
    def compute_distances(self,image1:Image, image2:Image, distance_string:str) -> float:
        """
        Given two images and the type of distance, it computes the distance/similarity between their histograms
            image1,image2: Image objects to analyse their similarities
            distance_string: Contains the label of the distance that will be used to compute it
        """

        histogram1 = image1.histogram
        histogram2 = image2.histogram
        
        #cast to float64 just in case
        histogram1 = np.float64(histogram1)
        histogram2 = np.float64(histogram2)

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
        """Given two histograms, computes the euclidean distance between them 
                histogram1, histogram2: histograms of the images to compare the similarities of
            Output: euclidean distance
        """
        sum_sq = np.sum(np.square(histogram2 - histogram1))
        euclidean_distance = np.sqrt(sum_sq)
        return euclidean_distance
    
    def compute_L1_distance(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the L1 distance between them. 
                histogram1, histogram2: histograms of the images to compare the similarities of
            Output: L1 distance
        """
        l1_distance = sum(abs(histogram2-histogram1))
        return l1_distance
    
    def compute_x2_distance(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the chi-squared distance between them. 
                histogram1, histogram2: histograms of the images to compare the similarities of
            Output: chi-squared distance
        """
        #if both bins have no pixels with that level (array pos==0), ignore them as the chi distance is 0
        np.seterr(divide='ignore', invalid='ignore')
        before_squared = (histogram2 - histogram1)/(histogram1+histogram2)
        before_squared[np.isnan(before_squared)] = 0    #fix nan values of this case (0/0 division)
        x2_distance = np.sum(np.square(before_squared))
        return x2_distance
    
    def compute_histogram_intersection(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes the histogram intersection
            histogram1, histogram2: histograms of the images to compare the similarities of
        Output: histogram intersection
            """
        histogram_intersection = np.sum(np.minimum(histogram1, histogram2))
        return histogram_intersection
    
    def compute_hellinger_kernel(self, histogram1 : np.ndarray, histogram2 : np.ndarray) -> float:
        """Given two histograms, computes hellinger kernel distance
            histogram1, histogram2: histograms of the images to compare the similarities of
        Output: hellinger kernel distance
        """
        hellinger_kernel =  np.sum(np.sqrt(np.multiply(histogram1,histogram2)))
        return hellinger_kernel

    #Task 3/4 -> F
    def retrieve_top_K_results(self, query_image: Image, K: int, distance_string:str) -> list:
        """Given an image of the query set, retrieve the top K images with the lowest distance speficied by distance_string from the dataset
                query_image: Image to compute the distances against the BBDD
                K: # of most similar images returned 
                distance_string: specifies which distance to use
            Output: list with the ids of the K most similar images to query_image
        """

        distances = []
        ids = []
        for BBDD_current_image in self.dataset:
        #for BBDD_current_image in bbdd_images:        #remove and comment once the system gets joined (proper line is the first one)
            current_distance = self.compute_distances(BBDD_current_image, query_image,distance_string )
            current_id = BBDD_current_image.id
            
            distances.append(current_distance)
            ids.append(current_id)

        list_distance_ids = list(zip(distances, ids))

        #sort ascending to descending if the highest score means the bigger the similarity
        if(distance_string=="HIST_INTERSECTION" or distance_string =="HELLINGER_KERNEL"):
            list_distance_ids.sort(reverse = True)
        else:
        #sort descending to ascending if the smallest distance  means the bigger the similarity
            list_distance_ids.sort()
        ids_sorted = [distances for ids, distances in list_distance_ids]
        return ids_sorted[:K]

    #Task 4 -> 
    def apk(self, actual, predicted, k):
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
    def compute_MAP_at_k(self, actual, predicted, k):
        """
        Code belongs to the benhammer repo https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of lists of items.
        actual : list
                A list of lists of elements that are to be predicted 
                (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        score : double
                The mean average precision at k over the input lists
        """
        return np.mean([self.apk(a,p,k) for a,p in zip(actual, predicted)])
    
    
def main():
    # read arguments
    args = docopt(__doc__)
    dataset_directory = args['<inputDir>']
    query_set_directory = args['<queryDir>']
    distance_arg = args['--distance']
    K = int(args['--K'])
    pickle_path = args['--picklePath']
    generateDB = args['--generateDB']
    museum = Museum(dataset_directory, query_set_directory,generateDB)
    if generateDB:
        image_list = []
        ##STORE DATABASE INTO FILE
        for BBDD_current_image in museum.dataset:
            image_list.append(BBDD_current_image)
        
        #save list of lists into pkl file
        with open("database.pkl", 'wb') as f:
            pickle.dump(image_list, f)
            

    ##GENERATE QUERY RESULTS
    predicted_top_K_results = []    #list containing in each position a K-element list of the predictions for that query
    #for each one of the queries
    
    print("Computing distances with DB images...")
    for current_query in museum.query_set:
        print("Query: ", current_query.file_directory)
        predicted_top_K_results.append(museum.retrieve_top_K_results(current_query,K,distance_arg))
        
    
    #print("querygt",museum.query_gt)
    #print("predictions",predicted_top_K_results)
    mapk_score = museum.compute_MAP_at_k(museum.query_gt, predicted_top_K_results, K)
    
    print("Using distance", distance_arg)
    #print("TOP ",K, " RESULTS: ",predicted_top_K_results)
    print("MAPK score: ",mapk_score)

    #save list of lists into pkl file
    with open(pickle_path, 'wb') as f:
        pickle.dump(predicted_top_K_results, f)
if __name__ == "__main__":
    main()

