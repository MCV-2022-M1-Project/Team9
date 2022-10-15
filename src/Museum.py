import numpy as np
import os
import pickle
from src.Image import Image
from src.Measures import Measures
from pathlib import Path

class Museum:
    def __init__(self, query_set_directory: str, db_pickle_path:str,gt_flag:str) -> None:
        self.query_set_directory = query_set_directory  #dir of the query set

        #if there's ground truth, store it
        if(gt_flag=='True'):
            self.query_gt = self.read_pickle(self.query_set_directory + '/gt_corresps.pkl')
            if Path(self.query_set_directory +'/text_boxes.pkl').is_file():
                self.text_boxes_gt = self.read_pickle(self.query_set_directory +'/text_boxes.pkl')
        else:
            self.query_gt=[]

        #load dataset from pkl file
        with open(db_pickle_path, 'rb') as f:
            database_data = pickle.load(f)
            self.dataset = database_data[0] #load image objects containing the id and descriptors
            self.relationships = database_data[1] #load DB relationships
            self.config = database_data[2]  #load config info of how the descriptors have been generated
            f.close()
        print("Reading query images")
        self.query_set = self.read_images(self.query_set_directory, self.config)
        
    @staticmethod
    def read_pickle(file_path):
        """
        Reads a pickle file and returns the content of it
            file_path: file_path of the .pkl file
        """
        with open(file_path, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        
        return content  
    @staticmethod
    def read_images( directory: str, museum_config:dict) -> list:        
        """
        Given a directory and the descriptor configuration, it creates an image object with those variables
            directory: Directory where the target images in
            distance_string: Contains the label of the distance that will be used to compute it
        """
        images = []
        for file in os.listdir(directory):
            if file.endswith(".jpg"): 
                file_directory = os.path.join(directory, file)
                filename_without_extension = file.split(".")[0]
                filename_id =  int(filename_without_extension.split("_")[-1])
                images.append(Image(file_directory,filename_id))
                
        return images

    def compute_distances(self,image1: Image, image2: Image, distance_string: str = "L2") -> float:
        """
        Given two images and the type of distance, it computes the distance/similarity between their histograms
            image1,image2: Image objects to analyse their similarities
            distance_string: Contains the label of the distance that will be used to compute it
        """

        histogram1 = image1.descriptor
        histogram2 = image2.descriptor
        

        if distance_string == "L2":
            distance = Measures.compute_euclidean_distance(histogram1, histogram2)
        elif distance_string =="L1":
            distance = Measures.compute_L1_distance(histogram1, histogram2)
        elif distance_string == "X2":
            distance = Measures.compute_x2_distance(histogram1, histogram2)
        elif distance_string == "HIST_INTERSECTION":
            distance = Measures.compute_histogram_intersection(histogram1, histogram2)
        elif distance_string == "HELLINGER_KERNEL":
            distance = Measures.compute_hellinger_kernel(histogram1, histogram2)

        return distance

    def retrieve_top_K_results(self, paintings: Image, K: int, distance_string:str, max_paintings : int) -> list:
        """Given an image of the query set, retrieve the top K images with the lowest distance speficied by distance_string from the dataset
                query_image: Image to compute the distances against the BBDD
                K: # of most similar images returned 
                distance_string: specifies which distance to use
            Output: list with the ids of the K most similar images to query_image
        """

        if max_paintings == 1:
            distances = []
            ids = []
            
            for BBDD_current_image in self.dataset:
                current_distance = self.compute_distances(BBDD_current_image, paintings[0],distance_string )
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
            
            ids_sorted = [ids for distances, ids in list_distance_ids]
            
            return ids_sorted[:K]

        else:
            ids_sorted_list = []
            for query_painting in paintings:
                distances = []
                ids = []
                for BBDD_current_image in self.dataset:
                    current_distance = self.compute_distances(BBDD_current_image, query_painting,distance_string )
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
                
                ids_sorted = [ids for distances, ids in list_distance_ids]
                ids_sorted_list.append(ids_sorted[:K])
            return ids_sorted_list

    def compute_MAP_at_k(self, actual, predicted, k: int = 3):
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
        return np.mean([Measures.apk(a,p,k) for a,p in zip(actual, predicted)])
    

