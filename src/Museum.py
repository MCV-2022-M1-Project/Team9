import numpy as np
import os
import pickle
import cv2
from src.Image import Image
from src.Measures import Measures
from pathlib import Path

class Museum:
    def __init__(self, query_set_directory: str, db_pickle_path:str,gt_flag:str,match_thresh:float) -> None:
        self.query_set_directory = query_set_directory  #dir of the query set
        self.match_thresh = match_thresh
        #if there's ground truth, store it
        if(gt_flag=='True'):
            self.query_gt = self.read_pickle(self.query_set_directory + '/gt_corresps.pkl')
            if Path(self.query_set_directory +'/text_boxes.pkl').is_file():
                self.text_boxes_gt = self.read_pickle(self.query_set_directory +'/text_boxes.pkl')
                box_list = []
                for box_paintings in self.text_boxes_gt:
                    box_painting_list = []
                    for box in box_paintings:
                        if(type(box[0])!=int and type(box[0])!=np.int32):
                                
                            tl = box[0]
                            br = box[2]
                            box_coordinates = [tl[0], tl[1], br[0], br[1]]
                            box_painting_list.append(box_coordinates)
                        else:
                            box_painting_list.append(box)
                    box_list.append(box_painting_list)
                self.text_boxes_gt = box_list

            if Path(self.query_set_directory +'/augmentations.pkl').is_file():
                self.augmentations_gt = self.read_pickle(self.query_set_directory +'/augmentations.pkl')
        else:
            self.query_gt=[]

        #load dataset from pkl file
        with open(db_pickle_path, 'rb') as f:
            database_data = pickle.load(f)
            self.dataset = database_data[0] #load image objects containing the id and descriptors
            self.relationships = database_data[1] #load DB relationships
            self.config = database_data[2]  #load config info of how the descriptors have been generated
            self.dict_artists_paintings = database_data[3]
            f.close()
        print("Reading query images")
        self.query_set, _ ,_= self.read_images(self.query_set_directory, is_query = True)
        
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
    def read_images( directory: str, is_query=False) -> list:        
        """
        Given a directory and the descriptor configuration, it creates an image object with those variables
            directory: Directory where the target images in
            distance_string: Contains the label of the distance that will be used to compute it
        """
        images = []
        dict_artists_paintings = {} #dictionary containing all the artists and their paintings as values
        dict_title_paintings = {} #dictionary containing all the artists and their paintings as values
        artist = None
        title = None
        for file in os.listdir(directory):
            if file.endswith(".jpg"): 
                file_directory = os.path.join(directory, file)
                filename_without_extension = file.split(".")[0]
                filename_path_without_extension = file_directory.rsplit('.',1)[0]
                filename_id =  int(filename_without_extension.split("_")[-1])

                if Path(str(filename_path_without_extension+'.txt')).is_file() and not is_query:
                #read .txt file containing title/artist info
                    with open(str(filename_path_without_extension+'.txt'), encoding = "ISO-8859-1") as f:
                        lines = f.readlines()
                        lines = lines[0]
                        f.close()

                    #replace with unknown if it doesnt have any information in the txt file
                    if lines == "\n":
                        lines = "('Unknown', 'Unknown')\n"
                        
                    artist = lines.split("'")[1]
                    title = lines.split("'")[3]

                    #store it so that the key is the artist and each value is [title of the painting, image_id]
                    if artist in dict_artists_paintings.keys():
                        dict_artists_paintings[artist]+=[ filename_id]
                    else:
                        dict_artists_paintings[artist] = [filename_id]
                    if title in dict_title_paintings.keys():
                        dict_title_paintings[title]+=[ filename_id]
                    else:
                        dict_title_paintings[title] = [filename_id]
                images.append(Image(file_directory,filename_id, artist = artist, title = title))
                
        return images, dict_artists_paintings,dict_title_paintings

    def compute_distances(self,image1: Image, image2: Image, distance_string: str = "L2") -> float:
        """
        Given two images and the type of distance, it computes the distance/similarity between their histograms
            image1,image2: Image objects to analyse their similarities
            distance_string: Contains the label of the distance that will be used to compute it
        """

        feature_vector1 = image1.descriptor
        feature_vector2 = image2.descriptor

        if distance_string == "L2":
            distance = Measures.compute_euclidean_distance(feature_vector1, feature_vector2)
        elif distance_string =="L1":
            distance = Measures.compute_L1_distance(feature_vector1, feature_vector2)
        elif distance_string == "X2":
            distance = Measures.compute_x2_distance(feature_vector1, feature_vector2)
        elif distance_string == "HIST_INTERSECTION":
            distance = Measures.compute_histogram_intersection(feature_vector1, feature_vector2)
        elif distance_string == "HELLINGER_KERNEL":
            distance = Measures.compute_hellinger_kernel(feature_vector1, feature_vector2)

        return distance

    def compute_matches(self, image1, image2):
        # load BF matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(image1.keypoints,image2.keypoints,k=2)
        # Apply ratio test
        thresh_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                thresh_matches.append([m.distance])
        return thresh_matches
    def compute_matches_FLANN(self, db_image, query_image, distance_string):

        des1 = db_image.keypoints
        des2 = query_image.keypoints
        
        if(des1 is not None and len(des1)>2 and des2 is not None and len(des2)>2):
            
            des1 = np.float32(des1)
            des2 = np.float32(des2)

            #ORB params
            if distance_string == "HAMMING":
                FLANN_INDEX_KDTREE = 0
                ratio_lowe = 0.75
                index_params= dict(algorithm = FLANN_INDEX_KDTREE, table_number = 6, key_size = 12,  multi_probe_level = 1)
            else:
                # FLANN parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)
            match_amount = 0
            ratio_lowe = 0.75
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < ratio_lowe*n.distance:
                    match_amount = match_amount +1
    
            return match_amount
        else:
            return 0
    def retrieve_top_K_results(self, paintings: Image, K: int, distance_string:str, max_paintings : int, text_string_list:str = None) -> list:
        """Given an image of the query set, retrieve the top K images with the lowest distance speficied by distance_string from the dataset
                query_image: Image to compute the distances against the BBDD
                K: # of most similar images returned 
                distance_string: specifies which distance to use
            Output: list with the ids of the K most similar images to query_image
        """

        amount_matches = 10
        match_treshold = 9
        ids_sorted_list = []
        for idx_painting, query_painting in enumerate(paintings):
            distances = []
            matches = []
            ids = []
            for BBDD_current_image in self.dataset:

                if hasattr(BBDD_current_image, 'descriptor'):
                    current_distance = self.compute_distances(BBDD_current_image, query_painting,distance_string )
                    distances.append(current_distance)
                    
                    current_id = BBDD_current_image.id
                    #discard if there's not enough matches
                    #if current_distance>4500:
                    #    current_id = -1
                    #else:
                    #    current_id = BBDD_current_image.id
                    ids.append(current_id)

                elif hasattr(BBDD_current_image, 'keypoints'):
                    amount_matches = self.compute_matches_FLANN(BBDD_current_image, query_painting,distance_string)
                    matches.append(amount_matches)
                    current_id = BBDD_current_image.id
                    ids.append(current_id)
            
            if hasattr(BBDD_current_image, 'descriptor'):
                list_distance_ids = list(zip(distances, ids))
                #sort ascending to descending if the highest score means the bigger the similarity
                if(distance_string=="HIST_INTERSECTION" or distance_string =="HELLINGER_KERNEL"):
                    list_distance_ids.sort(reverse = True)
                else:
                #sort descending to ascending if the smallest distance  means the bigger the similarity
                    list_distance_ids.sort()
                ids_sorted = [ids for distances, ids in list_distance_ids]
                if list_distance_ids[0][0]>self.match_thresh:
                    #shift list to the right 1 position and add unknown flag if matches are under a threshold
                    _ = ids_sorted.pop()
                    ids_sorted.insert(0, -1)
            else: 
                list_distance_ids = list(zip(matches, ids))
                #sort ascending to descending if the highest score means the bigger the similarity
                list_distance_ids.sort(reverse = True)
                ids_sorted = [ids for distances, ids in list_distance_ids]

                # DISTANCE ", ids_sorted)
                #print("LIST DISTANCE ", list_distance_ids)
                print(len(list_distance_ids))
                print(list_distance_ids)
                print("MAX MATCHES: ", list_distance_ids[0][0])
                if list_distance_ids[0][0]<self.match_thresh:
                    #shift list to the right 1 position and add unknown flag if matches are under a threshold
                    _ = ids_sorted.pop()
                    ids_sorted.insert(0, -1)

                #print("IDS DISTANCE AFTER", ids_sorted)
            #if improving the results with text is enabled
            if text_string_list is not None:
                if (K<10):
                    ids_sorted = ids_sorted[:10] #get top 10 results
                else:
                    ids_sorted = ids_sorted[:K]

                curr_artist_prediction = text_string_list[idx_painting]
                #ignore if prediction is empty
                if curr_artist_prediction != "":
                    import textdistance
                    #if theres an artist with that key in the db dictionary, use it
                    if curr_artist_prediction in self.dict_artists_paintings.keys():
                        possible_paintings_db = self.dict_artists_paintings[curr_artist_prediction]
                    else:
                        min_dist = -1
                        closest_artist_key = None
                        dist_text = "LEV"
                        for artist in self.dict_artists_paintings:
                            lev_dist = textdistance.levenshtein.distance(artist,curr_artist_prediction)
                            if(dist_text == "HAMMING"):
                                # function call
                                lev_dist=textdistance.hamming(artist, curr_artist_prediction)
                            elif(dist_text == "JACCARD"):
                                lev_dist = textdistance.jaccard.distance(artist,curr_artist_prediction)
                            if lev_dist<min_dist or min_dist == -1:
                                min_dist=lev_dist
                                closest_artist_key = artist

                        print("CLOSEST PREDICTION IN DB", closest_artist_key, "PREDICTION ", curr_artist_prediction)
                        possible_paintings_db = self.dict_artists_paintings[closest_artist_key]
                    
                    text_only = False
                    if text_only:
                        ids_sorted = possible_paintings_db
                        
                        ids_sorted = np.pad(ids_sorted, (0, 20), 'constant', constant_values=(4, 0))
                        
                    else:
                        found = False
                        for k_idx, predicted_painting_id in enumerate(ids_sorted):
                            if predicted_painting_id in possible_paintings_db:
                                found = True
                                #shift list to the right 1 position and add text prediction in the first one
                                _ = ids_sorted.pop()
                                ids_sorted.insert(0, predicted_painting_id)
                                break
                        if not found:
                            _ = ids_sorted.pop()
                            ids_sorted.insert(0, possible_paintings_db[0])
            ids_sorted_list.append(ids_sorted[:K])
        print("RES ", ids_sorted_list)
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
    

