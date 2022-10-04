import pandas as pd 
import sys
import os
from Image import Image


class Museum:
    def __init__(self, dataset_directory: str, query_set_directory: str) -> None:
        self.dataset_directory = dataset_directory
        self.query_set_directory = query_set_directory
        self.relationships = self.read_relationships()
        self.dataset = self.read_images(self.dataset_directory)
        self.query_set = self.read_images(self.query_set_directory)
        
    def read_relationships(self):
        print(self.dataset_directory)
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
    def compute_euclidean_distance(self, image1 : Image, image2: Image) -> float:
        pass
    
    def compute_L1_distance(self, image1 : Image, image2: Image) -> int:
        pass
    
    def compute_x2_distance(self, image1 : Image, image2: Image) -> float:
        pass
    
    def compute_histogram_intersection(self, image1 : Image, image2: Image) -> int:
        pass
    
    def compute_hellinger_kernel(self, image1 : Image, image2: Image) -> float:
        pass
    
    #Task 3/4 -> For an image of the query set, retrieve the top K images with the lowest distance or the highest score from the dataset
    def retrieve_top_K_results(self, image: Image) -> list:
        pass
    
    #Task 4 -> 
    def compute_MAP_at_k(self):
        pass
    

def main():
    dataset_directory = sys.argv[1]
    query_set_directory = sys.argv[2]
    museum = Museum(dataset_directory, query_set_directory)

    

if __name__ == "__main__":
    main()