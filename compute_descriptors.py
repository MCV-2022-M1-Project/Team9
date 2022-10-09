"""
Generate descriptors of the database
Usage:
  compute_descriptors.py <inputDir> [--DBpicklePath=<dbppath] [--histogramType=<histType>] [--nbins=<nbins>] [--descriptorType=<dtype>]
  compute_descriptors.py -h | --help
  -
  <inputDir>                Directory with database data 
  
Options:
  
  --DBpicklePath=<dbppath>    Filename/path to save the pkl database generated with compute_descriptors.py [default: database.pkl]
  --histogramType=<histType>  Type of histogram used to generate the descriptors (GRAYSCALE, BGR, HSV, YCRCB, LAB)  [default: GRAYSCALE]
  --nbins=<nbins>             Number of bins of the histograms [default: 16]
  --descriptorType=<dtype>    Type of descriptor (1Dhistogram) [default: 1Dhistogram]
"""

import pickle
from src.Museum import Museum
from docopt import docopt

def main():
    #read arguments
    args = docopt(__doc__)
    dataset_directory = args['<inputDir>']
    db_pickle_path = args['--DBpicklePath']
    descriptor_type = args['--descriptorType'] #type of descriptor used to compare the distances

    #read options specific to the descriptor type
    if descriptor_type =="1Dhistogram":
      nbins = int(args['--nbins'])              # # of bins of the histogram DB 
      histogram_type = args['--histogramType']   

    #save descriptor configuration 
    #CASE: descriptor = 1D histogram (the only implementation so far)
    if(descriptor_type == "1Dhistogram"):
      museum_config ={"descriptorType":descriptor_type, "histogramType": histogram_type ,"nbins": nbins} #empty dictionary with config info

    #print configuration
    print("Descriptor settings: ")
    for key, value in museum_config.items():
        print(key, ' : ', value)

    print("DB pickle path: ", db_pickle_path)
    #create Image objects (obtain ID and filepaths of each image)
    print("Loading images")
    museum_dataset = Museum.read_images(dataset_directory, museum_config)
    
    print("Computing descriptors...")
    #compute the descriptors of our database
    for image_object in museum_dataset:
      image_object.compute_descriptor(museum_config)

    #read relationships file
    db_relationships = Museum.read_pickle(dataset_directory + '/relationships.pkl')
    #save list of lists into pkl file
    #first field contains the image objects containing the descriptor information, second field contains relationships.pkl file and the last field contains the configuration of the descriptors
    with open(db_pickle_path, 'wb') as f:
            pickle.dump([museum_dataset,db_relationships, museum_config], f)


if __name__ == "__main__":
    main()