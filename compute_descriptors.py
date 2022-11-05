"""
Generate descriptors of the database
Usage:
  compute_descriptors.py <inputDir> [--DBpicklePath=<dbppath>] [--histogramType=<histType>] [--nbins=<nbins>] [--descriptorType=<dtype>] [--level=<lv>] [--max_level=<mlv>] [--lbp_radius=<lbpr>] [--dct_block_size=<dctb>] [--weights=<wg>]
  compute_descriptors.py -h | --help
  -
  <inputDir>                Directory with database data 
  
Options:
  
  --DBpicklePath=<dbppath>    Filename/path to save the pkl database generated with compute_descriptors.py [default: database.pkl]
  --histogramType=<histType>  Type of histogram used to generate the descriptors (GRAYSCALE, BGR, HSV, YCRCB, LAB)  [default: GRAYSCALE]
  --nbins=<nbins>             Number of bins of the histograms [default: 16]
  --descriptorType=<dtype>    Type of descriptor (1Dhistogram,mult_res_histogram,block_histogram,HOG,LBP,DCT) [default: 1Dhistogram]
  --level=<lv>                Levels of block histogram [default: 4]
  --max_level=<mlv>           Levels of multiresolution histogram [default: 2]
  --lbp_radius=<lbpr>         Radius used for the LBP descriptor [default: 4]
  --dct_block_size=<dctb>     Size of the blocks the DCT image will be split in [default: 32]
  --weights=<wg>              Weights of the descriptors (e.g.: 0.75,0.25) [default: -1]
"""

import pickle
from src.Museum import Museum
from docopt import docopt

def main():
    #read arguments
    args = docopt(__doc__)
    dataset_directory = args['<inputDir>']  #path where the database is
    db_pickle_path = args['--DBpicklePath'] #path to save the pkl database file
    descriptor_type = args['--descriptorType'] #type of descriptor used to compare the distances
    weights =  args['--weights']  #comma separated string where each position is the weight of a descriptor

    print("TYPE ", type(descriptor_type))
    descriptors_array = descriptor_type.split(",")
    if weights != "-1":
      weights_array = args['--weights'].split(",")
    museum_config = []

    #concatenate descriptors if needed (eg. colour descriptors and texture descriptors)
    for i, descriptor in enumerate(descriptors_array):
      if weights != "-1":
        curr_weight = float(weights_array[i])
      else:
        curr_weight = 1
      #read options specific to the descriptor type
      if descriptor =="1Dhistogram":
        nbins = int(args['--nbins'])              # # of bins of the histogram DB 
        histogram_type = args['--histogramType']   
        museum_config.append({"descriptorType":descriptor, "histogramType": histogram_type ,"nbins": nbins, "weight":curr_weight}) #empty dictionary with config info

      elif descriptor =="mult_res_histogram":
        nbins = int(args['--nbins'])              # # of bins of the histogram DB 
        histogram_type = args['--histogramType'] 
        max_level = int(args['--max_level'])
        museum_config.append({"descriptorType":descriptor, "histogramType": histogram_type ,"nbins": nbins, "max_level": max_level, "weight":curr_weight}) #empty dictionary with config info
      
      elif descriptor =="block_histogram":
        nbins = int(args['--nbins'])              # # of bins of the histogram DB 
        histogram_type = args['--histogramType'] 
        level = int(args['--level'])
        museum_config.append({"descriptorType":descriptor, "histogramType": histogram_type ,"nbins": nbins, "level": level, "weight":curr_weight}) #empty dictionary with config info
      
      elif descriptor=="HOG":
        museum_config.append({"descriptorType":descriptor, "weight":curr_weight}) #empty dictionary with config info

      elif descriptor=="LBP":
        radius = int(args['--lbp_radius'])
        museum_config.append({"descriptorType":descriptor, "lbp_radius": radius, "weight":curr_weight}) #empty dictionary with config info

      elif descriptor=="DCT":
        block_size = int(args['--dct_block_size'])
        museum_config.append({"descriptorType":descriptor, "dct_block_size":block_size, "weight":curr_weight}) #empty dictionary with config info
      
      elif descriptor=="SIFT":
        museum_config.append({"descriptorType":descriptor, "weight":curr_weight})

      elif descriptor=="ORB":
        museum_config.append({"descriptorType":descriptor, "weight":curr_weight})
      elif descriptor=="SURF":
        museum_config.append({"descriptorType":descriptor, "weight":curr_weight})
      elif descriptor=="DAISY":
        museum_config.append({"descriptorType":descriptor, "weight":curr_weight})

      #print configuration of each descriptor
      print("Descriptor settings: ")
      for key, value in museum_config[i].items():
          print(key, ' : ', value)
      print()

    print("DB pickle path: ", db_pickle_path)
    #create Image objects (obtain ID and filepaths of each image)
    print("Loading images")
    museum_dataset, dict_artists_paintings, dict_titles_paintings= Museum.read_images(dataset_directory)
  
    print("Computing descriptors...")
    #compute the descriptors of our database
    for image_object in museum_dataset:
      image = image_object.read_image_BGR()
      image_object.compute_descriptor(image,museum_config)

    #read relationships file
    db_relationships = Museum.read_pickle(dataset_directory + '/relationships.pkl')

    #save list of lists into pkl file
    #first field contains the image objects containing the descriptor information, second field contains relationships.pkl file and the last field contains the configuration of the descriptors
    with open(db_pickle_path, 'wb') as f:
            pickle.dump([museum_dataset,db_relationships, museum_config, dict_artists_paintings], f)


if __name__ == "__main__":
    main()