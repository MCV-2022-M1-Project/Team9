"""
Generate descriptors of the database
Usage:
  compute_descriptors.py <inputDir> [--DBpicklePath=<dbppath] [--histogramType=<histType>] [--nbins=<nbins>]
  compute_descriptors.py -h | --help
  -
  <inputDir>                Directory with database data 
Options:
  
  --DBpicklePath=<dbppath>    Filename/path to save the pkl database generated with compute_descriptors.py [default: database.pkl]
  --histogramType=<histType>  Type of histogram used to generate the descriptors (GRAYSCALE, BGR)  [default: "GRAYSCALE"]
  --nbins=<nbins>             Number of bins of the histograms [default: 16]

"""

import pickle
from Image import Image
from Museum import Museum
from docopt import docopt

def main():
    #read arguments
    args = docopt(__doc__)
    dataset_directory = args['<inputDir>']
    db_pickle_path = args['--DBpicklePath']

    #WIP adding options to the generation of the DB (they can be input but theyre not used)
    nbins = int(args['--nbins'])              ## of bins of the histogram DB 
    histogramType = args['--histogramType']

    print("Generating database pkl file")
    museum_dataset = Museum.read_images(dataset_directory)
    
    #save list of lists into pkl file
    with open(db_pickle_path, 'wb') as f:
            pickle.dump(museum_dataset, f)

if __name__ == "__main__":
    main()