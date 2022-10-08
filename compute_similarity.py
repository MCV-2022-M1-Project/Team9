"""
Generate similarity results given a query folder
Usage:
  compute_similarity.py <queryDir> [--distance=<dist>] [--K=<k>] [--picklePath=<ppath>] [--DBpicklePath=<dbppath>] [--removeBG=<bg>]
  compute_similarity.py -h | --help
  -
  <inputDir>                Directory with database data 
  <queryDir>                Directory with query data
Options:
  --distance=<dist>         Distance to compute image similarity (L1, L2, X2, HIST_INTERSECTION, HELLINGER_KERNEL) [default: L1]
  --K=<k>                   Number of similar results to output [default: 3]
  --picklePath=<ppath>      Filename/path to save the pkl results file (masks will be saved into the same dir if --removeBG==True) [default: ./]
  --DBpicklePath=<dbppath>  Filename/path to load the pkl database generated with compute_descriptors.py [default: ./database.pkl]
  --removeBG=<bg>           Whether or not to remove the background of the query images (True/False) [default: False]
"""

import pickle
from src.Museum import Museum
from docopt import docopt

def main():
    #read arguments
    args = docopt(__doc__)
    query_set_directory = args['<queryDir>']
    distance_arg = args['--distance']
    K = int(args['--K'])
    save_results_path = args['--picklePath']
    db_pickle_path = args['--DBpicklePath']
    print("DB PATH", db_pickle_path)
    remove_bg_flag = args['--removeBG']
    
    print("remove background", remove_bg_flag)

    #load database (descriptors and config of those descriptors/how they are defined) and load query images + compute their descriptions with said configuration
    museum = Museum(query_set_directory,db_pickle_path)
      
    ##GENERATE QUERY RESULTS
    predicted_top_K_results = []    #list containing in each position a K-element list of the predictions for that query
    #for each one of the queries
    
    print("Computing distances with DB images...")
    for current_query in museum.query_set:
      #if user input specified to remove the background, remove it
      
      print("remove_bg_flag", remove_bg_flag)
      if remove_bg_flag[0] == "True":
        print("INT")
        current_query.mask = current_query.remove_background(save_results_path)  #remove background and save the masks into the given path

      print("Query: ", current_query.file_directory)
      current_query.compute_descriptor(museum.config)
      predicted_top_K_results.append(museum.retrieve_top_K_results(current_query,K,distance_arg))
        
    #print("querygt",museum.query_gt)
    #print("predictions",predicted_top_K_results)
    mapk_score = museum.compute_MAP_at_k(museum.query_gt, predicted_top_K_results, K)
    
    print("Using distance", distance_arg)
    #print("TOP ",K, " RESULTS: ",predicted_top_K_results)
    print("MAPK score: ",mapk_score)

    #save list of lists into pkl file
    with open(str(save_results_path+"result.pkl"), 'wb') as f:
        pickle.dump(predicted_top_K_results, f)

if __name__ == "__main__":
    main()