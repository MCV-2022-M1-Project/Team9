"""
Generate similarity results given a query folder
Usage:
  compute_similarity.py <queryDir> [--distance=<dist>] [--K=<k>] [--picklePath=<ppath>] [--DBpicklePath=<dbppath>] [--removeBG=<bg>] [--GT=<gt>] [--BGColorspace=<bgc>]
  compute_similarity.py -h | --help
  -
  <inputDir>                Directory with database data 
  <queryDir>                Directory with query data
Options:
  --distance=<dist>         Distance to compute image similarity (L1, L2, X2, HIST_INTERSECTION, HELLINGER_KERNEL) [default: L1]
  --K=<k>                   Number of similar results to output [default: 3]
  --picklePath=<ppath>      Filename/path to save the pkl results file (masks will be saved into the same dir if --removeBG==True) [default: ./]
  --DBpicklePath=<dbppath>  Filename/path to load the pkl database generated with compute_descriptors.py [default: ./database.pkl]
  --removeBG=<bg>           Whether or not to remove the background of the query images. If the value is different than False, it will remove the background using the specified histogram technique (False, HSV, LAB, OTSU) [default: False]
  --GT=<gt>                 Whether or not there's ground truth available (True/False) [default: True]
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
    remove_bg_flag = args['--removeBG'][0]
    gt_flag = args['--GT']
    
    print("FLAG ",remove_bg_flag)
    #load database (descriptors and config of those descriptors/how they are defined) and load query images + compute their descriptions with said configuration
    museum = Museum(query_set_directory,db_pickle_path,gt_flag)
      
    ##GENERATE QUERY RESULTS
    predicted_top_K_results = []    #list containing in each position a K-element list of the predictions for that query
    #for each one of the queries
    avg_precision = 0
    avg_recall = 0
    avg_F1_score = 0
    number_of_queries_mask_evaluation = 0
    print("Computing distances with DB images...")
    for current_query in museum.query_set:
      #if user input specified to remove the background, remove it
      
      if remove_bg_flag != "False":
        
        current_query.mask , pixel_precision, pixel_recall, pixel_F1_score = current_query.remove_background(save_results_path,method=remove_bg_flag,computeGT=gt_flag )  #remove background and save the masks into the given path
        
        #compute metrics if there's a ground truth
        if(gt_flag=='True'):
          #add scores of each query to the average
          avg_F1_score = avg_F1_score + pixel_F1_score
          avg_recall= avg_recall + pixel_recall
          avg_precision = avg_precision + pixel_precision
          number_of_queries_mask_evaluation = number_of_queries_mask_evaluation+1

      print("Query: ", current_query.file_directory)
      current_query.compute_descriptor(museum.config)
      predicted_top_K_results.append(museum.retrieve_top_K_results(current_query,K,distance_arg))
        
    if(gt_flag=='True'):
      
      #compute mapk score if there's ground truth
      mapk_score = museum.compute_MAP_at_k(museum.query_gt, predicted_top_K_results, K)
      print("Ground truth: ",museum.query_gt)
      print("Predictions: ",predicted_top_K_results)
      print("MAPK score: ",mapk_score)

      #compute precision, recall and F1 score of masks if removeBG was activated
      if remove_bg_flag!= "False":
        avg_F1_score = float(avg_F1_score)/float(number_of_queries_mask_evaluation)
        avg_recall = float(avg_recall)/float(number_of_queries_mask_evaluation)
        avg_precision = float(avg_precision)/float(number_of_queries_mask_evaluation)
        print("Average precision: ", str(avg_precision))
        print("Average recall: ", str(avg_recall))
        print("Average F1 score: ", str(avg_F1_score))
        
    #save list of lists into pkl file
    with open(str(save_results_path+"result.pkl"), 'wb') as f:
        pickle.dump(predicted_top_K_results, f)

if __name__ == "__main__":
    main()