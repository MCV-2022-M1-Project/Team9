"""
Generate similarity results given a query folder
Usage:
  compute_similarity.py <queryDir> [--distance=<dist>] [--K=<k>] [--saveResultsPath=<ppath>] [--DBpicklePath=<dbppath>] [--removeBG=<bg>] [--GT=<gt>] [--max_paintings=<mp>]
  compute_similarity.py -h | --help
  -
  <inputDir>                Directory with database data 
  <queryDir>                Directory with query data
Options:
  --distance=<dist>         Distance to compute image similarity (L1, L2, X2, HIST_INTERSECTION, HELLINGER_KERNEL) [default: X2]
  --K=<k>                   Number of similar results to output [default: 3]
  --saveResultsPath=<ppath>      Filename/path to save the pkl results file (masks will be saved into the same dir if --removeBG==True) [default: ./]
  --DBpicklePath=<dbppath>  Filename/path to load the pkl database generated with compute_descriptors.py [default: ./database.pkl]
  --removeBG=<bg>           Whether or not to remove the background of the query images. If the value is different than False, it will remove the background using the specified technique (False, MORPHOLOGY, HSV, LAB, OTSU) [default: False]
  --GT=<gt>                 Whether or not there's ground truth available (True/False) [default: True]
  --max_paintings=<mp>      Max paintings per image [default: 1]

"""

import pickle
from src.Museum import Museum
from src.Measures import Measures
from docopt import docopt
import cv2

def main():
    #read arguments
    args = docopt(__doc__)
    query_set_directory = args['<queryDir>']
    distance_arg = args['--distance']
    K = int(args['--K'])
    save_results_path = args['--saveResultsPath']
    db_pickle_path = args['--DBpicklePath']
    remove_bg_flag = args['--removeBG'][0]
    gt_flag = args['--GT']
    max_paintings = int(args['--max_paintings'])
    
    print("Query directory path: ", save_results_path)
    print("DB .pkl file path ", db_pickle_path)
    print("Save results path: ", save_results_path)
    print("Remove background: ",remove_bg_flag)
    print("Compute ground truth metrics: ",gt_flag)
    print("K: ", K)
    print("Distance: ", distance_arg)
    #load query images and database
    museum = Museum(query_set_directory,db_pickle_path,gt_flag)
      
    ##GENERATE QUERY RESULTS
    predicted_top_K_results = []    #list containing in each position a K-element list of the predictions for that query
    avg_precision = 0
    avg_recall = 0
    avg_F1_score = 0
    number_of_queries_mask_evaluation = 0
    print("Computing distances with DB images...")
    
    #for each one of the queries
    for current_query in museum.query_set:
      #if user input specified to remove the background, remove it
      if remove_bg_flag != "False":
        current_query.mask = current_query.remove_background(method=remove_bg_flag)  #remove background and save the masks into the given path
        
        #postprocess mask to improve the results
        #current_query.postprocess_mask()

        #save mask into inputted path
        cv2.imwrite(str(save_results_path+str(current_query.id).zfill(5)+".png"), current_query.mask)

        #compute metrics if there's a ground truth
        if(gt_flag=='True'):
        
          #load gt mask
          mask_gt_path = str(current_query.file_directory.split(".jpg")[0]+".png")
          mask_gt = cv2.imread(mask_gt_path,0)
          #compute the fscore, precision and recall
          pixel_precision, pixel_recall, pixel_F1_score = Measures.compute_mask_metrics(mask = current_query.mask, mask_gt = mask_gt)
          #add scores of each query to the average
          avg_F1_score = avg_F1_score + pixel_F1_score
          avg_recall= avg_recall + pixel_recall
          avg_precision = avg_precision + pixel_precision
          number_of_queries_mask_evaluation = number_of_queries_mask_evaluation+1

        print("Query: ", current_query.file_directory)

      if (max_paintings>1):
        paintings = current_query.count_paintings(max_paintings)  #given a mask, count how may paintings there are

      else:
        paintings = [current_query]
      
      #for each painting in the query image, obtain the descriptor
      print("Found ", len(paintings), "painting(s)")
      for painting in paintings:
        print("Computing descriptor of painting")
        painting.compute_descriptor(museum.config)
        #detect text
        #painting.detect_text()

      predicted_top_K_results.append(museum.retrieve_top_K_results(paintings,K,distance_string = distance_arg, max_paintings = max_paintings))
        
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