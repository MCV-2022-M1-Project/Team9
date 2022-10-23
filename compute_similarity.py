"""
Generate similarity results given a query folder
Usage:
  compute_similarity.py <queryDir> [--distance=<dist>] [--K=<k>] [--saveResultsPath=<ppath>] [--DBpicklePath=<dbppath>] [--removeBG=<bg>] [--GT=<gt>] [--max_paintings=<mp>] [--read_text=<rt>]
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
  --read_text=<rt>          Whether or not there is text to read in the paintings [default: False]

"""

import pickle
from src.Museum import Museum
from src.Image import Image
from src.TextDetection import TextDetection
from evaluation.bbox_iou import bbox_iou
from src.Measures import Measures
from docopt import docopt
import cv2,sys
import numpy as np


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
    read_text = args['--read_text']
    
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
    text_boxes_list = []
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    IoU_average = 0
    number_of_queries_mask_evaluation = 0
    print("Computing distances with DB images...")
    
    #for each one of the queries
    for current_query in museum.query_set:
      #if user input specified to remove the background, remove it
      if remove_bg_flag != "False":
        current_query.mask = current_query.remove_background(method=remove_bg_flag)  #remove background and save the masks into the given path
        
        #save mask into inputted path
        cv2.imwrite(str(save_results_path+str(current_query.id).zfill(5)+".png"), current_query.mask)

        #compute metrics if there's a ground truth
        if(gt_flag=='True'):
        
          #load gt mask
          mask_gt_path = str(current_query.file_directory.split(".jpg")[0]+".png")
          mask_gt = cv2.imread(mask_gt_path,0)
          #compute the TP,TN,...
          pixelTP,pixelFP,pixelFN,pixelTN= Measures.compute_TP_FP_FN_TN(mask = current_query.mask, mask_gt = mask_gt)
          #add pixels of each query to the total
          total_TP =total_TP+ pixelTP
          total_FP=total_FP+ pixelFP
          total_TN =total_TN+ pixelTN
          total_FN=total_FN+pixelFN

        print("Query: ", current_query.file_directory)

      if (max_paintings>1):
        paintings = current_query.count_paintings(max_paintings)  #given a mask, count how may paintings there are

      else:
        paintings = [current_query]
      
      #for each painting in the query image, obtain the descriptor
      print("Found ", len(paintings), "painting(s)")

      text_coordinates_query = [] #array to store all the text coordinates of a single query (there may be multiple paintings per query)
      for painting in paintings:

        if read_text=="True":
          top_left_coordinate_offset = [0,0]
          #if there's more than one painting, crop the region of the current one to feed it to the text detection module instead of sending the whole image
          if remove_bg_flag!="False":
            img_cropped, top_left_coordinate_offset = painting.crop_image_with_mask_bbox()
          #else send the entire image
          else:
            img_cropped = painting.read_image_BGR()
        
          #detect text and get coordinates
          [tlx1, tly1, brx1, bry1], text_mask= TextDetection.detect_text(img = img_cropped)
          #if background has to be removed, pad the image to the original size (bounding box has been made in the cropped image)
          if remove_bg_flag!="False":
            text_mask = cv2.copyMakeBorder( text_mask,  top_left_coordinate_offset[0], painting.mask.shape[0]-text_mask.shape[0]-top_left_coordinate_offset[0], top_left_coordinate_offset[1], painting.mask.shape[1]-text_mask.shape[1]-top_left_coordinate_offset[1], cv2.BORDER_CONSTANT, None, value = 255)
          else:
            painting.mask = np.ones((painting.read_image_BGR().shape[0],painting.read_image_BGR().shape[1]))
          
          #cast to uint8
          text_mask = text_mask.astype(np.uint8)
          painting.mask = painting.mask.astype(np.uint8)
          
          #obtain intersection of the original mask and the one with the text area==0
          painting.mask = cv2.bitwise_and(painting.mask, text_mask)
          
          #get coordinates in the full image (text coordinates were originally from the cropped image)
          tlx1=tlx1+top_left_coordinate_offset[1]
          brx1=brx1+top_left_coordinate_offset[1]
          tly1=tly1+top_left_coordinate_offset[0]
          bry1=bry1+top_left_coordinate_offset[0]

          text_coordinates = [tlx1, tly1, brx1, bry1]
          bounding_box_im = cv2.rectangle(painting.read_image_BGR(), (tlx1,tly1), (brx1,bry1), (0,0,0), -1)
          painting.text_coordinates = text_coordinates
          
          img = painting.read_image_BGR()
          cropped_textbox = img[tly1:bry1,tlx1:brx1]
          #if there's a textbox (something got detected), read it
          if(len(cropped_textbox)>0):
            text_string = TextDetection.read_text(cropped_textbox)
            cv2.imwrite(str("test.png"), cropped_textbox)
          #add to query list
          text_coordinates_query.append(text_coordinates)
        print("Computing descriptor of painting")
        painting.compute_descriptor(museum.config)
      #append it to global list
      if read_text=="True":
        text_boxes_list.append(text_coordinates_query)
        
      #compute top k results
      predicted_top_K_results.append(museum.retrieve_top_K_results(paintings,K,distance_string = distance_arg, max_paintings = max_paintings))
        
    #compute mapk score if there's ground truth
    if(gt_flag=='True'):
      mapk_average = 0  #average mapk
      IoU_total = 0 #amount of paintings where IoU>0 (box has been detected)
      #counters to keep track of the query number and the painting number
      query_num = 0
      paintings_num = 0
      for query in museum.query_gt:
        i = 0
        for painting in query:
          #mapk = 0 if we didnt predict the painting (don't add anything to the score)
          if len(predicted_top_K_results[query_num])-1<i:
            paintings_num = paintings_num+1
            continue
            
          #compute IoU if there's ground truth
          if read_text=="True":
            IoU = bbox_iou(museum.text_boxes_gt[query_num][i], text_boxes_list[query_num][i])
            print("IoU", IoU)
            IoU_average = IoU_average+IoU
          #increase counter of correct detections
            if IoU!= 0:
              IoU_total = IoU_total+1

          mapk_average = mapk_average+museum.compute_MAP_at_k([[painting]], [predicted_top_K_results[query_num][i]], K)
          i = i+1
          paintings_num = paintings_num+1
        
        query_num = query_num+1
      mapk_score = mapk_average/paintings_num

      print("Ground truth: ",museum.query_gt)
      print("Predictions: ",predicted_top_K_results)
      print("MAPK score: ",mapk_score)

      #compute precision, recall and F1 score of masks if removeBG was activated
      if remove_bg_flag!= "False":
        #obtain precision and recall
        pixel_precision,pixel_recall,pixel_F1_score =Measures.compute_precision_recall_F1(total_TP, total_FP, total_FN, total_TN)
            
        print("Average precision: ", str(pixel_precision))
        print("Average recall: ", str(pixel_recall))
        print("Average F1 score: ", str(pixel_F1_score))
      if read_text == "True":
        IoU_average = IoU_average/IoU_total
        print("Average IoU: ", str(IoU_average))
        print("Paintings num",paintings_num )
        
    #save list of lists of predictions into pkl file
    with open(str(save_results_path+"result.pkl"), 'wb') as f:
        pickle.dump(predicted_top_K_results, f)
    
    #save list of lists of bboxes in pkl file
    if read_text=='True':
      print("TEXT BOXES ",text_boxes_list)
      with open(str(save_results_path+"text_boxes.pkl"), 'wb') as f:
          pickle.dump(text_boxes_list, f)

if __name__ == "__main__":
    main()