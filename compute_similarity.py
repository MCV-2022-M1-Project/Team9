"""
Generate similarity results given a query folder
Usage:
  compute_similarity.py <queryDir> [--distance=<dist>] [--K=<k>] [--save_results_path=<ppath>] [--db_pickle_path=<dbppath>] [--remove_bg=<bg>] [--GT=<gt>] [--max_paintings=<mp>] [--read_text=<rt>] [--text_as_descriptor=<td>] [--denoise=<dn>] [--match_thresh=<mt>]
  compute_similarity.py -h | --help
  -
  <queryDir>                Directory with query data
Options:
  --distance=<dist>             Distance to compute image similarity (L1, L2, X2, HIST_INTERSECTION, HELLINGER_KERNEL) [default: X2]
  --K=<k>                       Number of similar results to output [default: 3]
  --save_results_path=<ppath>   Filename/path to save the pkl results file (masks will be saved into the same dir if --remove_bg==True) [default: ./]
  --db_pickle_path=<dbppath>    Filename/path to load the pkl database generated with compute_descriptors.py [default: ./database.pkl]
  --remove_bg=<bg>              Whether or not to remove the background of the query images. If the value is different than False, it will remove the background using the specified technique (False, CANNY, MORPHOLOGY, HSV, LAB, OTSU) [default: CANNY]
  --GT=<gt>                     Whether or not there's ground truth available (True/False) [default: True]
  --max_paintings=<mp>          Max paintings per image [default: 1]
  --read_text=<rt>              Whether or not there is text to read in the paintings [default: True]
  --text_as_descriptor=<tad>    Whether or not the text will be used to improve the k results [default: False]
  --denoise=<dn>                Denoising mode (simple,BM3D,False) [default: False]
  --match_thresh=<mt>           Threshold to decide if an image is a match or not [default: 20]   
"""

import pickle
from src.Museum import Museum
from src.Rotation import Rotation
from src.Image import Image
from src.TextDetection import TextDetection
from evaluation.bbox_iou import bbox_iou, shape_iou
from src.Measures import Measures
from src.Denoise import Denoise
from docopt import docopt
import skimage
from skimage import data, draw, io
import cv2,sys
import numpy as np

from pathlib import Path


def main():
    #read arguments
    args = docopt(__doc__)
    query_set_directory = args['<queryDir>']
    distance_arg = args['--distance']
    K = int(args['--K'])
    save_results_path = args['--save_results_path']
    db_pickle_path = args['--db_pickle_path']
    remove_bg_flag = args['--remove_bg'][0]
    gt_flag = args['--GT']
    max_paintings = int(args['--max_paintings'])
    read_text = args['--read_text']
    text_as_descriptor = args['--text_as_descriptor']
    denoise_mode = args['--denoise']
    match_thresh = float(args['--match_thresh'])

    print("Query directory path: ", save_results_path)
    print("DB .pkl file path ", db_pickle_path)
    print("Save results path: ", save_results_path)
    print("Remove background: ",remove_bg_flag)
    print("Compute ground truth metrics: ",gt_flag)
    print("K: ", K)
    print("Distance: ", distance_arg)
    #load query images and database
    museum = Museum(query_set_directory,db_pickle_path,gt_flag, match_thresh)
      
    ##GENERATE QUERY RESULTS
    predicted_top_K_results = []    #list containing in each position a K-element list of the predictions for that query
    text_boxes_list = []  #list containing the bbox coordinates of the predicted text boxes
    text_predictions = [] #list containing the text predictions 
    frame_coordinates = []

    #mask measurements
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    #whether noise gets detected properly or not measurements
    total_noisedetect_TP = 0
    total_noisedetect_FP = 0
    total_noisedetect_TN = 0
    total_noisedetect_FN = 0
    psnr_avg = 0
    total_noisy = 0

    #fscore detecting images as -1
    total_TP_detect_unknowns = 0
    total_FP_detect_unknowns = 0
    total_TN_detect_unknowns = 0
    total_FN_detect_unknowns = 0
    temp_var = 0
    var_text = 0
    img_cropped = None
    idx_temp = 0
    #text IoU
    IoU_average = 0 #store average of text IoU
    IoU_average_frame = 0 #store average of frame IoU
    number_of_queries_mask_evaluation = 0
    print("Processing queries...")
    #for each one of the queries
    for idx_query, current_query in enumerate(museum.query_set):

      #denoise the image if necessary 
      img = cv2.imread(current_query.file_directory)
      if denoise_mode=="simple":
        img, is_noisy = Denoise.remove_noise_simple(img)
      elif denoise_mode=='BM3D':
        img, is_noisy = Denoise.remove_noise_BM3D(img)
        #cv2.imwrite("./denoised_sd1w5/"+str(current_query.id).zfill(5)+".jpg", img)
        #continue
      else:
        is_noisy = False

      #compute tp/tn/fp/fn if there's gt of the denoising
      if hasattr(museum, 'augmentations_gt'):
        current_gt_augm = museum.augmentations_gt[idx_query]
        filename_path_without_extension = current_query.file_directory.rsplit('/',1)[0]
        non_augmented = cv2.imread(str(filename_path_without_extension+"/non_augmented/"+str(current_query.id).zfill(5)+".jpg"))
        
        if is_noisy:
          psnr  = cv2.PSNR(non_augmented, img)
          psnr_avg = psnr+psnr_avg
          total_noisy = total_noisy+1
        is_noisy_gt = not "None" in current_gt_augm #==True if ground truth does not contain None (used to show it doesnt have noise)
        if is_noisy_gt== False and is_noisy==True:
          total_noisedetect_FP = total_noisedetect_FP+1
        elif is_noisy_gt== True and is_noisy==True:
          total_noisedetect_TP = total_noisedetect_TP+1
        elif is_noisy_gt== False and is_noisy==False:
          total_noisedetect_TN = total_noisedetect_TN+1
        elif is_noisy_gt== True and is_noisy==False:
          total_noisedetect_FN = total_noisedetect_FN+1
        
        
      #if user input specified to remove the background, remove it
      if remove_bg_flag != "False":
        current_query.mask = current_query.remove_background(image = img, method=remove_bg_flag)  #remove background and save the masks into the given path
        
        #save mask into inputted path
        cv2.imwrite(str(save_results_path+str(current_query.id).zfill(5)+".png"), current_query.mask)

        #accumulate TP/TN/FP/FN if there's a mask ground truth
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

      #split mask into multiple ones if necessary
      if (max_paintings>1):
        paintings = current_query.count_paintings(max_paintings)  #given a mask, count how may paintings there are and store them in paintings
      else:
        paintings = [current_query]
      
      print("Found ", len(paintings), "painting(s)")

      #text coordinates variables
      text_coordinates_query = []   #array to store all the text coordinates of a single query (there may be multiple paintings per query)
      text_predictions_query = []   #array to store all the text predictions of a single query (there may be multiple paintings per query)
      frame_coordinates_query = []  #array to store all the frame coordinates of a single query (there may be multiple paintings per query)

      #for each painting in the query image, obtain the descriptor
      for idx_painting, painting in enumerate(paintings):
        #if there's more than one painting, crop the region of the current one to feed it to the text detection module instead of sending the whole image
        if remove_bg_flag!="False":
          img_cropped, top_left_coordinate_offset, bottom_right_coordinate_offset = painting.crop_image_with_mask_bbox(img)
        
          #cv2.imwrite( "./kpimg/"+str(temp_var)+".png", img_cropped)
          #temp_var = temp_var+1

        #else send the entire image
        else:
          img_cropped = img
          top_left_coordinate_offset = [0,0]
          bottom_right_coordinate_offset = [img.shape[1]-1, img.shape[0]-1]

        #undo rotation of the image
        fixed_rotation_img, angle = Rotation.fix_image_rotation(img_cropped)
        img_cropped = fixed_rotation_img

        ###TODO convert coordinates to original domain
        x1 = top_left_coordinate_offset[1]
        x2 = bottom_right_coordinate_offset[1]
        y1 = top_left_coordinate_offset[0]
        y2 = bottom_right_coordinate_offset[0]
        ##########TEMP
        #angle = -1
        mean_x = (x2-x1)/2
        mean_y = (y2-y1)/2
        origin = (mean_x, mean_y)
        coord1, coord2, coord3, coord4 = Rotation.rotate_coordinates((x1,y1), (x2,y1), (x2,y2), (x1,y2), -angle, origin)
        coordinates_original_domain = [coord1, coord2, coord3, coord4]
        draw = False
        if draw :
          row, col = draw.polygon_perimeter((coord1[1],coord2[1],coord3[1],coord4[1]), (coord1[0],coord2[0],coord3[0],coord4[0]))
          temp_img = img.copy()
          temp_img[row, col] = (255,0,0)
          cv2.imwrite("./rotation/"+str(idx_temp)+".png",temp_img)
        idx_temp = idx_temp +1
        if True:
          img_cropped_with_padding, _, _ = painting.crop_image_with_mask_bbox(img, margins = 50)
          #cv2.imwrite("./cropped_w5/"+str(idx_temp)+".png", img_cropped_with_padding)
          
        #mask_cropped_with_padding, _, _ = painting.crop_image_with_mask_bbox(curr_mask_before_text, margins = 50)
        #cv2.imwrite("./masks_w5/"+str(idx_temp)+".png",mask_cropped_with_padding)
        #idx_temp = idx_temp +1

        #save to list
        frame_coordinates_query.append([angle,coordinates_original_domain])
        curr_mask_before_text = painting.mask
        if read_text=="True":
        
          #detect text and get coordinates
          [tlx1, tly1, brx1, bry1], text_mask= TextDetection.detect_text(img = img_cropped)
          #if background has to be removed, pad the mask to the original size (bounding box has been made in the cropped image)
          if remove_bg_flag!="False":
            text_mask = cv2.copyMakeBorder( text_mask,  top_left_coordinate_offset[0], painting.mask.shape[0]-text_mask.shape[0]-top_left_coordinate_offset[0], top_left_coordinate_offset[1], painting.mask.shape[1]-text_mask.shape[1]-top_left_coordinate_offset[1], cv2.BORDER_CONSTANT, None, value = 255)
          else:
            painting.mask = np.ones((img.shape[0],img.shape[1]))
          
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
          bounding_box_im = cv2.rectangle(img.copy(), (tlx1,tly1), (brx1,bry1), (0,0,0), -1)
          painting.text_coordinates = text_coordinates
          
          cropped_textbox = img[tly1:bry1,tlx1:brx1]
          
          #if there's a textbox (something got detected), read it
          if(len(cropped_textbox)>0):
            
            text_string = TextDetection.read_text(cropped_textbox, j= var_text)
            var_text = var_text+1
          else:
            text_string = ""

          #add to query list
          text_coordinates_query.append(text_coordinates)
          text_predictions_query.append(text_string)

        print("Computing descriptor of painting")
        painting.compute_descriptor(img, museum.config,cropped_img = img_cropped)
      coordinates_and_angle = frame_coordinates_query
      frame_coordinates.append(coordinates_and_angle)
      #append it to global list
      if read_text=="True":
        text_boxes_list.append(text_coordinates_query)
        text_predictions.append(text_predictions_query)
        print("Predicted names", text_predictions_query)
        # open file in write mode
        with open(str(save_results_path+str(current_query.id).zfill(5)+".txt"), 'w') as fp:
            for prediction in text_predictions_query:
                # write each item on a new line
                fp.write("%s\n" % prediction)
      #send variable with text information if it is going to be used to try to improve the results
      if text_as_descriptor == "True":
        #compute top k results
        predicted_top_K_results.append(museum.retrieve_top_K_results(paintings,K,distance_string = distance_arg, max_paintings = max_paintings,text_string_list = text_predictions_query))
      else:
        #compute top k results
        predicted_top_K_results.append(museum.retrieve_top_K_results(paintings,K,distance_string = distance_arg, max_paintings = max_paintings,text_string_list = None))

    #save list of lists of frame coordinates
    with open(str(save_results_path+"frames.pkl"), 'wb') as f:
        pickle.dump(frame_coordinates, f)
    #compute mapk score if there's ground truth
    if(gt_flag=='True'):
      mapk_average = 0  #average mapk
      IoU_total = 0 #amount of paintings where IoU>0 (box has been detected)
      #counters to keep track of the query number and the painting number
      IoU_frame_total = 0
      query_num = 0
      paintings_num = 0
      for query in museum.query_gt:
        i = 0
        query_text_path = str(query_set_directory+str(query_num).zfill(5)+'.txt')
        for painting in query:
          #mapk = 0 if we didnt predict the painting (don't add anything to the score)
          if len(predicted_top_K_results[query_num])-1<i:
            paintings_num = paintings_num+1
            continue
          
          #compute frame iou if there's ground truth
          if hasattr(museum, 'frames_gt'):
            coordinates_gt = museum.frames_gt[query_num][i][1]
            predicted_coordinates = frame_coordinates[query_num][i][1]
            if(len(coordinates_gt)==4 and len(predicted_coordinates)==4):
              IoU_frame = shape_iou(coordinates_gt, predicted_coordinates)
              print("Frame IoU", IoU_frame)
              IoU_average_frame = IoU_average_frame + IoU_frame
              IoU_frame_total = IoU_frame_total+1
          #compute IoU if there's ground truth
          if read_text=="True":
            IoU = bbox_iou(museum.text_boxes_gt[query_num][i], text_boxes_list[query_num][i])
            print("Text IoU", IoU)
            IoU_average = IoU_average+IoU

            #increase counter of correct detections
            if IoU!= 0:
              IoU_total = IoU_total+1

          mapk_average = mapk_average+museum.compute_MAP_at_k([[painting]], [predicted_top_K_results[query_num][i]], K)
          print("PRED ", predicted_top_K_results[query_num][i][0])
          print("QUERY ",painting )
          predicted_detection = predicted_top_K_results[query_num][i][0]
          query_detection = painting
          if query_detection == -1 and predicted_detection == -1:
            total_TP_detect_unknowns = total_TP_detect_unknowns+1
          elif query_detection != -1 and predicted_detection != -1:
            total_TN_detect_unknowns = total_TN_detect_unknowns+1
          elif query_detection == -1 and predicted_detection != -1:
            total_FN_detect_unknowns = total_FN_detect_unknowns+1
          elif query_detection != -1 and predicted_detection == -1:
            total_FP_detect_unknowns = total_FP_detect_unknowns+1
          i = i+1
          paintings_num = paintings_num+1
        
        query_num = query_num+1
      mapk_score = mapk_average/paintings_num

      print("Ground truth: ",museum.query_gt)
      print("Predictions: ",predicted_top_K_results)
      print("MAPK score: ",mapk_score)
    
      pixel_precision_unknowns,pixel_recall_unknowns,pixel_F1_score_unknowns =Measures.compute_precision_recall_F1(total_TP_detect_unknowns, total_FP_detect_unknowns, total_FN_detect_unknowns, total_TN_detect_unknowns)
      print("Average precision (detect unkowns): ", str(pixel_precision_unknowns))
      print("Average recall (detect unkowns): ", str(pixel_recall_unknowns))
      print("Average F1 score (detect unkowns): ", str(pixel_F1_score_unknowns))

      #compute precision, recall and F1 score of masks if removeBG was activated
      if remove_bg_flag!= "False":
        #obtain precision and recall
        pixel_precision,pixel_recall,pixel_F1_score =Measures.compute_precision_recall_F1(total_TP, total_FP, total_FN, total_TN)
            
        print("Average precision: ", str(pixel_precision))
        print("Average recall: ", str(pixel_recall))
        print("Average F1 score: ", str(pixel_F1_score))
      if read_text == "True":
        IoU_average = IoU_average/IoU_total
        print("Average IoU (text): ", str(IoU_average))
        print("Paintings num",paintings_num )
        print("Text predictions: ", text_predictions)
      if hasattr(museum, 'augmentations_gt'):
        #obtain precision and recall of the noise detection
        pixel_precision_noise,pixel_recall_noise,pixel_F1_score_noise = Measures.compute_precision_recall_F1(total_noisedetect_TP, total_noisedetect_FP, total_noisedetect_FN, total_noisedetect_TN)
        print("Average precision (noise detection): ", str(pixel_precision_noise))
        print("Average recall (noise detection): ", str(pixel_recall_noise))
        print("Average F1 score (noise detection): ", str(pixel_F1_score_noise))
        if total_noisy != 0:
          print("PSNR AVERAGE: ", psnr_avg/total_noisy)
      if hasattr(museum, 'frames_gt'):
        print("PAINTINGS NUM ", paintings_num)
        IoU_average_frame = IoU_average_frame/IoU_frame_total
        print("Average IoU (frames): ",IoU_average_frame)
    #save list of lists of predictions into pkl file
    with open(str(save_results_path+"result.pkl"), 'wb') as f:
        pickle.dump(predicted_top_K_results, f)
    
    #save list of lists of bboxes in pkl file
    if read_text=='True':
      print("TEXT BOXES ",text_boxes_list)
      with open(str(save_results_path+"text_boxes.pkl"), 'wb') as f:
          pickle.dump(text_boxes_list, f)
    
    #save list of lists of frame coordinates
    with open(str(save_results_path+"frames.pkl"), 'wb') as f:
        pickle.dump(frame_coordinates, f)
    

if __name__ == "__main__":
    main()