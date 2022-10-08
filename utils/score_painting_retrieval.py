"""
Score painting retrieval tasks

Usage:
  score_painting_retrieval.py <inputDir> <gtQueryFile> [--gtPicFrames=<gpf>] [--gtTextBoxes=<gtb>] [--imaDir=<id>] [--kVal=<kv>] [--outDir=<od>] [--augList=<al>]
  score_painting_retrieval.py -h | --help
  -
  <inputDir>               Directory with hypothesis data (result.pkl file, masks, etc.)
  <gtQueryFile>            GT .pkl file. For each query image contains the index of the corresponding DDBB painting
Options:
  --gtPicFrames=<gpf>      Ground truth for picture bboxes (.pkl) [default: None]
  --gtTextBoxes=<gtb>      Ground truth for text boxes (.pkl) [default: None]
  --imaDir=<id>            Images dir (necessary to evaluate bboxes) (.pkl) [default: None]
  --kVal=<kv>              Value of k for map@k computation [default: 1]
  --outDir=<od>            Directory for output masks [default: None]
  --augList=<al>           List of augmentations (noise, color changes, etc.) applied to each image [default: None]
"""

import numpy as np
import cv2
import fnmatch
import os
import sys
import pickle
from docopt import docopt
import imageio
from ml_metrics import mapk,apk
#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score
import evaluation.evaluation_funcs as evalf
import geometry_utils as gu
from operator import itemgetter
from Levenshtein import distance

#from shapely.geometry import Polygon
#import scipy
#from munkres import Munkres

# Compute the depth of a list (of lists (of lists ...) ...)
# Empty list not allowed!!
#https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
list_depth = lambda L: isinstance(L, list) and max(map(list_depth, L))+1

def add_list_level(input_list):
    out = []
    for ll in input_list:
        tmp = []
        for q in ll:
            tmp.append([q])
        out.append(tmp)
    return (out)

def add_list_level2(input_list):
    out = []
    for ll in input_list:
        out.append([ll])
    return (out)


def compute_mapk(gt,hypo,k_val):

    hypo = list(hypo)
    if list_depth(hypo) == 2:
        hypo = add_list_level(hypo.copy())

    apk_list = []
    for ii,query in enumerate(gt):
        for jj,sq in enumerate(query):
            apk_val = 0.0
            if len(hypo[ii]) > jj:
                apk_val = apk([sq],hypo[ii][jj], k_val)
            apk_list.append(apk_val)
            
    return np.mean(apk_list)

def score_bboxes(hypo_bboxes, gt_bboxes):

    if len(hypo_bboxes) != len(gt_bboxes):
        print ('Error: the number of bboxes in GT and hypothesis differ!')
        sys.exit()
    
    bboxTP = 0
    bboxFN = 0
    bboxFP = 0

    bbox_precision = 0
    bbox_accuracy  = 0

    iou = 0
    for ii in range(len(gt_bboxes)):
        [local_bboxTP, local_bboxFN, local_bboxFP,local_iou] = evalf.performance_accumulation_window(hypo_bboxes[ii], gt_bboxes[ii])
        bboxTP = bboxTP + local_bboxTP
        bboxFN = bboxFN + local_bboxFN
        bboxFP = bboxFP + local_bboxFP
        iou    = iou + local_iou

    # Plot performance evaluation
    [bbox_precision, bbox_sensitivity, bbox_accuracy] = evalf.performance_evaluation_window(bboxTP, bboxFN, bboxFP)
    iou = 0 if bboxTP == 0 else iou / bboxTP
    
    bbox_f1 = 0
    if (bbox_precision + bbox_sensitivity) != 0:
        bbox_f1 = 2*((bbox_precision*bbox_sensitivity)/(bbox_precision + bbox_sensitivity))

    return bbox_precision, bbox_sensitivity, bbox_f1, bbox_accuracy, iou
        

def score_pixel_masks(result_masks, test_masks):
    pixelTP  = 0
    pixelFN  = 0
    pixelFP  = 0
    pixelTN  = 0

    for ii, mask_name in enumerate(result_masks):
        
        # Read mask file
        pixelCandidates = imageio.imread(mask_name)>0
        if len(pixelCandidates.shape) == 3:
            pixelCandidates = pixelCandidates[:,:,0]

        # Accumulate pixel performance of the current image %%%%%%%%%%%%%%%%%
        gt_mask_name = test_masks[ii]

        pixelAnnotation = imageio.imread(gt_mask_name)>0
        if len(pixelAnnotation.shape) == 3:
            pixelAnnotation = pixelAnnotation[:,:,0]


        if pixelAnnotation.shape != pixelCandidates.shape:
            print ('Error: hypothesis ({}) and  GT masks ({})dimensions do not match!'.format(pixelCandidates.shape,pixelAnnotation.shape))
            sys.exit()

        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = evalf.performance_accumulation_pixel(pixelCandidates, pixelAnnotation)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN

    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)
    pixelF1 = 0
    if (pixelPrecision + pixelSensitivity) != 0:
        pixelF1 = 2*((pixelPrecision*pixelSensitivity)/(pixelPrecision + pixelSensitivity))
        
    return pixelPrecision, pixelSensitivity, pixelF1      

    
def create_mask (ima_dir, ima_name, box):
    base, ext = os.path.splitext(ima_name)
    index  = int(base.split('_')[1])

    full_ima_name = '{}/{}'.format(ima_dir,ima_name)
        
    img = cv2.imread(full_ima_name) # trainImage

    height, width, depth = img.shape

    mask = np.zeros((height,width), np.uint8)

    box = np.array(box, np.int32).reshape((-1,1,2))

    line_type = 8
    cv2.fillPoly(mask, [box], (255), line_type)

    return mask


def create_masks (ima_dir, ima_name, boxes):
    base, ext = os.path.splitext(ima_name)
    index  = int(base)

    full_ima_name = '{}/{}'.format(ima_dir,ima_name)
    img = cv2.imread(full_ima_name) # trainImage

    height, width, depth = img.shape
    mask = np.zeros((height,width), np.uint8)

    boxes = [np.array(box, np.int32).reshape((-1,1,2)) for box in boxes]
    line_type = 8
    cv2.fillPoly(mask, boxes, (1))

    return mask
    
def angular_error_boxes(box1, box2):

    # Compute angle 1
    # 1: take the two points in the lower  part and compute angle
    lower_points = sorted(box1, key=lambda x: x[1], reverse=True)[:2]
    rho1, theta1 = gu.line_polar_params_from_points(lower_points[0], lower_points[1])
    
    # Compute angle 2
    # 2: take the two points in the lower  part and compute angle
    lower_points = sorted(box2, key=lambda x: x[1], reverse=True)[:2]
    rho2, theta2 = gu.line_polar_params_from_points(lower_points[0], lower_points[1])

    print (theta1, theta2, file=sys.stderr)
    return abs(theta1-theta2)

def angular_error_box_angle (gt_box, hyp_angle):
    # Compute angle 1
    # 1: take the two points in the lower  part and compute angle
    lower_points = sorted(gt_box, key=lambda x: x[1], reverse=True)[:2]
    rho1, theta1 = gu.line_polar_params_from_points(lower_points[0], lower_points[1])

    if theta1 > 0:
        gt_angle = ((3.0*np.pi)/2 - theta1) * (180.0/np.pi)
    else:
        gt_angle = ((3.0*np.pi)/2 + theta1) * (180.0/np.pi)

    print (theta1*(180.0/np.pi), gt_angle, hyp_angle, file=sys.stderr)
    return abs(gt_angle - hyp_angle)
    
          
if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)
     
    input_dir     = args['<inputDir>']
    gt_query_file = args['<gtQueryFile>']
    
    gt_pic_frames = args['--gtPicFrames']
    gt_text_boxes = args['--gtTextBoxes']
    ima_dir       = args['--imaDir']
    out_dir       = args['--outDir']
    aug_list_file = args['--augList']

    k_val         = args['--kVal'].rsplit(sep=',')
    k_val         = [int(kv) for kv in k_val]

    
    print ('inputDir        = {}'.format(input_dir), file=sys.stderr)
    print ('gtQueryFile     = {}'.format(gt_query_file), file=sys.stderr)
    
    print ('gtPicFrames     = {}'.format(gt_pic_frames), file=sys.stderr)
    print ('gtTextBoxes     = {}'.format(gt_text_boxes), file=sys.stderr)
    print ('imaDir          = {}'.format(ima_dir), file=sys.stderr)
    print ('kVal            = {}'.format(k_val), file=sys.stderr)
    print ('outDir          = {}'.format(out_dir), file=sys.stderr)
    print ('augList         = {}'.format(aug_list_file), file=sys.stderr)

    # Query GT. Must be always present
    with open(gt_query_file, 'rb') as gtfd:
        gtquery_list = pickle.load(gtfd)

    if gt_text_boxes != 'None':
        with open(gt_text_boxes, 'rb') as gttb:
            gt_tboxes = pickle.load(gttb)

    aug1,aug2, aug3 = [],[],[]
    if aug_list_file != 'None':
        with open(aug_list_file, 'rb') as al:
            aug_list = pickle.load(al)

        aug1 = [ii for ii, val in enumerate(aug_list) if val == 'None'] 
        aug2 = [ii for ii, val in enumerate(aug_list) if val == 'UnnamedImpulseNoise'] 
        aug3 = [ii for ii, val in enumerate(aug_list) if val == 'None-MultiplyHue'] 
        
    # Use full path to results in students home (i.e. /home/mcv02/m1-results/week5/QST1')        
    path_list = os.path.normpath(input_dir).split(os.path.sep)
    team = path_list[1]
    week = path_list[2]

    # Initialize
    total_text_dist = -1
    avg_text_dist   = -1
    
    images_list = []
    if ima_dir != "None":
        images_list = sorted(fnmatch.filter(os.listdir(ima_dir), '*.png'))
        images_list = ['{}/{}'.format(ima_dir,xx) for xx in images_list]

        text_list   = sorted(fnmatch.filter(os.listdir(ima_dir), '*.txt'))
        text_list   = ['{}/{}'.format(ima_dir,xx) for xx in text_list]
        
    for dirName, subdirList, fileList in os.walk(input_dir):
        for fname in fileList:
            if fname == 'result.pkl' or fname == 'results.pkl': 
                method = os.path.normpath(dirName).split(os.path.sep)[-1]
                hypo_name = '{}/{}'.format(dirName, fname)

                masks_list = sorted(fnmatch.filter(os.listdir(dirName), '*.png'))
                masks_list = ['{}/{}'.format(dirName,xx) for xx in masks_list]

                hyp_txt_list = sorted(fnmatch.filter(os.listdir(dirName), '*.txt'))
                hyp_txt_list = ['{}/{}'.format(dirName,xx) for xx in hyp_txt_list]

                
                with open (hypo_name, 'rb') as fd:
                    hypo = pickle.load(fd)

                if type(hypo[0][0]) is float or type(hypo[0][0]) is np.float64 or type(hypo[0][0]) is np.int64:
                    print ('WARNING: wrong type!!', file=sys.stderr)
                    hypo_tmp = []
                    for corrs in hypo:
                        corrs = [int(xx) for xx in corrs]
                            
                        hypo_tmp.append(corrs)
                    hypo = hypo_tmp

                # Remove empty third level lists, like [[[2,1,4,5],[]], ...]
                if type(hypo[0][0]) is list:
                    hypo_tmp = [[hy[0]] if (len(hy) > 1 and len(hy[1]) == 0) else hy for hy in hypo]
                    hypo = hypo_tmp
                    
                for kv in k_val:
                    score = compute_mapk(gtquery_list, hypo, kv)
                    print ('Q,K{},{},{},{:.3f}'.format(kv, team, method, score))

                if aug_list_file != 'None':
                    gtql_none  = itemgetter(*aug1)(gtquery_list)
                    hypo_none  = itemgetter(*aug1)(hypo)
                    gtql_noise = itemgetter(*aug2)(gtquery_list)
                    hypo_noise = itemgetter(*aug2)(hypo)
                    gtql_color = itemgetter(*aug3)(gtquery_list)
                    hypo_color = itemgetter(*aug3)(hypo)

                    for kv in k_val:
                        score_none  = compute_mapk(gtql_none,  hypo_none,  kv)
                        score_noise = compute_mapk(gtql_noise, hypo_noise, kv)
                        score_color = compute_mapk(gtql_color, hypo_color, kv)
                        print ('QN,K{},{}, {}, {:.3f}'.format(kv,team, method, score_none))
                        print ('QS,K{},{}, {}, {:.3f}'.format(kv,team, method, score_noise))
                        print ('QC,K{},{}, {}, {:.3f}'.format(kv,team, method, score_color))

                if week != 'week5':
                    if len(masks_list) == len(gtquery_list):
                        pixelPrecision, pixelSensitivity, pixelF1 = score_pixel_masks(masks_list, images_list)
                        #print ('Team {} background, method {} : Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}\n'.format(team, method, pixelPrecision, pixelSensitivity, pixelF1))      
                        print ('PM,{},{},{:.2f}, {:.2f},{:.2f}'.format(team, method, pixelPrecision, pixelSensitivity, pixelF1))      

                # Evaluate OCR text
                if len(hyp_txt_list) == len(gtquery_list) and len(hyp_txt_list) == len(text_list):
                    total_text_dist =  0
                    checked_lines   =  0
                    total_lines     =  0
                    for ii in range(len(text_list)):
                        text_lines = []
                        for line in open(text_list[ii]).read().splitlines():
                            text_lines.append(line.split('\'')[1])
                        total_lines = total_lines + len(text_lines)
                        with open(hyp_txt_list[ii]) as fhy:
                            hypo_lines = fhy.read().splitlines()
                        for jj in range(min(len(text_lines),len(hypo_lines))):
                            current_dist    = distance(text_lines[jj], hypo_lines[jj])
                            checked_lines   = checked_lines + 1
                            total_text_dist = total_text_dist + current_dist
                    avg_text_dist = total_text_dist/checked_lines if checked_lines != 0 else -1
                    print ('TD,{},{},{},{},{}'.format(team, method, avg_text_dist, total_lines, checked_lines))
                    
            # Evaluate text bbox using bbox-based measures
            if fname == 'text_boxes.pkl' and gt_text_boxes != 'None':
                method = os.path.normpath(dirName).split(os.path.sep)[-1]
                text_hypo_name = '{}/{}'.format(dirName, fname)
                with open (text_hypo_name, 'rb') as fd:
                    hypo_tboxes = pickle.load(fd)

                    hypo_tboxes = [list(tb) for tb in hypo_tboxes]
                    if list_depth(hypo_tboxes) == 2:
                        hypo_tboxes = add_list_level2(hypo_tboxes.copy())

                bbox_precision, bbox_recall, bbox_f1, _, iou = score_bboxes(hypo_tboxes, gt_tboxes)
                print ('TB,{},{},{:.2f}, {:.2f},{:.2f},{:.2f}'.format(team, method, bbox_precision, bbox_recall, bbox_f1, iou))      

    # Evaluate painting boxes using angular error and pixel-based precision & recall (W5)
    # Pixel based measures are used because the bboxes can be rotated
    # BBoxes are given by specifying the 4 corner points
    # This code can be improved and simplified using shapely
    if gt_pic_frames != 'None':
        gtpic_list = []
        with open(gt_pic_frames, 'rb') as gtpf:
            gtpic_list = pickle.load(gtpf)
        
        images_list = sorted(fnmatch.filter(os.listdir(ima_dir), '*.jpg'))

        for dirName, subdirList, fileList in os.walk(input_dir):
            #print('Found directory: %s' % dirName)
            for fname in fileList:
                if fname == 'frames.pkl':
                    method = os.path.normpath(dirName).split(os.path.sep)[-1]
                    hypo_name = '{}/{}'.format(dirName, fname)

                    with open(hypo_name, 'rb') as fd:
                        hypo_pic = pickle.load(fd)
                        if len(hypo_pic) != len(gtpic_list) or len(hypo_pic) != len(images_list):
                            print ('Error: length of frame boxes positions does not match ground truth')
                            print (len(hypo_pic), len(gtpic_list), len(images_list))
                            sys.exit()

                            
                        avg_pic_iou       = 0.0
                        avg_angular_error = 0.0
                        count = 0
                        '''
                        for gt,hy in zip(gtpic_list,hypo_pic):
                            # GT and Hypo may have different number of elements.
                            # Create optimal assignment using Munkres 
                            gt_frames    = [Polygon(x[1]) for x in gt]
                            gt_centroids = [(fr.centroid.x,fr.centroid.y) for fr in gt_frames]
                            gt_angles    = [x[0] for x in gt]
                            hy_frames    = [Polygon(x[1]) for x in hy]
                            hy_centroids = [(fr.centroid.x,fr.centroid.y) for fr in hy_frames]
                            hy_angles    = [x[0] for x in hy]
                            
                            cmat = scipy.spatial.distance.cdist(gt_centroids, hy_centroids)
                            m = Munkres()
                            indexes = m.compute(cmat)
                            
                            # Incomplete, needs to be finished ...
                            
                            gt_angle = gt[0]
                            hy_angle = hy[0]
                            gt_frame = Polygon(gt[1])
                            hy_frame = Polygon(hy[1])
                            avg_pic_iou += gt_frame.intersection(hy_frame).area / gt_frame.union(hy_frame).area
                            avg_angular_error += abs(gt_angle - hy_angle)
                        '''
                            
                        for jj, ima_name in enumerate(images_list):
                            gt_frames    = [x[1] for x in gtpic_list[jj]]
                            hy_frames    = [x[1] for x in hypo_pic[jj]]
                            gt_mask  = create_masks(ima_dir, ima_name, gt_frames)
                            hy_mask  = create_masks(ima_dir, ima_name, hy_frames)

                            #gt_mask  = create_mask(ima_dir, ima_name, gtpic_list[jj][1])
                            #hyp_mask = create_mask(ima_dir, ima_name, hypo_pic[jj][1])

                            if out_dir != None:
                                out_name = '{}/{}_gt_mask.png'.format(out_dir, os.path.splitext(ima_name)[0])
                                cv2.imwrite(out_name, gt_mask)
                                out_name = '{}/{}_hyp_mask.png'.format(out_dir, os.path.splitext(ima_name)[0])
                                cv2.imwrite(out_name, hy_mask)
                            avg_pic_iou       += jaccard_score(gt_mask.flatten(), hy_mask.flatten())
                            #avg_angular_error += angular_error_boxes (gtpic_list[jj], hypo_pic[jj][1])
                            #avg_angular_error += angular_error_box_angle (gtpic_list[jj], float(hypo_pic[jj][0]))

                            gt_angles    = [x[0] for x in gtpic_list[jj]]
                            hy_angles    = [x[0] for x in hypo_pic[jj]]
                            common_vals = min(len(gt_angles), len(hy_angles))
                            for kk in range(common_vals):
                                gta = gt_angles[kk] * np.pi / 180
                                hya = hy_angles[kk] * np.pi / 180
                                
                                v1 = [abs(np.cos(gta)),np.sin(gta)]
                                v2 = [abs(np.cos(hya)),np.sin(hya)]
                                ang_error = abs(np.arccos(np.dot(v1,v2)) * 180 / np.pi)
                                avg_angular_error += ang_error
                                
                                #avg_angular_error += abs(gt_angles[kk] - hy_angles[kk])
                                count = count + 1
                                            
                        avg_pic_iou       /= len(hypo_pic)
                        avg_angular_error /= count
                        print ('F,{}, {}, {:.3f}, {:.3f}'.format(team, method, avg_angular_error, avg_pic_iou))
