#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Use this script to test your submission. Do not look at the results, as they are computed with fake annotations and masks.
This is just to see if there's any problem with files, paths, permissions, etc.
If you find a bug, please report it to ramon.morros@upc.edu 

Usage:
  test_submission.py <weekNumber> <teamNumber> <baseDir> <testDir> 
  test_submission.py -h | --help
  ------------------ 
  baseDir        Base folder with your results (must contain subdirectories in the form week?/QST1/method1/, week?/QST2/method1, etc.)
  testDir        Directory with the test images & masks 

Options:
"""



import fnmatch
import os
import sys
import pickle
import imageio
import numpy as np
from docopt import docopt


# Compute the depth of a list (of lists (of lists ...) ...)
# Empty list not allowed!!
#https://stackoverflow.com/questions/6039103/counting-depth-or-the-deepest-level-a-nested-list-goes-to
list_depth = lambda L: isinstance(L, list) and max(map(list_depth, L))+1




def check_query_file(hypo_name, week, k_val, num_queries, qsn):
    if not os.path.isfile(hypo_name):
        print ('File {} not found!'.format(hypo_name))
        sys.exit()
    with open(hypo_name, 'rb') as fd:
        hypo = pickle.load(fd)

        if type(hypo) is not list:
            print ('File {} must contain a list!'.format(hypo_name))
            sys.exit()

        ld = list_depth(hypo)
        if week == 1 and ld != 2:
            print ('File {} list must have two levels!'.format(hypo_name))
            sys.exit()

        if week > 1 and ld != 3:
            print ('File {} list must have three levels!'.format(hypo_name))
            sys.exit()
                
        elem = hypo[0][0] if week == 1 else hypo[0][0][0]
                
        if type(elem) is not int:
            print ('File {} list must contain integers, not {}'.format(hypo_name,type(elem)))
            sys.exit()
            
        if (len(hypo) != num_queries):
            print("HYPO NAME ", hypo_name, "NUM QUERIES ", num_queries)
            with open(hypo_name, "rb") as fp:   # Unpickling
                temp = pickle.load(fp)
            print(len(temp))
            print ('File {} should contain {} queries.'.format(hypo_name), num_queries)
            sys.exit()

        for hyp in hypo:
            if week == 1:
                if len(hyp) != k_val:
                    pass
            else:
                if qsn == 1:
                    if len(hyp) != 1:
                        print ('In file {}, only one result per image should be given  !'.format(hypo_name))
                        sys.exit()
                for hy in hyp:
                    if week < 4:
                        if len(hy) != k_val:
                            print ('In file {}, {} results per query must be given!'.format(hypo_name, k_val))
                            sys.exit()
                    else:
                        if hy[0] == -1 and len(hy) != 1:
                            print ('In file {}, if there is a -1, no further results must be given in this query'.format(hypo_name))
                            sys.exit()
                        if hy[0] != -1 and len(hy) != k_val:
                            print ('In file {}, {} results per query or [-1] must be given!'.format(hypo_name, k_val))
                            sys.exit()


                            
def check_text_box_file(tb_name, week, num_queries, qsn):

    if not os.path.isfile(tb_name):
        print ('File {} not found!'.format(tb_name))
        sys.exit()

    with open(tb_name, 'rb') as fd:
        tb = pickle.load(fd)

        if type(tb) is not list:
            print ('File {} must contain a list!'.format(tb_name))
            sys.exit()

        if (len(tb) != num_queries):
            print ('File {} contains an incorrect number of queries!'.format(tb_name))
            sys.exit()
    
        if list_depth(tb) != 3:
            print ('File {} list must have three levels!'.format(tb_name))
            sys.exit()

        for tt in tb:
            if qsn == 1:
                if len(tt) != 1:
                    print ('In file {}, only one result per image should be given  !'.format(tb_name))
                    sys.exit()
            for tti in tt:
                if len(tti) != 4:
                    print ('In file {}, text boxes should contain 4 coordinates (int values)!'.format(tb_name))
                    sys.exit()
    
def check_frames_file(frames_name, num_queries):
    if not os.path.isfile(frames_name):
        print ('File {} not found!'.format(frames_name))
        sys.exit()

    with open(frames_name, 'rb') as fd:
        frams = pickle.load(fd)

        if type(frams) is not list:
            print ('File {} must contain a list!'.format(frames_name))
            sys.exit()

        if (len(frams) != num_queries):
            print ('File {} contains an incorrect number of queries!'.format(frames_name))
            sys.exit()
    
        if list_depth(frams) != 5:
            print ('File {} list must have three levels!'.format(frames_name))
            sys.exit()

        for fram in frams:
            if qsn == 1:
                if len(fram) != 1:
                    print ('In file {}, only one result per image should be given  !'.format(frames_name))
                    sys.exit()
            for fr in fram:
                if len(fr) != 2:
                    print ('In file {}, format must be [alpha,[[px1,py1],[px2,py2],[px3,py3],[px4,py4]]'.format(frames_name))
                    sys.exit()
                if not isinstance(fr[0], (float,np.float32)):
                    print ('In file {}, angle must be given as floating point'.format(frames_name))
                    sys.exit()
                if len(fr[1]) != 4:
                    print ('In file {}, format must be [alpha,[[px1,py1],[px2,py2],[px3,py3],px4,py4]]'.format(frames_name))
                    sys.exit()
                for pt in fr[1]:
                    if len(pt) != 2:
                        print ('In file {}, format must be [alpha,[[px1,py1],[px2,py2],[px3,py3],px4,py4]]'.format(frames_name))
                        sys.exit()
                
                    

                    
if __name__ == '__main__':
    # read arguments
    args = docopt(__doc__)

    week      = int(args['<weekNumber>'])
    team      = int(args['<teamNumber>'])
    base_dir  = args['<baseDir>']
    
    # This folder contains fake masks and text annotations. 
    test_dir = args['<testDir>']

    test_qs1 = '{}/W{}/QST1/gt_corresps.pkl'.format(test_dir,week)
    with open(test_qs1, 'rb') as fd:
        gt_query = pickle.load(fd)
    num_queries = len(gt_query)

    
    k_val = 10

    
    # This folder contains your results: mask images and window list pkl files. 
    results_dir = '{}/week{}'.format(base_dir, week)

    # Test Query set 1:
    qs1_dir = '{}/QST1'.format(results_dir)
    if not os.path.isdir(qs1_dir):
        print ('{} does not exist!'.format(qs1_dir))
        sys.exit()

    # 1 stands for simple and 2 for complex    
    qsn = 1 if week < 4 else 2  # Starting at week 4, there is only a query set and is the complex one
    
    # List all folders (corresponding to the different methods) in the results directory
    methods = next(os.walk(qs1_dir))[1]
    for method in methods:

        # Correspondences Hypotesis file
        hypo_name = '{}/{}/result.pkl'.format(qs1_dir, method)
        check_query_file(hypo_name, week, k_val, num_queries, qsn)

        # Text boxes Hypotesis file
        if week > 1 and week < 4:
            tb_name = '{}/{}/text_boxes.pkl'.format(qs1_dir, method)
            check_text_box_file(tb_name, week, num_queries, qsn)

        if week > 2 and week < 5:  # txt files
            result_txt     = sorted(fnmatch.filter(os.listdir('{}/{}'.format(qs1_dir, method)), '*.txt'))
            result_txt_num = len(result_txt)
            if result_txt_num != num_queries:
                print (f'Method {method} : {result_txt_num} result txt files found but there are {num_queries} test txt files', file = sys.stderr) 
                sys.exit()
            
        # Frames 
        if week == 5:
            # Frames Hypotesis file
            frame_name = '{}/{}/frames.pkl'.format(qs1_dir, method)
            check_frames_file(frame_name, num_queries)

        print ('Submission for QST1 {} seems OK'.format(method))
    
    # From week 4, only one test set
    if week > 3:
        sys.exit()


    ##################    
    # Test Query set 2:
    ##################
    
    test_dir_qs2 = '{}/W{}/QST2'.format(test_dir,week)
    with open('{}/gt_corresps.pkl'.format(test_dir_qs2), 'rb') as fd1:
        gt_query = pickle.load(fd1)
    num_queries = len(gt_query)

    qs2_dir = '{}/QST2'.format(results_dir)
    if not os.path.isdir(qs2_dir):
        print ('{} does not exist!'.format(qs2_dir))
        sys.exit()

    qsn = 2
    # List all folders (corresponding to the different methods) in the results directory
    methods = next(os.walk(qs2_dir))[1]

    # Load mask names in the given directory
    test_masks     = sorted(fnmatch.filter(os.listdir(test_dir_qs2), '*.png'))

    for method in methods:

        # Correspondences Hypotesis file
        hypo_name = '{}/{}/result.pkl'.format(qs2_dir, method)
        check_query_file(hypo_name, week, k_val, num_queries, qsn)

        # Text boxes Hypotesis file
        if week > 1 and week < 4:
            tb_name = '{}/{}/text_boxes.pkl'.format(qs2_dir, method)
            check_text_box_file(tb_name, week, num_queries, qsn)


        if week > 2 and week < 5:  # txt files
            result_txt     = sorted(fnmatch.filter(os.listdir('{}/{}'.format(qs2_dir, method)), '*.txt'))
            result_txt_num = len(result_txt)
            if result_txt_num != num_queries:
                print ('Method {} : {} result txt files found but there are {} test txt files'.format(method, result_txt_num, num_queries), file = sys.stderr) 
                sys.exit()
            
            
        # Read masks (if any)
        result_masks     = sorted(fnmatch.filter(os.listdir('{}/{}'.format(qs2_dir, method)), '*.png'))
        result_masks_num = len(result_masks)

        if result_masks_num != num_queries:
            print ('Method {} : {} result masks found but there are {} test masks'.format(method, result_masks_num, num_queries), file = sys.stderr) 
            sys.exit()

        for ii in range(len(result_masks)):

            # Read mask file
            candidate_masks_name = '{}/{}/{}'.format(qs2_dir, method, result_masks[ii])
            #print ('File: {}'.format(candidate_masks_name), file = sys.stderr)

            pixelCandidates = imageio.imread(candidate_masks_name)>0
            if len(pixelCandidates.shape) == 3:
                pixelCandidates = pixelCandidates[:,:,0]
            
            # Accumulate pixel performance of the current image %%%%%%%%%%%%%%%%%
            name, ext = os.path.splitext(test_masks[ii])
            gt_mask_name = '{}/{}.png'.format(test_dir_qs2, name)

            pixelAnnotation = imageio.imread(gt_mask_name)>0
            if len(pixelAnnotation.shape) == 3:
                pixelAnnotation = pixelAnnotation[:,:,0]

            if pixelAnnotation.shape != pixelCandidates.shape:
                print ('Error: hypothesis and  GT masks do not match!')
                print (pixelAnnotation.shape, pixelCandidates.shape)
                sys.exit()

            '''
            if window_evaluation == 1:
                # Read .pkl file
            
                name_r, ext_r = os.path.splitext(result_masks[ii])
                pkl_name      = '{}/{}/{}.pkl'.format(results_dir, method, name_r)
                

                with open(pkl_name, "rb") as fp:   # Unpickling
                    windowCandidates = pickle.load(fp)

                gt_annotations_name = '{}/gt/gt.{}.txt'.format(test_dir_qs2, name)
                windowAnnotations = load_annotations(gt_annotations_name)
            '''
        print ('Submission for QST2 {} seems OK'.format(method))

