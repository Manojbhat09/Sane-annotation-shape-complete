import os
import sys
import numpy as np
import os.path
import time
import random
from preprocess import preprocess
from pyntcloud import PyntCloud


pythonApp = "python "
script_seg = "eval_rcnn.py --cfg_file cfgs/argo_config_sampling_trainfull.yaml --rcnn_ckpt checkpoint_epoch_40.pth --rpn_ckpt checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn --test"

def check_succes_sys_call(_command, file_check):
    i = 0
    while os.path.isfile(file_check) == False :
        print(i, os.path.isfile(file_check), file_check)
        os.system(" killall python3.6 & ")
        os.system(_command)  
        if(i>0):
            time.sleep(random.randint(5,30))
        i = i + 1
    return True

def get_pointcnn_labels_axcrf(filename):
    return None

def get_pointcnn_labels(filename, settingsControls, ground_removed=False, foreground_only=True):
    

    seg_dir = "PointCNN/output/rcnn/argo_config_sampling_trainfull/eval/epoch_no_number/sample/test_mode/rpn_result/data"
    drivename, fname = filename.split("/")

    seg_file = seg_dir + "/" + fname + ".npy" 

    if not os.path.isfile(seg_file):
        # execute pointrcnn
        # currently have to run on the whole files to generate corresponding out
        # will be replaced by only inferencing on this specific file
        preprocess()
    
    
    seg_points = np.load(seg_file).astype('float32').reshape(-1, 5)     
    bounded_indices = np.load(seg_file).reshape(-1, 5)[:, -1].flatten()
   
    if(foreground_only):
        bounded_indices =   (bounded_indices == 1.0 ).nonzero()[0]
        return bounded_indices.tolist()
    else:
        return bounded_indices
    

