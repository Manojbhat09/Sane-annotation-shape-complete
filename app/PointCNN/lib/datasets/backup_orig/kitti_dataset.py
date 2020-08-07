import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import argoverse
#from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import lib.datasets.ground_segmentation as gs
from pyntcloud import PyntCloud
import random
import copy


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir,"sample/argoverse/lidar")

        lidarfile_list = os.listdir(self.imageset_dir)
        
        self.image_idx_list = [x.split('.')[0] for x in lidarfile_list]
        self.num_sample = self.image_idx_list.__len__()

        
        self.lidar_dir = self.imageset_dir
        
        self.argo_to_kitti = np.array([[6.927964e-03, -9.999722e-01, -2.757829e-03],
                                       [-1.162982e-03, 2.749836e-03, -9.999955e-01],
                                       [9.999753e-01, 6.931141e-03, -1.143899e-03]])

        self.ground_removal = True
        
    def get_lidar(self,idx):
        lidar_file = os.path.join(self.lidar_dir,"%06d.bin"%idx)
        assert os.path.exists(lidar_file)
        
        

        pts_lidar = np.fromfile(lidar_file).reshape(-1,3)[:,:3]
        #x = copy.deepcopy(pts_lidar[:,1])
        #y = copy.deepcopy(pts_lidar[:,0])
        #pts_lidar[:,0] = x
        #pts_lidar[:,1] = y
        
        if self.ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        
        pts_lidar = np.dot(self.argo_to_kitti,pts_lidar.T).T
        
        return pts_lidar
