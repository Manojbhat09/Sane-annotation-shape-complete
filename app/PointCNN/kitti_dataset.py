import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import lib.datasets.ground_segmentation as gs
from pyntcloud import PyntCloud
import random


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = (self.split == 'test')
        self.lidar_pathlist = []
        self.label_pathlist = []
        
        if split == 'train':
            for i in np.arange(1,5):
                self.imageset_dir = os.path.join(root_dir,split+str(i))
                data_loader = ArgoverseTrackingLoader(os.path.join(root_dir,split+str(i)))
                self.log_list = data_loader.log_list
                for log in self.log_list:
                    self.lidar_pathlist.extend(data_loader.get(log).lidar_list)
                    self.label_pathlist.extend(data_loader.get(log).label_list)
                print(len(self.lidar_pathlist))
        else:
            self.imageset_dir = os.path.join(root_dir,split)
            data_loader = ArgoverseTrackingLoader(os.path.join(root_dir,split))
            self.lidar_pathlist.extend(data_loader.lidar_list)
            self.label_pathlist.extend(data_loader.label_list)
        
        
        self.calib_file = data_loader.calib_filename
        
        assert len(self.lidar_pathlist) == len(self.label_pathlist)
        #z = list(zip(self.lidar_pathlist, self.label_pathlist))
        #random.shuffle(z)
        #self.lidar_pathlist[:], self.label_pathlist[:] = zip(*z)

        self.num_sample = len(self.lidar_pathlist)
        self.image_idx_list = np.arange(self.num_sample)
        
        self.argo_to_kitti = np.array([[6.927964e-03, -9.999722e-01, -2.757829e-03],
                                       [-1.162982e-03, 2.749836e-03, -9.999955e-01],
                                       [9.999753e-01, 6.931141e-03, -1.143899e-03]])

        self.ground_removal = True
        
        self.image_dir = os.path.join('/data/')
        self.lidar_dir = os.path.join('/data/')
        self.calib_dir = os.path.join('/data/')
        self.label_dir = os.path.join('/data/')
        
    def get_lidar(self,idx):
        lidar_file = self.lidar_pathlist[idx]
        assert os.path.exists(lidar_file)
        
        
        data = PyntCloud.from_file(lidar_file)
        x = np.array(data.points.x)[:, np.newaxis]
        y = np.array(data.points.y)[:, np.newaxis]
        z = np.array(data.points.z)[:, np.newaxis]
        pts_lidar = np.concatenate([x,y,z], axis = 1)
        
        if self.ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        
        pts_lidar = np.dot(self.argo_to_kitti,pts_lidar.T).T
        
        return pts_lidar
        
        
    def get_label(self,idx):
        
        label_file = self.label_pathlist[idx]
        assert os.path.exists(label_file)
        
        return kitti_utils.get_objects_from_label(label_file)
    
    def get_calib(self, idx):
        # Single Calibration File for All, One Calib to Rule them all
        assert os.path.exists(self.calib_file)
        return calibration.Calibration(self.calib_file)

    def get_image_shape(self, idx):
        return 1200, 1920, 3
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
