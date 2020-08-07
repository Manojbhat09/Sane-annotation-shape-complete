import numpy as np
import os
import pickle
import torch

from lib.datasets.kitti_dataset import KittiDataset
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg


class KittiRCNNDataset(KittiDataset):
    def __init__(self, root_dir, npoints=65536, split='train', classes='VEHICLE', mode='TRAIN', random_select=True,
                 logger=None, rcnn_training_roi_dir=None, rcnn_training_feature_dir=None, rcnn_eval_roi_dir=None,
                 rcnn_eval_feature_dir=None, gt_database_dir=None):
        
        super().__init__(root_dir=root_dir, split=split)
        if classes == 'VEHICLE':
            self.classes = ('Background', 'VEHICLE')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

        self.num_class = self.classes.__len__()
        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger

        if split == 'train_aug':
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')
        else:
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')

        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir

        self.gt_database = None

        if not self.random_select:
            self.logger.warning('random select is False')

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if cfg.RPN.ENABLED:
            if mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                self.sample_id_list = [int(sample_id) for sample_id in self.lidar_idx_list]
                self.logger.info('Load testing samples from %s' % self.lidar_dir)
                self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
        elif cfg.RCNN.ENABLED:
            for idx in range(0, self.num_sample):
                sample_id = int(self.lidar_idx_list[idx])
                obj_list = self.filtrate_objects(self.get_label(sample_id))
                if len(obj_list) == 0:
                    # logger.info('No gt classes: %06d' % sample_id)
                    continue
                self.sample_id_list.append(sample_id)

            print('Done: filter %s results for rcnn training: %d / %d\n' %
                  (self.mode, len(self.sample_id_list), len(self.lidar_idx_list)))

    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """

        # num_sample --> variable --> Total lidar frames
        # lidar_idx_list --> list --> ID of lidar frames
        # sample_id_list --> list --> Valid Lidar frames IDs is stored in self.sample_id_list

        for idx in range(0, self.num_sample):
            sample_id = int(self.lidar_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                # self.logger.info('No gt classes: %06d' % sample_id)
                continue
            self.sample_id_list.append(sample_id)

        #self.logger.info('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list),
                                                                 #len(self.lidar_idx_list)))

    def get_label(self, idx):
        # Check for Kitti Utils
        assert os.path.exists(self.label_pathlist[idx])
        return kitti_utils.get_objects_from_label(self.label_pathlist[idx])

    def get_lidar(self, idx):
        return super().get_lidar(idx)

    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
        if cfg.RCNN.USE_SEG_SCORE:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list of Object class instances
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'VEHICLE' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:  # rm Van, 20180928
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    @staticmethod
    def get_valid_flag(pts_rect):
        """
        Valid point should be in the lidar (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            pts_valid_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
        return pts_valid_flag

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_id_list)
            else:
                return len(self.lidar_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                if cfg.RCNN.ROI_SAMPLE_JIT:
                    return self.get_rcnn_sample_jit(index)
                else:
                    return self.get_rcnn_training_sample_batch(index)
            else:
                return self.get_proposal_from_file(index)
        else:
            raise NotImplementedError

    def get_rpn_sample(self, index):
        sample_id = int(self.sample_id_list[index])
        pts_lidar = self.get_lidar(sample_id)
        
        # get valid point (projected points should be in lidar)
        pts_orig = pts_lidar[:, 0:3]
        valid_mask = self.get_valid_flag(pts_orig)
        
        pts_rect = pts_orig[valid_mask]
        pts_left = pts_orig[~valid_mask]

        pts_intensity = np.arange(pts_lidar.shape[0])
        
        
        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                
                pts_near_flag = np.abs(pts_depth) < 40.0
                
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                
                near_idxs = np.where(pts_near_flag == 1)[0]
                
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=True)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=True)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}

        rem_index = np.setdiff1d(np.arange(pts_rect.shape[0]),choice)

        if self.mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
                print(pts_input.shape,rem_index.shape,pts_left.shape)
                sample_info['pts_input'] = pts_input
                sample_info['rem_pts'] = pts_left
                sample_info['pts_rect']  = ret_pts_rect
                sample_info['pts_features'] = ret_pts_features
                return sample_info

        # data augmentation Removed
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
       

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        if cfg.RPN.FIXED:
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = aug_pts_rect
            sample_info['pts_features'] = ret_pts_features
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            return sample_info

        # generate training labels
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d)
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['rem_pts'] = np.vstack((pts_left, pts_rect[rem_index,:]))
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        
        return sample_info

    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d):
        # Heavily dependent on Kitti utils module, which assumes Kitti input
        # x,y,z,ry,h,w,l
        # Generates Classification Label --> Foreground / Background
        # Generates Regression Label --> x,y,z,h,w,l,ry

        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    def rotate_box3d_along_y(self, box3d, rot_angle):
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry

        box3d = kitti_utils.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta

        return box3d

    def get_rcnn_sample_info(self, roi_info):
        sample_id, gt_box3d = roi_info['sample_id'], roi_info['gt_box3d']
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.rpn_feature_list[sample_id]

        # augmentation original roi by adding noise
        roi_box3d = self.aug_roi_by_noise(roi_info)

        # point cloud pooling based on roi_box3d
        pooled_boxes3d = kitti_utils.enlarge_box3d(roi_box3d.reshape(1, 7), cfg.RCNN.POOL_EXTRA_WIDTH)

        boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(rpn_xyz),
                                                                 torch.from_numpy(pooled_boxes3d))
        pt_mask_flag = (boxes_pts_mask_list[0].numpy() == 1)
        cur_pts = rpn_xyz[pt_mask_flag].astype(np.float32)

        # data augmentation
        aug_pts = cur_pts.copy()
        aug_gt_box3d = gt_box3d.copy().astype(np.float32)
        aug_roi_box3d = roi_box3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            # calculate alpha by ry
            temp_boxes3d = np.concatenate([aug_roi_box3d.reshape(1, 7), aug_gt_box3d.reshape(1, 7)], axis=0)
            temp_x, temp_z, temp_ry = temp_boxes3d[:, 0], temp_boxes3d[:, 2], temp_boxes3d[:, 6]
            temp_beta = np.arctan2(temp_z, temp_x).astype(np.float64)
            temp_alpha = -np.sign(temp_beta) * np.pi / 2 + temp_beta + temp_ry

            # data augmentation
            aug_pts, aug_boxes3d, aug_method = self.data_augmentation(aug_pts, temp_boxes3d, temp_alpha, mustaug=True, stage=2)
            aug_roi_box3d, aug_gt_box3d = aug_boxes3d[0], aug_boxes3d[1]
            aug_gt_box3d = aug_gt_box3d.astype(gt_box3d.dtype)

        # Pool input points
        valid_mask = 1  # whether the input is valid

        if aug_pts.shape[0] == 0:
            pts_features = np.zeros((1, 128), dtype=np.float32)
            input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            pts_input = np.zeros((1, input_channel), dtype=np.float32)
            valid_mask = 0
        else:
            pts_features = rpn_features[pt_mask_flag].astype(np.float32)
            pts_intensity = rpn_intensity[pt_mask_flag].astype(np.float32)

            pts_input_list = [aug_pts, pts_intensity.reshape(-1, 1)]
            if cfg.RCNN.USE_INTENSITY:
                pts_input_list = [aug_pts, pts_intensity.reshape(-1, 1)]
            else:
                pts_input_list = [aug_pts]

            if cfg.RCNN.USE_MASK:
                if cfg.RCNN.MASK_TYPE == 'seg':
                    pts_mask = seg_mask[pt_mask_flag].astype(np.float32)
                elif cfg.RCNN.MASK_TYPE == 'roi':
                    pts_mask = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(aug_pts),
                                                                  torch.from_numpy(aug_roi_box3d.reshape(1, 7)))
                    pts_mask = (pts_mask[0].numpy() == 1).astype(np.float32)
                else:
                    raise NotImplementedError

                pts_input_list.append(pts_mask.reshape(-1, 1))

            if cfg.RCNN.USE_DEPTH:
                pts_depth = np.linalg.norm(aug_pts, axis=1, ord=2)
                pts_depth_norm = (pts_depth / 70.0) - 0.5
                pts_input_list.append(pts_depth_norm.reshape(-1, 1))

            pts_input = np.concatenate(pts_input_list, axis=1)  # (N, C)

        aug_gt_corners = kitti_utils.boxes3d_to_corners3d(aug_gt_box3d.reshape(-1, 7))
        aug_roi_corners = kitti_utils.boxes3d_to_corners3d(aug_roi_box3d.reshape(-1, 7))
        iou3d = kitti_utils.get_iou3d(aug_roi_corners, aug_gt_corners)
        cur_iou = iou3d[0][0]

        # regression valid mask
        reg_valid_mask = 1 if cur_iou >= cfg.RCNN.REG_FG_THRESH and valid_mask == 1 else 0

        # classification label
        cls_label = 1 if cur_iou > cfg.RCNN.CLS_FG_THRESH else 0
        if cfg.RCNN.CLS_BG_THRESH < cur_iou < cfg.RCNN.CLS_FG_THRESH or valid_mask == 0:
            cls_label = -1

        # canonical transform and sampling
        pts_input_ct, gt_box3d_ct = self.canonical_transform(pts_input, aug_roi_box3d, aug_gt_box3d)
        pts_input_ct, pts_features = self.rcnn_input_sample(pts_input_ct, pts_features)

        sample_info = {'sample_id': sample_id,
                       'pts_input': pts_input_ct,
                       'pts_features': pts_features,
                       'cls_label': cls_label,
                       'reg_valid_mask': reg_valid_mask,
                       'gt_boxes3d_ct': gt_box3d_ct,
                       'roi_boxes3d': aug_roi_box3d,
                       'roi_size': aug_roi_box3d[3:6],
                       'gt_boxes3d': aug_gt_box3d}

        return sample_info

    @staticmethod
    def canonical_transform(pts_input, roi_box3d, gt_box3d):
        roi_ry = roi_box3d[6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_box3d[0:3]
        # shift to center
        pts_input[:, [0, 1, 2]] = pts_input[:, [0, 1, 2]] - roi_center
        gt_box3d_ct = np.copy(gt_box3d)
        gt_box3d_ct[0:3] = gt_box3d_ct[0:3] - roi_center
        # rotate to the direction of head
        gt_box3d_ct = kitti_utils.rotate_pc_along_y(gt_box3d_ct.reshape(1, 7), roi_ry).reshape(7)
        gt_box3d_ct[6] = gt_box3d_ct[6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y(pts_input, roi_ry)

        return pts_input, gt_box3d_ct

    @staticmethod
    def canonical_transform_batch(pts_input, roi_boxes3d, gt_boxes3d):
        """
        :param pts_input: (N, npoints, 3 + C)
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        roi_ry = roi_boxes3d[:, 6] % (2 * np.pi)  # 0 ~ 2pi
        roi_center = roi_boxes3d[:, 0:3]
        # shift to center
        pts_input[:, :, [0, 1, 2]] = pts_input[:, :, [0, 1, 2]] - roi_center.reshape(-1, 1, 3)
        gt_boxes3d_ct = np.copy(gt_boxes3d)
        gt_boxes3d_ct[:, 0:3] = gt_boxes3d_ct[:, 0:3] - roi_center
        # rotate to the direction of head
        gt_boxes3d_ct = kitti_utils.rotate_pc_along_y_torch(torch.from_numpy(gt_boxes3d_ct.reshape(-1, 1, 7)),
                                                            torch.from_numpy(roi_ry)).numpy().reshape(-1, 7)
        gt_boxes3d_ct[:, 6] = gt_boxes3d_ct[:, 6] - roi_ry
        pts_input = kitti_utils.rotate_pc_along_y_torch(torch.from_numpy(pts_input), torch.from_numpy(roi_ry)).numpy()

        return pts_input, gt_boxes3d_ct

    @staticmethod
    def rcnn_input_sample(pts_input, pts_features):
        choice = np.random.choice(pts_input.shape[0], cfg.RCNN.NUM_POINTS, replace=True)

        if pts_input.shape[0] < cfg.RCNN.NUM_POINTS:
            choice[:pts_input.shape[0]] = np.arange(pts_input.shape[0])
            np.random.shuffle(choice)
        pts_input = pts_input[choice]
        pts_features = pts_features[choice]

        return pts_input, pts_features

    def aug_roi_by_noise(self, roi_info):
        """
        add noise to original roi to get aug_box3d
        :param roi_info:
        :return:
        """
        roi_box3d, gt_box3d = roi_info['roi_box3d'], roi_info['gt_box3d']
        original_iou = roi_info['iou3d']
        temp_iou = cnt = 0
        pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(-1, 7))
        aug_box3d = roi_box3d
        while temp_iou < pos_thresh and cnt < 10:
            if roi_info['type'] == 'gt':
                aug_box3d = self.random_aug_box3d(roi_box3d)  # GT, must random
            else:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
            aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(-1, 7))
            iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
            temp_iou = iou3d[0][0]
            cnt += 1
            if original_iou < pos_thresh:  # original bg, break
                break
        return aug_box3d

    

    def get_proposal_from_file(self, index):
        sample_id = int(self.lidar_idx_list[index])
        proposal_file = os.path.join(self.rcnn_eval_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(proposal_file)

        rpn_xyz, rpn_features, rpn_intensity, seg_mask = self.get_rpn_features(self.rcnn_eval_feature_dir, sample_id)
        pts_rect, pts_rpn_features, pts_intensity = rpn_xyz, rpn_features, rpn_intensity

        roi_box3d_list, roi_scores = [], []
        for obj in roi_obj_list:
            box3d = np.array([obj.pos[0], obj.pos[1], obj.pos[2], obj.h, obj.w, obj.l, obj.ry], dtype=np.float32)
            roi_box3d_list.append(box3d.reshape(1, 7))
            roi_scores.append(obj.score)

        roi_boxes3d = np.concatenate(roi_box3d_list, axis=0)  # (N, 7)
        roi_scores = np.array(roi_scores, dtype=np.float32)  # (N)

        if cfg.RCNN.ROI_SAMPLE_JIT:
            sample_dict = {'sample_id': sample_id,
                           'rpn_xyz': rpn_xyz,
                           'rpn_features': rpn_features,
                           'seg_mask': seg_mask,
                           'roi_boxes3d': roi_boxes3d,
                           'roi_scores': roi_scores,
                           'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

            if self.mode != 'TEST':
                gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

                roi_corners = kitti_utils.boxes3d_to_corners3d(roi_boxes3d)
                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d)
                iou3d = kitti_utils.get_iou3d(roi_corners, gt_corners)
                if gt_boxes3d.shape[0] > 0:
                    gt_iou = iou3d.max(axis=1)
                else:
                    gt_iou = np.zeros(roi_boxes3d.shape[0]).astype(np.float32)

                sample_dict['gt_boxes3d'] = gt_boxes3d
                sample_dict['gt_iou'] = gt_iou
            return sample_dict

        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [pts_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            cur_depth = np.linalg.norm(pts_rect, axis=1, ord=2)
            cur_depth_norm = (cur_depth / 70.0) - 0.5
            pts_extra_input_list.append(cur_depth_norm.reshape(-1, 1))

        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)
        pts_input, pts_features = roipool3d_utils.roipool3d_cpu(roi_boxes3d, pts_rect, pts_rpn_features,
                                                                pts_extra_input, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                sampled_pt_num=cfg.RCNN.NUM_POINTS)

        sample_dict = {'sample_id': sample_id,
                       'pts_input': pts_input,
                       'pts_features': pts_features,
                       'roi_boxes3d': roi_boxes3d,
                       'roi_scores': roi_scores,
                       'roi_size': roi_boxes3d[:, 3:6]}

        if self.mode == 'TEST':
            return sample_dict

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = np.zeros((gt_obj_list.__len__(), 7), dtype=np.float32)

        for k, obj in enumerate(gt_obj_list):
            gt_boxes3d[k, 0:3], gt_boxes3d[k, 3], gt_boxes3d[k, 4], gt_boxes3d[k, 5], gt_boxes3d[k, 6] \
                = obj.pos, obj.h, obj.w, obj.l, obj.ry

        if gt_boxes3d.__len__() == 0:
            gt_iou = np.zeros((roi_boxes3d.shape[0]), dtype=np.float32)
        else:
            roi_corners = kitti_utils.boxes3d_to_corners3d(roi_boxes3d)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d)
            iou3d = kitti_utils.get_iou3d(roi_corners, gt_corners)
            gt_iou = iou3d.max(axis=1)
        sample_dict['gt_boxes3d'] = gt_boxes3d
        sample_dict['gt_iou'] = gt_iou

        return sample_dict

    def get_rcnn_training_sample_batch(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        # calculate original iou
        iou3d = kitti_utils.get_iou3d(kitti_utils.boxes3d_to_corners3d(roi_boxes3d),
                                      kitti_utils.boxes3d_to_corners3d(gt_boxes3d))
        max_overlaps, gt_assignment = iou3d.max(axis=1), iou3d.argmax(axis=1)
        max_iou_of_gt, roi_assignment = iou3d.max(axis=0), iou3d.argmax(axis=0)
        roi_assignment = roi_assignment[max_iou_of_gt > 0].reshape(-1)

        # sample fg, easy_bg, hard_bg
        fg_rois_per_lidar = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_lidar))
        fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
        fg_inds = np.nonzero(max_overlaps >= fg_thresh)[0]
        fg_inds = np.concatenate((fg_inds, roi_assignment), axis=0)  # consider the roi which has max_overlaps with gt as fg

        easy_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO))[0]
        hard_bg_inds = np.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                  (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO))[0]

        fg_num_rois = fg_inds.size
        bg_num_rois = hard_bg_inds.size + easy_bg_inds.size

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_lidar = min(fg_rois_per_lidar, fg_num_rois)
            rand_num = np.random.permutation(fg_num_rois)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_lidar]]

            # sampling bg
            bg_rois_per_this_lidar = cfg.RCNN.ROI_PER_lidar  - fg_rois_per_this_lidar
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_lidar)

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_lidar ) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
            fg_inds = fg_inds[rand_num]
            fg_rois_per_this_lidar = cfg.RCNN.ROI_PER_lidar
            bg_rois_per_this_lidar = 0
        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_lidar = cfg.RCNN.ROI_PER_lidar
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_lidar)
            fg_rois_per_this_lidar = 0
        else:
            import pdb
            pdb.set_trace()
            raise NotImplementedError

        # augment the rois by noise
        roi_list, roi_iou_list, roi_gt_list = [], [], []
        if fg_rois_per_this_lidar > 0:
            fg_rois_src = roi_boxes3d[fg_inds].copy()
            gt_of_fg_rois = gt_boxes3d[gt_assignment[fg_inds]]
            fg_rois, fg_iou3d = self.aug_roi_by_noise_batch(fg_rois_src, gt_of_fg_rois, aug_times=10)
            roi_list.append(fg_rois)
            roi_iou_list.append(fg_iou3d)
            roi_gt_list.append(gt_of_fg_rois)

        if bg_rois_per_this_lidar > 0:
            bg_rois_src = roi_boxes3d[bg_inds].copy()
            gt_of_bg_rois = gt_boxes3d[gt_assignment[bg_inds]]
            bg_rois, bg_iou3d = self.aug_roi_by_noise_batch(bg_rois_src, gt_of_bg_rois, aug_times=1)
            roi_list.append(bg_rois)
            roi_iou_list.append(bg_iou3d)
            roi_gt_list.append(gt_of_bg_rois)

        rois = np.concatenate(roi_list, axis=0)
        iou_of_rois = np.concatenate(roi_iou_list, axis=0)
        gt_of_rois = np.concatenate(roi_gt_list, axis=0)

        # collect extra features for point cloud pooling
        if cfg.RCNN.USE_INTENSITY:
            pts_extra_input_list = [rpn_intensity.reshape(-1, 1), seg_mask.reshape(-1, 1)]
        else:
            pts_extra_input_list = [seg_mask.reshape(-1, 1)]

        if cfg.RCNN.USE_DEPTH:
            pts_depth = (np.linalg.norm(rpn_xyz, ord=2, axis=1) / 70.0) - 0.5
            pts_extra_input_list.append(pts_depth.reshape(-1, 1))
        pts_extra_input = np.concatenate(pts_extra_input_list, axis=1)

        pts_input, pts_features, pts_empty_flag = roipool3d_utils.roipool3d_cpu(rois, rpn_xyz, rpn_features,
                                                                                pts_extra_input,
                                                                                cfg.RCNN.POOL_EXTRA_WIDTH,
                                                                                sampled_pt_num=cfg.RCNN.NUM_POINTS,
                                                                                canonical_transform=False)


        valid_mask = (pts_empty_flag == 0).astype(np.int32)

        # regression valid mask
        reg_valid_mask = (iou_of_rois > cfg.RCNN.REG_FG_THRESH).astype(np.int32) & valid_mask

        # classification label
        cls_label = (iou_of_rois > cfg.RCNN.CLS_FG_THRESH).astype(np.int32)
        invalid_mask = (iou_of_rois > cfg.RCNN.CLS_BG_THRESH) & (iou_of_rois < cfg.RCNN.CLS_FG_THRESH)
        cls_label[invalid_mask] = -1
        cls_label[valid_mask == 0] = -1

        # canonical transform and sampling
        pts_input_ct, gt_boxes3d_ct = self.canonical_transform_batch(pts_input, rois, gt_of_rois)

        sample_info = {'sample_id': sample_id,
                       'pts_input': pts_input_ct,
                       'pts_features': pts_features,
                       'cls_label': cls_label,
                       'reg_valid_mask': reg_valid_mask,
                       'gt_boxes3d_ct': gt_boxes3d_ct,
                       'roi_boxes3d': rois,
                       'roi_size': rois[:, 3:6],
                       'gt_boxes3d': gt_of_rois}

        return sample_info

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_lidar):
        if hard_bg_inds.size > 0 and easy_bg_inds.size > 0:
            hard_bg_rois_num = int(bg_rois_per_this_lidar * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_lidar - hard_bg_rois_num

            # sampling hard bg
            rand_num = np.floor(np.random.rand(hard_bg_rois_num) * hard_bg_inds.size).astype(np.int32)
            hard_bg_inds = hard_bg_inds[rand_num]
            # sampling easy bg
            rand_num = np.floor(np.random.rand(easy_bg_rois_num) * easy_bg_inds.size).astype(np.int32)
            easy_bg_inds = easy_bg_inds[rand_num]

            bg_inds = np.concatenate([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.size > 0 and easy_bg_inds.size == 0:
            hard_bg_rois_num = bg_rois_per_this_lidar
            # sampling hard bg
            rand_num = np.floor(np.random.rand(hard_bg_rois_num) * hard_bg_inds.size).astype(np.int32)
            bg_inds = hard_bg_inds[rand_num]
        elif hard_bg_inds.size == 0 and easy_bg_inds.size > 0:
            easy_bg_rois_num = bg_rois_per_this_lidar
            # sampling easy bg
            rand_num = np.floor(np.random.rand(easy_bg_rois_num) * easy_bg_inds.size).astype(np.int32)
            bg_inds = easy_bg_inds[rand_num]
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_batch(self, roi_boxes3d, gt_boxes3d, aug_times=10):
        """
        :param roi_boxes3d: (N, 7)
        :param gt_boxes3d: (N, 7)
        :return:
        """
        iou_of_rois = np.zeros(roi_boxes3d.shape[0], dtype=np.float32)
        for k in range(roi_boxes3d.__len__()):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]
            gt_box3d = gt_boxes3d[k]
            pos_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_box3d.reshape(1, 7))
            aug_box3d = roi_box3d
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                aug_corners = kitti_utils.boxes3d_to_corners3d(aug_box3d.reshape(1, 7))
                iou3d = kitti_utils.get_iou3d(aug_corners, gt_corners)
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d
            iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois

    def get_rcnn_sample_jit(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        sample_info = {'sample_id': sample_id,
                       'rpn_xyz': rpn_xyz,
                       'rpn_features': rpn_features,
                       'rpn_intensity': rpn_intensity,
                       'seg_mask': seg_mask,
                       'roi_boxes3d': roi_boxes3d,
                       'gt_boxes3d': gt_boxes3d,
                       'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

        return sample_info

    def collate_batch(self, batch):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict


if __name__ == '__main__':
    pass
