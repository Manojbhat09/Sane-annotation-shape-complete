import _init_path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lib.net.point_rcnn import PointRCNN
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.utils.bbox_transform import decode_bbox_target

from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import argparse
import lib.utils.kitti_utils as kitti_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
from datetime import datetime
import logging
import re
import glob
import time
from tensorboardX import SummaryWriter
import tqdm
from pyntcloud import PyntCloud

argo_to_kitti = np.array([[0, -1, 0],[0, 0, -1],[1, 0, 0]])
kitti_to_argo = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])

np.random.seed(1024)  # set the same seed

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--cfg_file', type=str, default='cfgs/default.yml', help='specify the config for evaluation')
parser.add_argument("--eval_mode", type=str, default='rpn', required=True, help="specify the evaluation mode")

parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify the checkpoint of rpn if trained separated")
parser.add_argument("--rcnn_ckpt", type=str, default=None, help="specify the checkpoint of rcnn if trained separated")
parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")

parser.add_argument('--batch_size', type=int, default=1, help='batch size for evaluation')
parser.add_argument('--save_result', action='store_true', default=True, help='save evaluation results to files')
parser.add_argument('--single_file', type=str, default = None, help='evaluate a specified file')



args = parser.parse_args()

# Remove
def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def save_kitti_format(sample_id, bbox3d, kitti_output_dir, scores, lidar_name_table):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    if(args.single_file):
        kitti_output_file = os.path.join(kitti_output_dir, lidar_name_table + '.txt')
    else:
        kitti_output_file = os.path.join(kitti_output_dir, lidar_name_table['%06d'%sample_id] + '.txt')
    with open(kitti_output_file, 'w') as f: 
        for k in range(bbox3d.shape[0]):
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            #beta = np.arctan2(z, x)
            #alpha = -np.sign(beta) * np.pi / 2 + beta + ry
            print('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k,2], -bbox3d[k,0], -bbox3d[k,1],
                   (np.pi/2. - bbox3d[k, 6]), scores[k]), file=f)





def eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(666)

    # Loads the mean size of the CLASS from CFG YAML file
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    # Assign the MODE as TEST unless EVAL specified
    mode = 'TEST' if args.test else 'EVAL'

    # Make output directory result_dir/final_result/data
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    # Save data if args.save_result is True or not(default now True)
    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
        os.makedirs(rpn_output_dir, exist_ok=True)
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    dataset = dataloader.dataset
    lidar_name_table = dataset.lidar_name_table
    
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0
    # Iterate through data in dataloader
    for data in dataloader:
        cnt += 1
        sample_id, pts_rect, pts_features, pts_input, rem_pts = data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input'], data['rem_pts']
        batch_size = len(sample_id)
        inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        input_data = {'pts_input': inputs}

        # model inference
        ret_dict = model(input_data)

        roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
        roi_boxes3d = ret_dict['rois']  # (B, M, 7)
        seg_result = ret_dict['seg_result'].long()  # (B, N)

        rcnn_cls = ret_dict['rcnn_cls'].view(batch_size, -1, ret_dict['rcnn_cls'].shape[1])
        rcnn_reg = ret_dict['rcnn_reg'].view(batch_size, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

        # bounding box regression
        anchor_size = MEAN_SIZE
        if cfg.RCNN.SIZE_RES_ON_ROI:
            assert False

        pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                          anchor_size=anchor_size,
                                          loc_scope=cfg.RCNN.LOC_SCOPE,
                                          loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                          num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                          get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                          loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                          get_ry_fine=True).view(batch_size, -1, 7)

        # scoring
        if rcnn_cls.shape[2] == 1:
            raw_scores = rcnn_cls  # (B, M, 1)
            norm_scores = torch.sigmoid(raw_scores)
            pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
        else:
            pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
            cls_norm_scores = F.softmax(rcnn_cls, dim=1)
            raw_scores = rcnn_cls[:, pred_classes]
            norm_scores = cls_norm_scores[:, pred_classes]

        
        if args.save_result:
            # save roi and refine results
            roi_boxes3d_np = roi_boxes3d.cpu().numpy()
            pred_boxes3d_np = pred_boxes3d.cpu().numpy()
            roi_scores_raw_np = roi_scores_raw.cpu().numpy()
            raw_scores_np = raw_scores.cpu().numpy()

            rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
            rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
            rpn_xyz_np = np.concatenate([rpn_xyz_np[0][:,2].reshape(-1,1),-rpn_xyz_np[0][:,0].reshape(-1,1),-rpn_xyz_np[0][:,1].reshape(-1,1)], axis = 1).reshape(1,-1,3)
            seg_result_np = seg_result.cpu().numpy()


            #print(rem_pts.shape)
            rem_pts = np.concatenate([rem_pts[0,:,2].reshape(-1,1),-rem_pts[0,:,0].reshape(-1,1), -rem_pts[0,:,1].reshape(-1,1)], axis = 1)
            
            rest_lidar_pts = np.hstack((rem_pts, np.zeros(rem_pts.shape[0]).reshape(-1,1), np.zeros(rem_pts.shape[0]).reshape(-1,1))).reshape(1,-1,5)

            #print(rest_lidar_pts.shape)
            output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(batch_size, -1, 1),
                                          seg_result_np.reshape(batch_size, -1, 1)), axis=2)
            
            output_data = np.hstack((rest_lidar_pts,output_data))
            #print(output_data.shape)
            

            for k in range(batch_size):
                cur_sample_id = sample_id[k]
                output_file = os.path.join(rpn_output_dir, lidar_name_table['%06d'%cur_sample_id] + '.npy')
                np.save(output_file, output_data.astype(np.float32))

        # scores thresh
        inds = norm_scores > cfg.RCNN.SCORE_THRESH

        for k in range(batch_size):
            cur_inds = inds[k].view(-1)
            if cur_inds.sum() == 0:
                continue

            pred_boxes3d_selected = pred_boxes3d[k, cur_inds]
            raw_scores_selected = raw_scores[k, cur_inds]
            norm_scores_selected = norm_scores[k, cur_inds]

            # NMS thresh
            # rotated nms
            boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
            keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
            pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
            scores_selected = raw_scores_selected[keep_idx]
            pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()

            cur_sample_id = sample_id[k]
            final_total += pred_boxes3d_selected.shape[0]
            save_kitti_format(cur_sample_id, pred_boxes3d_selected, final_output_dir, scores_selected,lidar_name_table)

    ret_dict = {}
    avg_det_num = (final_total / max(len(dataset), 1.0))
    logger.info('final average detections: %.3f' % avg_det_num)
    
    return ret_dict


def eval_one_epoch_joint_single_file(model, input_list, file_path, result_dir,logger):
    np.random.seed(666)

    # Loads the mean size of the CLASS from CFG YAML file
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    # Assign the MODE as TEST unless EVAL specified
    mode = 'TEST'

    filename = file_path.split('/')[-1].split('.')[0]

    # Make output directory result_dir/final_result/data
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)

    # Save data if args.save_result is True or not(default now True)
    if args.save_result:
        roi_output_dir = os.path.join(result_dir, 'roi_result', 'data')
        refine_output_dir = os.path.join(result_dir, 'refine_result', 'data')
        rpn_output_dir = os.path.join(result_dir, 'rpn_result', 'data')
        os.makedirs(rpn_output_dir, exist_ok=True)
        os.makedirs(roi_output_dir, exist_ok=True)
        os.makedirs(refine_output_dir, exist_ok=True)

    #logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch_id)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    #input_data = input_data.cuda() 
    
    cnt = final_total = total_cls_acc = total_cls_acc_refined = total_rpn_iou = 0
    # Iterate through data in dataloader
    #cnt += 1
    #sample_id, pts_rect, pts_features, pts_input = data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
    #batch_size = len(sample_id)
    #inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()

    pts_lidar = input_list[0]
    rem_index = input_list[2]
    input_data = {'pts_input': torch.from_numpy(pts_lidar[input_list[1], :]).view(1,-1,3).float().cuda()}

    # model inference
    ret_dict = model(input_data)

    roi_scores_raw = ret_dict['roi_scores_raw']  # (B, M)
    roi_boxes3d = ret_dict['rois']  # (B, M, 7)
    seg_result = ret_dict['seg_result'].long()  # (B, N)

    rcnn_cls = ret_dict['rcnn_cls'].view(1, -1, ret_dict['rcnn_cls'].shape[1])
    rcnn_reg = ret_dict['rcnn_reg'].view(1, -1, ret_dict['rcnn_reg'].shape[1])  # (B, M, C)

    # bounding box regression
    anchor_size = MEAN_SIZE
    if cfg.RCNN.SIZE_RES_ON_ROI:
        assert False

    pred_boxes3d = decode_bbox_target(roi_boxes3d.view(-1, 7), rcnn_reg.view(-1, rcnn_reg.shape[-1]),
                                      anchor_size=anchor_size,
                                      loc_scope=cfg.RCNN.LOC_SCOPE,
                                      loc_bin_size=cfg.RCNN.LOC_BIN_SIZE,
                                      num_head_bin=cfg.RCNN.NUM_HEAD_BIN,
                                      get_xz_fine=True, get_y_by_bin=cfg.RCNN.LOC_Y_BY_BIN,
                                      loc_y_scope=cfg.RCNN.LOC_Y_SCOPE, loc_y_bin_size=cfg.RCNN.LOC_Y_BIN_SIZE,
                                      get_ry_fine=True).view(1, -1, 7)

    # scoring
    if rcnn_cls.shape[2] == 1:
        raw_scores = rcnn_cls  # (B, M, 1)
        norm_scores = torch.sigmoid(raw_scores)
        pred_classes = (norm_scores > cfg.RCNN.SCORE_THRESH).long()
    else:
        pred_classes = torch.argmax(rcnn_cls, dim=1).view(-1)
        cls_norm_scores = F.softmax(rcnn_cls, dim=1)
        raw_scores = rcnn_cls[:, pred_classes]
        norm_scores = cls_norm_scores[:, pred_classes]

    
    if args.save_result:
        # save roi and refine results
        roi_boxes3d_np = roi_boxes3d.cpu().numpy()
        pred_boxes3d_np = pred_boxes3d.cpu().numpy()
        roi_scores_raw_np = roi_scores_raw.cpu().numpy()
        raw_scores_np = raw_scores.cpu().numpy()

        rpn_cls_np = ret_dict['rpn_cls'].cpu().numpy()
        rpn_xyz_np = ret_dict['backbone_xyz'].cpu().numpy()
        rpn_xyz_np = np.concatenate([rpn_xyz_np[0][:,2].reshape(-1,1),-rpn_xyz_np[0][:,0].reshape(-1,1),-rpn_xyz_np[0][:,1].reshape(-1,1)], axis = 1).reshape(1,-1,3)
        seg_result_np = seg_result.cpu().numpy()

        rest_lidar_pts = np.hstack((pts_lidar[rem_index,:], np.zeros(len(rem_index)).reshape(-1,1), np.zeros(len(rem_index)).reshape(-1,1))).reshape(1,-1,5)

        output_data = np.concatenate((rpn_xyz_np, rpn_cls_np.reshape(1, -1, 1),
                                      seg_result_np.reshape(1, -1, 1)), axis=2)
        output_data = np.hstack((rest_lidar_pts,output_data))
        
        # Change this
        cur_sample_id = 0
        output_file = os.path.join(rpn_output_dir, filename + '.npy')
        np.save(output_file, output_data.astype(np.float32))

    # scores thresh
    inds = norm_scores > cfg.RCNN.SCORE_THRESH

    cur_inds = inds[0].view(-1)
    pred_boxes3d_selected = pred_boxes3d[0, cur_inds]
    raw_scores_selected = raw_scores[0, cur_inds]
    norm_scores_selected = norm_scores[0, cur_inds]

    # NMS thresh
    # rotated nms
    boxes_bev_selected = kitti_utils.boxes3d_to_bev_torch(pred_boxes3d_selected)
    keep_idx = iou3d_utils.nms_gpu(boxes_bev_selected, raw_scores_selected, cfg.RCNN.NMS_THRESH).view(-1)
    pred_boxes3d_selected = pred_boxes3d_selected[keep_idx]
    scores_selected = raw_scores_selected[keep_idx]
    pred_boxes3d_selected, scores_selected = pred_boxes3d_selected.cpu().numpy(), scores_selected.cpu().numpy()

    cur_sample_id = 0
    final_total += pred_boxes3d_selected.shape[0]
    save_kitti_format(cur_sample_id, pred_boxes3d_selected, final_output_dir, scores_selected,filename)

    ret_dict = {}
    logger.info('final average detections: %.3f' % final_total)
    
    return ret_dict

def get_lidar(lidar_file):
        
        assert os.path.exists(lidar_file)
        ground_removal = False
        
        data = PyntCloud.from_file(lidar_file)
        x = np.array(data.points.x)[:, np.newaxis]
        y = np.array(data.points.y)[:, np.newaxis]
        z = np.array(data.points.z)[:, np.newaxis]
        pts_lidar = np.concatenate([x,y,z], axis = 1)
        
        '''
        if ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        '''
        pts_lidar_orig = np.dot(argo_to_kitti,pts_lidar.T).T
        total_index = pts_lidar_orig.shape[0]

        valid_mask = get_valid_flag(pts_lidar_orig)
        pts_lidar = pts_lidar_orig[valid_mask]

        npoints = 65536
        if npoints < len(pts_lidar):
            #Selecting the depth column
            pts_depth = pts_lidar[:, 2]
            # Creating a Mask for points within a radius of 60.0 
            pts_near_flag = np.abs(pts_depth) < 60.0
            # Creating the complimentary mask for far points
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            # Creating index for near points
            near_idxs = np.where(pts_near_flag == 1)[0]
            # randomly select points from near points indexes, total upto (max points- far points)
            # near_points + far_points --> total points
            near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=True)
            # concatenate the randomly chosen near points indexes with far points indexes
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)
            
            # max_points > total_points 
        else:
            # Case : self.npoints(max_points) == len(pts_rect) (total points)
            choice = np.arange(0, len(pts_lidar), dtype=np.int32)
            if npoints > len(pts_lidar):
                extra_choice = np.random.choice(choice, npoints - len(pts_lidar), replace=True)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        rem_index = np.setdiff1d(np.arange(total_index),choice)
        #pts_lidar = torch.from_numpy(pts_lidar[choice, :]).view(1,-1,3).float().cuda()
        
        return [pts_lidar_orig, choice, rem_index]

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

def eval_one_epoch(model, dataloader, epoch_id, result_dir, logger):
    # Returns a dictionary -->
    # Checks if RPN and RCNN is enabled or not
    if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir, logger)
    elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_rcnn(model, dataloader, epoch_id, result_dir, logger)
    elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
        ret_dict = eval_one_epoch_joint(model, dataloader, epoch_id, result_dir, logger)
    else:
        raise NotImplementedError
    return ret_dict


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model_state = checkpoint['model_state']

        update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


def load_ckpt_based_on_args(model, logger):
    """
    Input: model and logger instance
    Output: None
    Task: Loads ckpt based on the args --rpn_ckpt and  --rcnn_ckpt
    """
    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)

    if cfg.RCNN.ENABLED and args.rcnn_ckpt is not None:
        load_part_ckpt(model, filename=args.rcnn_ckpt, logger=logger, total_keys=total_keys)


def eval_single_ckpt(root_result_dir):
    root_result_dir = os.path.join(root_result_dir, 'eval')
    
    # set epoch_id and output dir
    epoch_id = 'no_number'
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, cfg.TEST.SPLIT)
    
    # Checks if TEST mode in on or not
    if args.test:
        root_result_dir = os.path.join(root_result_dir, 'test_mode')

    # Create root_result_dir if it doesn't exists
    os.makedirs(root_result_dir, exist_ok=True)

    # Log File initialize
    log_file = os.path.join(root_result_dir, 'log_eval_one.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    save_config_to_file(cfg, logger=logger)

    # Check if its a Single file or not
    model = PointRCNN(num_classes=2, use_xyz=True, mode='TEST')
    model.cuda()
    # load checkpoint
    load_ckpt_based_on_args(model, logger)

    if(args.single_file != None):
        eval_one_epoch_joint_single_file(model,get_lidar(args.single_file),args.single_file,root_result_dir,logger)

    else:
        # create dataloader & network
        test_loader = create_dataloader(logger)
        # start evaluation
        eval_one_epoch(model, test_loader, epoch_id, root_result_dir, logger)


def create_dataloader(logger):
    mode = 'TEST' if args.test else 'EVAL'
    #DATA_PATH = os.path.join('..', 'data')
    # calls from app/frame_handler
    DATA_PATH = '../../test_dataset/0_drive_0064_sync'
    
    # create dataloader
    test_set = KittiRCNNDataset(root_dir=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TEST.SPLIT, mode=mode,
                                random_select=True,
                                rcnn_eval_roi_dir=None,
                                rcnn_eval_feature_dir=None,
                                classes=cfg.CLASSES,
                                logger=logger)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                             num_workers=8, collate_fn=test_set.collate_batch)

    return test_loader


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

if __name__ == "__main__":
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    cfg.RCNN.ENABLED = True
    cfg.RPN.ENABLED = cfg.RPN.FIXED = True
    root_result_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG)
    ckpt_dir = os.path.join('../', 'output', 'rcnn', cfg.TAG, 'ckpt')
    os.makedirs(root_result_dir, exist_ok=True)

    # Check with Args --> eval_all --> 'whether to evaluate all checkpoints'
    # root_result_dir --> directory where results will be stored --> ../output/rpn/cfg.TAG/
    with torch.no_grad():
        eval_single_ckpt(root_result_dir)
