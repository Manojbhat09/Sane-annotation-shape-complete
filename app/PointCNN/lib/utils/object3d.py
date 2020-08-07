import numpy as np
from scipy.spatial.transform import Rotation

def cls_type_to_id(cls_type):
    type_to_id = {'VEHICLE': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        self.argo_to_kitti = np.array([[6.927964e-03, -9.999722e-01, -2.757829e-03],
                                       [-1.162982e-03, 2.749836e-03, -9.999955e-01],
                                       [9.999753e-01, 6.931141e-03, -1.143899e-03]])
        
        label = line
        self.src = line
        self.cls_type = label['label_class']
        self.cls_id = cls_type_to_id(self.cls_type)
        
        self.trucation = 0.0
        self.occlusion = 0.0  
        self.alpha = np.arctan2(label['center']['z'],label['center']['x'])
        
        self.h = float(label['height'])
        self.w = float(label['width'])
        self.l = float(label['length'])
        self.pos_argo = np.array([float(label['center']['x']), float(label['center']['y']), float(label['center']['z'])], dtype=np.float32)
        
        #KITTI Frame
        self.pos = np.dot(self.argo_to_kitti,self.pos_argo)
        w,x,y,z = label['rotation']['w'],label['rotation']['x'],label['rotation']['y'],label['rotation']['z']
        self.q = np.array([x, y, z, w])       
        self.rot_mat_argo = Rotation.from_quat(self.q).as_dcm()
        
        
        self.ry = -Rotation.from_quat(self.q).as_euler('xyz')[-1] + np.pi/2.
        self.score = -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        # Orginal : Assign level based on height of bounidng box in image, truncation, and occulusion value
        
        # Modified: Assign level based on distance from Origin of Lidar. Done
        distance = np.linalg.norm(self.pos)

        if distance <= 30.0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif distance > 30.0 and distance <= 60.0:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif distance > 60 :
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4
        
    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h/2., h/2., h/2., h/2., -h/2., -h/2., -h/2., -h/2.]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str

