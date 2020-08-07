import numpy as np

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
        content = [line.strip().split(' ') for line in lines]
        annotations['name'] = np.array([x[0] for x in content])
        annotations['dimensions'] = np.array(
            [[float(info) for info in x[1:4]] for x in content]).reshape(
                -1, 3)
        annotations['location'] = np.array(
            [[float(info) for info in x[4:7]] for x in content]).reshape(-1, 3)
        annotations['rotation_y'] = np.array(
            [float(x[7]) for x in content]).reshape(-1)
    return annotations

if __name__ == "__main__":
    # example file
    # VEHICLE 1.7742 1.9809 4.5410 22.0288 15.6219 0.1392 3.1450 1.6872
    # name, h, w, l, x, y, z, ry, score
    # in LiDAR's frame
    detections_dir = "PointCNN/output/rpn/argo_config_sampling_trainfull/eval/epoch_no_number/sample/test_mode/detections/data"
    label_path = detections_dir + "/000000.txt"
    #print(get_label_anno(label_path)) 