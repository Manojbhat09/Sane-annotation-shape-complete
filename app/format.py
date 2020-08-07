import numpy as np

def sane_to_kitti(sane):
    # saves sane's json object to kitti format
    kitti = {}
    frame = sane["frame"]
    fname = frame["fname"]
    bounding_boxes = frame["bounding_boxes"]
    kitti["fname"] = frame
    for i in range(len(bounding_boxes)):
        kitti[i] = {}
        kitti[i]["id"] = bounding_boxes[i]["center"]["box_id"]
        kitti[i]["class"] = bounding_boxes[i]["center"]["object_id"]
        kitti[i]["center"] = {}
        kitti[i]["center"]['x'] = ((bounding_boxes[i]["center"][0]['z'] + 
                                  bounding_boxes[i]["center"][1]['z']) / 2)
        kitti[i]["center"]['y'] = ((bounding_boxes[i]["center"][0]['x'] + 
                                  bounding_boxes[i]["center"][1]['x']) / 2)

        kitti[i]["theta"] = bounding_boxes[i]["center"]["angle"]
        kitti[i]["l"] = bounding_boxes[i]["center"]["length"]
        kitti[i]["w"] = bounding_boxes[i]["center"]["width"]
        kitti[i]["h"] = bounding_boxes[i]["center"]["height"]
    return kitti

def kitti_to_sane(kitti):
    sane = {}
    return sane


