import torch
import numpy as np
import numba

@numba.jit(forceobj=True)
def post_processing(out, det_threshold = .2, nms_threshold = .5 , s= 14, b=2, classes_num=20):
    boxes = []
    conf = []
    classes_index = []
    print(out.shape)
    for i in range(s):
        for j in range(s):
            for k in range(b):
                boxes.append([(out[i][j][k*5+0]+j)/s,(out[i][j][k*5+1]+i)/s,out[i][j][k*5+2], out[i][j][k*5+3]])
                conf.append(out[i][j][k*5+4]*np.max(out[i][j][b*5+1:]))
                classes_index.append(np.argmax(out[i][j][b*5:]))

                
    boxes = np.array(boxes)

    box_xy = np.zeros_like(boxes)

    box_xy[...,:2] = boxes[...,:2] - 0.5 * boxes[...,2:]
    box_xy[...,2:] = boxes[...,:2] + 0.5 * boxes[...,2:]

    boxes = box_xy
    conf = np.array(conf)
    classes_index = np.array(classes_index)

    chosen = np.where(conf>det_threshold)

    boxes = boxes[chosen]
    classes_index = classes_index[chosen]
    conf = conf[chosen]
  
    if len(boxes) ==0:
        boxes = np.zeros((1,4))
        conf = np.zeros(1)
        classes_index = np.zeros(1)

    keep = nms(boxes,conf,nms_threshold)
    return boxes[keep],classes_index[keep],conf[keep]

  
@numba.jit(forceobj=True)
def nms(dets, scores, thresh=.5):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
