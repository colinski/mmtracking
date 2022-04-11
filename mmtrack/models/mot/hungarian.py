import numpy as np
import lap
import torch

#compute Hungarian algo
def match(dets, tracks, iou_thres=0.3):
    device = dets.device
    dets = dets.detach().cpu().numpy()
    tracks = tracks.detach().cpu().numpy()
    
    unmatched_dets = np.arange(len(dets)) #all idx are unmatched to start
    
    if len(dets) == 0 or len(tracks) == 0:
        matches = torch.empty(0, 2, device=device).long()
        unmatched_dets = torch.from_numpy(unmatched_dets).long()
        unmatched_dets = unmatched_dets.to(device=device)
        return matches, unmatched_dets
    
    #compute matching using IoU as metric
    iou_matrix = iou(dets, tracks) #D x T
    matches = linear_assignment(-iou_matrix)
    
    #remove matches that have low IoU
    valid = np.array([iou_matrix[d, t] >= iou_thres for d, t in matches])
    matches = matches[valid]
    
    #find all det idx not in a match
    unmatched_dets = np.setdiff1d(unmatched_dets, matches[:, 0])
    
    matches = torch.from_numpy(matches).long()
    matches = matches.to(device=device)
    unmatched_dets = torch.from_numpy(unmatched_dets).long()
    unmatched_dets = unmatched_dets.to(device=device)
    return matches, unmatched_dets

#compute linear assignment using lap library
def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    matches = np.array([[y[i], i] for i in x if i >= 0])
    return matches

#https://minibatchai.com/cv/detection/2021/07/18/VectorizingIOU.html
#bbox is ... x 4 
def area(bbox):
    x1 = bbox[..., 0]
    y1 = bbox[..., 1]
    x2 = bbox[..., 2]
    y2 = bbox[..., 3]
    return (x2 - x1) * (y2 - y1)

#bbox1 is N x 4, bbox2 is M x 4
#ouput is N x M
def iou(bbox1, bbox2):
    bbox1 = np.expand_dims(bbox1, 1)
    bbox2 = np.expand_dims(bbox2, 0)
    xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)
    wh = w * h
    iou = wh / (area(bbox1) + area(bbox2) - wh)
    return iou