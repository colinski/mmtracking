import numpy as np
import torch
from .kalman_track import KalmanTrack
from .hungarian import match
import time

class SORT(object):
    def __init__(self, max_age=1, min_hits=3, iou_thres=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thres = iou_thres
        self.tracks = []
        self.frame_count = 0 
    
    def update(self, dets=torch.empty(0, 6)):
        self.frame_count += 1

        
        #get predictions from existing tracks
        for track in self.tracks:
            track.predict()
        
        #collect all the new bbox predictions
        preds = torch.zeros(0, 4)
        if len(self.tracks) > 0:
            preds = [track.state for track in self.tracks]
            preds = torch.stack(preds, dim=0)
        
        #Hungarian algo
        bbox = dets[:, 0:4]
        matches, unmatched_dets = match(bbox, preds, self.iou_thres)

        #update tracks with matched detections
        for d, t in matches:
            self.tracks[t].update(dets[d])

        #start new track for each unmatched detection
        for d in unmatched_dets:
            new_track = KalmanTrack(dets[d])
            self.tracks.append(new_track)
            
        #collect final outputs
        states, ids = [torch.empty(0,4).cuda()], []
        labels, scores = [], []
        for t, track in enumerate(self.tracks):
            onstreak = track.hit_streak >= self.min_hits
            warmingup = self.frame_count <= self.min_hits
            if track.wasupdated and (onstreak or warmingup):
                states.append(track.state.unsqueeze(0))
                ids.append(track.id)
                labels.append(track.label)
                scores.append(track.score)
                
        states = torch.cat(states, dim=0)
        # ids = torch.tensor(ids).reshape(-1, 1).cuda()
        # labels = torch.tensor(labels).reshape(-1, 1).cuda()
        ids = torch.tensor(ids).cuda()
        labels = torch.tensor(labels).cuda()
        scores = torch.tensor(scores).cuda()

        ret = (states, labels, ids, scores)
        #print(states.shape, ids.shape)
        #ret = torch.cat([states, ids], axis=-1)
        
        #remove tracks that have expired
        self.tracks = [track for track in self.tracks\
                       if track.time_since_update < self.max_age]
        return ret
