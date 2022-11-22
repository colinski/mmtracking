import numpy as np
import torch
import pyro
from pyro.contrib.tracking.dynamic_models import NcpContinuous, NcvContinuous
from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.measurements import PositionMeasurement

#from filterpy.kalman import KalmanFilter

#xyar = x,y position of center, area, aspect ratio
#xyxy = default format for mmdet and cv2 rectange
def xyxy_to_xyar(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2
    y = bbox[1] + h / 2
    a = w * h
    r = w / h
    bbox = [v.view(1) for v in [x,y,a,r]]
    bbox = torch.cat(bbox)
    return bbox

def xyar_to_xyxy(bbox):
    w = torch.sqrt(bbox[2] * bbox[3])
    h = bbox[2] / w
    x1 = bbox[0] - w / 2
    y1 = bbox[1] - h / 2
    x2 = bbox[0] + w / 2
    y2 = bbox[1] + h / 2
    bbox = [v.view(1) for v in [x1,y1,x2,y2]]
    bbox = torch.cat(bbox)
    return bbox

class KalmanTrack(torch.nn.Module):
    count = 0 #global count across tracks for id
    def __init__(self, det):
        super().__init__()
        self.time_since_update = 0
        self.id = KalmanTrack.count
        KalmanTrack.count += 1
        self.hit_streak = 0
        self.age = 0
        
        #bbox is 1 x 6 
        bbox = det[0:4]
        self.score = det[4].cpu()
        self.label = det[5].cpu()
        
        mean = xyxy_to_xyar(bbox).cpu()
        cov = torch.eye(4)
        #cov[2:, 2:] *= 10
        self.dymodel = NcpContinuous(4, 2.0)
        self.kf = EKFState(self.dymodel, mean, cov, time=0)
    
    @property
    def wasupdated(self):
        return self.time_since_update < 1
    
    @property
    def state(self):
        mean = self.kf.mean.cuda()
        state = xyar_to_xyxy(mean)
        return state

    def update(self, det):
        bbox = det[0:4]
        self.time_since_update = 0
        self.hit_streak += 1
        bbox = xyxy_to_xyar(bbox).cpu()
        cov = torch.eye(4) * 0.01
        cov = cov
        m = PositionMeasurement(bbox, cov, time=self.kf.time)
        self.kf, _ = self.kf.update(m)

    def predict(self):        
        self.age += 1
        self.kf = self.kf.predict(dt=self.age)        
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

class MocapTrack(torch.nn.Module):
    count = 0 #global count across tracks for id
    def __init__(self, mean, cov=None, sa=1e-2):
        super().__init__()
        self.sa = sa
        self.time_since_update = 0
        self.id = MocapTrack.count
        MocapTrack.count += 1
        self.hit_streak = 0
        self.age = 0
        
        self.state_size = len(mean)
        if cov is None:
            cov = torch.ones(self.state_size) * 0.01
        # self.dymodel = self.dymodel.to(mean.device)
        
        self.dymodel = NcvContinuous(self.state_size*2, self.sa)

        mean = torch.cat([mean, torch.zeros_like(mean) + 0.0])

        if len(cov.shape) == 2: #full matrix
            S = cov.new_zeros(self.state_size*2, self.state_size*2)
            S[0:self.state_size,0:self.state_size] = cov
            self.kf = EKFState(self.dymodel, mean.cpu(), S.cpu(), time=0)
        else:
            cov = torch.cat([cov, torch.zeros_like(cov) + 1.0])
            self.kf = EKFState(self.dymodel, mean.cpu(), torch.diag(cov).cpu(), time=0)


        # self.dymodel = NcpContinuous(self.state_size, 1)
        #self.kf = EKFState(self.dymodel, mean.cpu().unsqueeze(0), cov.cpu().unsqueeze(0), time=0)
        #self.kf = EKFState(self.dymodel, mean.cpu(), torch.diag(cov).cpu(), time=0)
        # self.kf = EKFState(self.dymodel, mean.cpu(), torch.diag(cov).cpu(), time=0)
    
    @property
    def wasupdated(self):
        return self.time_since_update < 1
    
    # @property
    # def state(self):
        # mean = self.kf.mean.cuda()
        # state = mean
        # return state

    @property
    def mean(self):
        return self.kf.mean[..., 0:self.state_size]

    @property
    def cov(self):
        return self.kf.cov[..., 0:self.state_size, 0:self.state_size]

    def update(self, mean, cov=None):
        # bbox = det[0:4]
        self.time_since_update = 0
        self.hit_streak += 1
        # bbox = xyxy_to_xyar(bbox).cpu()
        # cov = torch.eye(4) * 0.01
        # cov = cov
        if cov is None:
            cov = torch.ones(self.state_size) * 0.01
        #m = PositionMeasurement(mean.cpu(), torch.diag(cov).cpu(), time=self.kf.time)
        m = PositionMeasurement(mean.cpu(), cov.cpu(), time=self.kf.time)
        # inno = self.kf.innovation(m)
        self.kf, _ = self.kf.update(m)

    def predict(self):        
        self.age += 1
        self.kf = self.kf.predict(dt=1)
        # self.kf = self.kf.predict(dt=self.age)
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
