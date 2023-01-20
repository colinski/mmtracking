import torch
import torchvision
import h5py
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from tqdm import tqdm, trange
import cv2
import numpy as np


normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().cuda()

def forward(img):
    x = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.permute(2,0,1)
    x = normalize(x)
    x = x.unsqueeze(0)
    x = x.cuda()
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
    x = x.squeeze()
    x = x.permute(1,2,0)
    return x.cpu().numpy()

def process_file_zed(hdf5_fname, node_num, pivot='zed_camera_left'):
    base_path = '/'.join(hdf5_fname.split('/')[0:-1])
    out_file = h5py.File(base_path  + '/zed_r50v2.hdf5', 'w')
    f = h5py.File(hdf5_fname, 'r')
    timesteps = [k for k in f.keys()]
    for i in trange(len(timesteps)):
        ts = timesteps[i]
        subfile = f[ts][f'node_{node_num}']
        if pivot not in subfile.keys():
            continue

        code = subfile[pivot][:]
        img = cv2.imdecode(code, 1)
        feats = forward(img)
        g = out_file.create_group(ts)
        g = g.create_group(f'node_{node_num}')
        g.create_dataset('zed_camera_left_r50', data=feats, compression="gzip")
    f.close()
    out_file.close()

def process_file(hdf5_fname, node_num):
    base_path = '/'.join(hdf5_fname.split('/')[0:-1])
    out_file = h5py.File(base_path  + '/realsense_r50.hdf5', 'w')
    f = h5py.File(hdf5_fname, 'r')
    timesteps = [k for k in f.keys()]
    for i in trange(len(timesteps)):
        ts = timesteps[i]
        subfile = f[ts][f'node_{node_num}']
        if 'realsense_camera_img' not in subfile.keys():
            continue

        code = subfile['realsense_camera_img'][:]
        img = cv2.imdecode(code, 1)
        feats = forward(img)
        g = out_file.create_group(ts)
        g = g.create_group(f'node_{node_num}')
        g.create_dataset('realsense_camera_r50', data=feats, compression="gzip")
    f.close()
    out_file.close()

data_root = 'data/mmm/2022-09-01/trucks2_lightsT_obstaclesF/'
for node_num in [1,2,3,4]:
    for split in ['train', 'test']:
        print(node_num, split)
        hdf5_fname= f'{data_root}/{split}/node_{node_num}/realsense.hdf5'
        process_file(hdf5_fname, node_num)
