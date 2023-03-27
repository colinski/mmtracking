import torch
from mmtrack.datasets import build_dataloader, build_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange
import numpy as np
import cv2
from tracker import TorchMultiObsKalmanFilter
from mmtrack.datasets.mocap.viz import init_fig, gen_rectange, gen_ellipse, rot2angle, points_in_rec
import json




img_norm_cfg = dict(mean=[0,0,0], std=[255,255,255], to_rgb=True)
img_pipeline = [
    dict(type='DecodeJPEG'),
    dict(type='LoadFromNumpyArray'),
    dict(type='Resize', img_scale=(270, 480), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]



pipelines = {
    'zed_camera_left': img_pipeline,
} 

valid_mods=['mocap', 'zed_camera_left']
valid_nodes=[1,2,3,4]

data_root = 'data/mmm/2022-09-01_1080p/trucks2_lightsT_obstaclesF/train'
trainset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_train/',
        hdf5_fnames=[
            f'{data_root}/mocap.hdf5',
            f'{data_root}/node_1/zed.hdf5',
            f'{data_root}/node_2/zed.hdf5',
            f'{data_root}/node_3/zed.hdf5',
            f'{data_root}/node_4/zed.hdf5',
        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        include_z=False, #(x,y) position only
    ),
    num_future_frames=0,
    num_past_frames=0, #sequence of length 5
    pipelines=pipelines,
)
trainset = build_dataset(trainset)

data_root = 'data/mmm/2022-09-01_1080p/trucks2_lightsT_obstaclesF/val'
valset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_val/',
        hdf5_fnames=[
            f'{data_root}/mocap.hdf5',
            f'{data_root}/node_1/zed.hdf5',
            f'{data_root}/node_2/zed.hdf5',
            f'{data_root}/node_3/zed.hdf5',
            f'{data_root}/node_4/zed.hdf5',
        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        include_z=False, #(x,y) position only
    ),
    num_future_frames=0,
    num_past_frames=0, #sequence of length 5
    pipelines=pipelines,
)
valset = build_dataset(valset)

data_root = 'data/mmm/2022-09-01_1080p/trucks2_lightsT_obstaclesF/test'
testset=dict(type='HDF5Dataset',
    cacher_cfg=dict(type='DataCacher',
        cache_dir= f'/dev/shm/cache_test/',
        hdf5_fnames=[
            f'{data_root}/mocap.hdf5',
            f'{data_root}/node_1/zed.hdf5',
            f'{data_root}/node_2/zed.hdf5',
            f'{data_root}/node_3/zed.hdf5',
            f'{data_root}/node_4/zed.hdf5',
        ],
        valid_mods=valid_mods,
        valid_nodes=valid_nodes,
        include_z=False, #(x,y) position only
    ),
    num_future_frames=0,
    num_past_frames=0, #sequence of length 5
    pipelines=pipelines,
)
testset = build_dataset(testset)


def collect_outputs(preds):
    for key, val in preds.items():
        num_frames = len(val) #should be same for all keys
    mean_seq, cov_seq = [], []
    for t in range(num_frames):
        means = [preds[key][t]['mean'].cpu() for key in preds.keys()]
        covs = [preds[key][t]['cov'].cpu() for key in preds.keys()]
        
        means = torch.stack(means) 
        covs = torch.stack(covs)
        nV, H, W, _ = means.shape
        means = means.reshape(nV*H*W, 2).t()
        covs = covs.reshape(nV*H*W, 2,2)
        mean_seq.append(means)
        cov_seq.append(covs)
    outputs = {'det_means': mean_seq, 'det_covs': cov_seq}
    return outputs

grid_x, grid_y = torch.meshgrid([torch.arange(0,700), torch.arange(0,500)])
grid = torch.stack([grid_x, grid_y], dim=-1).cuda().float()


def write_vid(frames, vid_name='pdf_node_1.mp4', size=100, key='log_probs', norm=True):
    vid = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (700,500))
    for lp in frames:
        lp = lp.numpy()
        lp = np.fliplr(lp)
        lp = lp.T
        # lp = cv2.resize(lp, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        if norm:
            lp = cv2.normalize(lp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lp = lp * 255
        lp = lp.astype(np.uint8)
        lp = cv2.applyColorMap(lp, cv2.COLORMAP_JET)
        #lp = cv2.copyMakeBorder(lp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
        #lp = cv2.resize(lp, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        vid.write(lp)
    vid.release()


def get_pdf_frames(preds, key='zed_camera_left_node_1', size=500):
    frames = []
    for vals in preds[key][0:size]:
        mean, cov, weights = vals['mean'].cuda(), vals['cov'].cuda(), vals['weights'].cuda()
        H, W, _ = mean.shape
        mean = mean.reshape(H*W,2)
        cov = cov.reshape(H*W,2,2)
        weights = weights.reshape(H*W)
        normal = torch.distributions.MultivariateNormal(mean, cov)
        mix = torch.distributions.Categorical(probs=weights)
        dist = torch.distributions.MixtureSameFamily(mix, normal)
        nll_vals = dist.log_prob(grid).cpu()
        nll_vals = nll_vals.exp()
        #nll_vals[nll_vals < -15] = -15
        frames.append(nll_vals)
    frames = torch.stack(frames)
    #frames = frames / frames.max()
    return frames

datasets = {'train': trainset, 'val': valset, 'test': testset}

expdir = 'logs/trucks1_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_28x20'
#'logs/trucks1_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_28x20', 
expdirs = [#'logs/trucks1_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_7x5', 
           #'logs/trucks1_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_1x1',
           'logs/trucks2_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_28x20', 
           'logs/trucks2_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_7x5', 
           'logs/trucks2_lightsT_obstaclesF_zed_nodes1234_yolo_tiny_1x1']

for expdir in expdirs:
    print(expdir)
    for ds in ['train', 'val', 'test']:
        print(ds)
        dataset = datasets[ds]
        preds = torch.load(f'{expdir}/{ds}/outputs.pt')
        #gt = dataset.collect_gt()
        #outputs = collect_outputs(preds)
        #res, outputs = dataset.track_eval(outputs, gt)
        dataset.write_video(None, logdir=f'{expdir}/{ds}/', video_length=500)

        # frames = get_pdf_frames(preds, 'zed_camera_left_node_1', size=500)
        # write_vid(frames, f'{expdir}/{ds}/pdf_node_1.mp4')

        # frames = get_pdf_frames(preds, 'zed_camera_left_node_2', size=500)
        # write_vid(frames, f'{expdir}/{ds}/pdf_node_2.mp4')

        # frames = get_pdf_frames(preds, 'zed_camera_left_node_3', size=500)
        # write_vid(frames, f'{expdir}/{ds}/pdf_node_3.mp4')

        # frames = get_pdf_frames(preds, 'zed_camera_left_node_4', size=500)
        # write_vid(frames, f'{expdir}/{ds}/pdf_node_4.mp4')
assert 1==2

frames = get_pdf_frames(preds, 'zed_camera_left_node_1', size=500)
write_vid(frames, f'{expdir}/train/pdf_node_1.mp4')

frames = get_pdf_frames(preds, 'zed_camera_left_node_2', size=500)
write_vid(frames, f'{expdir}/train/pdf_node_2.mp4')

frames = get_pdf_frames(preds, 'zed_camera_left_node_3', size=500)
write_vid(frames, f'{expdir}/train/pdf_node_3.mp4')

frames = get_pdf_frames(preds, 'zed_camera_left_node_4', size=500)
write_vid(frames, f'{expdir}/train/pdf_node_4.mp4')
import ipdb; ipdb.set_trace() # noqa

assert 1==2
import ipdb; ipdb.set_trace() # noqa 


# In[70]:


nll_vals.max()


# In[71]:


plt.imshow(nll_vals.t())


# plt.imshow(preds['zed_camera_left_node_1'][0]['weights'])

# In[ ]:


#val_outputs = collect_outputs(torch.load(f'{expdir}/val/outputs.pt'))
test_outputs = collect_outputs()
res, test_outputs = valset.track_eval(test_outputs, test_gt)


# In[ ]:


testset.write_video(test_outputs, logdir=f'{expdir}/test/', video_length=500)


# In[ ]:


test_outputs = collect_outputs(torch.load(f'{expdir}/test/outputs.pt'))



def collect_outputs(key='zed_camera_left_node_1', size=200):
    outputs = {'det_means': [], 'det_covs': [], 'det_weights': [], 'log_probs': []}
    for i in trange(size):
        weights = preds[key][i]['weights']
        means = preds[key][i]['mean'].reshape(x*y,2)
        covs = preds[key][i]['cov'].reshape(x*y,2,2)
        gt_pos = gt['all_gt_pos'][i]
        log_probs = []
        for j in range(len(means)):
            mu, S = means[j], covs[j]
            dist = torch.distributions.MultivariateNormal(mu, S)
            lp = dist.log_prob(gt_pos)
            lp[lp < -500] = -500
            log_probs.append(lp.mean())
        log_probs = torch.tensor(log_probs)
        log_probs = log_probs.reshape(x,y)
        outputs['log_probs'].append(log_probs)
        
        outputs['det_means'].append(means.t())
        outputs['det_covs'].append(covs)
        outputs['det_weights'].append(weights)
    return outputs


# In[ ]:


def write_vid(outputs, vid_name='pdf_node_1.mp4', size=200, key='log_probs', norm=True):
    vid = cv2.VideoWriter(f'{loc}/{vid_name}', cv2.VideoWriter_fourcc(*'mp4v'), 20, (700,500))
    for lp in tqdm(outputs[key][0:size]):
        lp = lp.numpy()
        lp = np.fliplr(lp)
        lp = lp.T
        lp = cv2.resize(lp, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        if norm:
            lp = cv2.normalize(lp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lp = lp * 255
        lp = lp.astype(np.uint8)
        lp = cv2.applyColorMap(lp, cv2.COLORMAP_JET)
        lp = cv2.copyMakeBorder(lp, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
        lp = cv2.resize(lp, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        vid.write(lp)
    vid.release()


# In[ ]:


def write_plot(outputs, vid_name='dists_node_1.mp4', size=200):
    #fig, axes = init_fig([('mocap','mocap')], num_rows=1)
    fig = plt.figure(figsize=(1*7, 1*5))
    key = ('mocap', 'mocap')
    axes = {}
    axes[key] = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fsize = (fig.get_figwidth()*50, fig.get_figheight()*50)
    fsize = tuple([int(s) for s in fsize])  
    vid = cv2.VideoWriter(f'{loc}/{vid_name}', cv2.VideoWriter_fourcc(*'mp4v'), 20, (700,500))
    
    for i in trange(size):
        axes[key].clear()
        axes[key].grid('on', linewidth=1)
        axes[key].set_xlim(0,700)
        axes[key].set_ylim(0,500)
        axes[key].set_aspect('equal')
        
        means = outputs['det_means'][i].t()
        covs = outputs['det_covs'][i]
        axes[key].scatter(means[:, 0], means[:, 1], color='black')
        for j in range(len(means)):
            #axes[key].scatter(means[j][0], means[j][1], color='black')
            ellipse = gen_ellipse(means[j], covs[j], edgecolor='black', fc='None', lw=1, linestyle='--')
            axes[key].add_patch(ellipse)


        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = cv2.resize(data, dsize=fsize)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = cv2.resize(data, dsize=(700,500), interpolation=cv2.INTER_LINEAR)
        vid.write(data) 
    vid.release()




outputs_node_1 = collect_outputs(key='zed_camera_left_node_1')
outputs_node_2 = collect_outputs(key='zed_camera_left_node_2')
outputs_node_3 = collect_outputs(key='zed_camera_left_node_3')
outputs_node_4 = collect_outputs(key='zed_camera_left_node_4')


# In[ ]:


write_vid(outputs_node_1, vid_name='pdf_node_1.mp4', key='log_probs')
write_vid(outputs_node_2, vid_name='pdf_node_2.mp4', key='log_probs')
write_vid(outputs_node_3, vid_name='pdf_node_3.mp4', key='log_probs')
write_vid(outputs_node_4, vid_name='pdf_node_4.mp4', key='log_probs')


# In[ ]:


write_vid(outputs_node_1, vid_name='weights_node_1.mp4', key='det_weights', norm=False)
write_vid(outputs_node_2, vid_name='weights_node_2.mp4', key='det_weights', norm=False)
write_vid(outputs_node_3, vid_name='weights_node_3.mp4', key='det_weights', norm=False)
write_vid(outputs_node_4, vid_name='weights_node_4.mp4', key='det_weights', norm=False)


# In[ ]:


write_plot(outputs_node_1, vid_name='dists_node_1.mp4')
write_plot(outputs_node_2, vid_name='dists_node_2.mp4')
write_plot(outputs_node_3, vid_name='dists_node_3.mp4')
write_plot(outputs_node_4, vid_name='dists_node_4.mp4')


# In[ ]:


# ffmpeg -i pdf_node_1.mp4 -i pdf_node_2.mp4 -i pdf_node_3.mp4 -i pdf_node_4.mp4 -filter_complex hstack=inputs=4 pdf.mp4
# ffmpeg -i weights_node_1.mp4 -i weights_node_2.mp4 -i weights_node_3.mp4 -i weights_node_4.mp4 -filter_complex hstack=inputs=4 weights.mp4
# ffmpeg -i dists_node_1.mp4 -i dists_node_2.mp4 -i dists_node_3.mp4 -i dists_node_4.mp4 -filter_complex hstack=inputs=4 dists.mp4
# ffmpeg -i pdf.mp4 -i weights.mp4 -i dists.mp4 -filter_complex vstack=inputs=3 stacked.mp4


# In[ ]:





# In[ ]:


lp.max()


# In[ ]:


np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)


# In[ ]:


key = 'zed_camera_left_node_1'
outputs = {'det_means': [], 'det_covs': [], 'det_weights': []}
for i in trange(len(dataset)):
    weights = preds[key][i]['weights']
    means = preds[key][i]['mean']
    covs = preds[key][i]['cov']
    idx = weights > 0.00
    
    print(means.shape)
    break
    mean = means[idx]
    cov = covs[idx]
    if len(mean) == 0:
        print('is zero')
    outputs['det_means'].append(mean.t())
    outputs['det_covs'].append(cov)
    outputs['det_weights'].append(weights.flatten())


# In[ ]:


all_weights = torch.stack(outputs['det_weights']).flatten()


# In[ ]:





# In[ ]:


plt.hist(all_weights, bins=100);
plt.xlabel('Mixture Weight in [0,1]')
plt.ylabel('Count')
plt.ylim(0,2000)
plt.xlim(0,1)


# In[ ]:


dataset.write_video(outputs=outputs, logdir='/home/csamplawski/logs/', video_length=500)


# In[ ]:


plt.savefig('test_fig.png')


# In[ ]:


weights.shape


# In[ ]:


plt.imshow(weights.t(), cmap='gray')


# In[ ]:





# In[ ]:


plt.figure(figsize=(28/4,20/4))

sns.heatmap(weights.t())


# In[ ]:


weights.max()


# ## outputs['det_means'][0].shape

# In[ ]:


idx = weights.flatten().argmax()


# In[ ]:


means.view(28*20, 2)[idx]


# In[ ]:


covs.view(28*20, 2,2)[idx]


# In[ ]:


covs.view(28*20, 2,2)[1][1]


# In[ ]:


weights[0][0]


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(weights.t())


# In[ ]:


plt.scatter(means[...,0], means[...,1])


# In[ ]:


data.keys()


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


data_loader = build_dataloader(
    dataset,
    samples_per_gpu=2, #batch_size per gpu
    workers_per_gpu=1, #workers per gu
    num_gpus=1,
    samples_per_epoch=None,
    dist=True,
    shuffle=True,
    seed=42,
    persistent_workers=False
)


# In[ ]:


data.keys()


# In[ ]:




