import torch
import numpy as np

def gen_heatmap(filtered_dets, gt, grid_width=700, grid_height=500, bin_size=100):
    num_bins_x = int(grid_width / bin_size)
    num_bins_y = int(grid_height / bin_size)
    cov_grid = [[[] for j in range(num_bins_y)] for i in range(num_bins_x)]
    
    for i in range(len(filtered_dets)):
        dets = filtered_dets[i]
        num_dets = len(dets['mean'])
        for gt_pos in gt['all_gt_pos'][i]:
            bin_col = int(gt_pos[0] / bin_size)
            bin_row = int(gt_pos[1] / bin_size)
            cov_grid[bin_col][bin_row].extend([S for S in dets['cov']])

    heatmap = torch.zeros(num_bins_x, num_bins_y, 2, 2)
    for i in range(num_bins_x):
        for j in range(num_bins_y):
            grid_vals = cov_grid[i][j]
            if len(grid_vals) == 0:
                cov = torch.ones(2,2) * np.nan
            else:
                grid_vals = torch.stack(grid_vals)
                cov = grid_vals.mean(dim=0)
            heatmap[i,j] = cov
    return heatmap
