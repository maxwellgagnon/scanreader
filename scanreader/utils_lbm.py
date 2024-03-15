from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.stats import gamma
import numpy as np
from tqdm import tqdm
from . import core
import itertools
import sys
import copy
sys.path.append('/mnt/lab/users/maxgagnon/src/mosaic-picasso')
import mosaic_picasso.mosaic as mp
import mosaic_picasso.utils as utils

''' 
For the initial/2023 Rockefellar LBM scans (And potentially future lbm scans) there are significant portions 
of the data was not stored/formatted in a way that scanreader, and some earlier tools, were initially designed 
to handle. Here are a few examples: 

- LBM scan depths are treated as channels by the scan collection software.
- LBM scan depths are not stored in sequential order.
- There is significant channel crosstalk/interference that must be corrected before processing

The functions in this file are intended to modify and add to the functionality of 
scanreader s.t. scanreader can handle light bead microscopy scans.

NOTE: There are other locations within scanreader that have been modified to handle lbm 
scans. I've tried to keep the bulk of them here, however

The goal of these additions to scanreader are to have the user (human, datajoint pipeline, etc...) interact with an LBM scan object like all other scans, and all of the 
lbm-unique formatting/differences are handled under the hood.
'''


def none_crosstalk(scan, correction_matrix, pre_images, post_images, mean_imgs, user_params=None):

    pre_images = mean_imgs.copy()
    post_images = mean_imgs.copy()
    correction_matrix = np.eye(scan.num_lbm_beads * scan.num_fields)

    return correction_matrix, pre_images, post_images


def mosaic_pairwise(scan, correction_matrix, pre_images, post_images, user_params=None, mean_imgs=None):
    
    # Default Params
    params = {  'bins': 256,
                'beta': 0,
                'gamma': 0.1,
                'cycles': 20,
                'nch': 2, 
                'threshold': 50}
    
    # Override default params with user params
    if user_params is not None:
        params.update(user_params)
    mosaic = mp.MosaicPicasso(**params)
    
    # Mosaic is calculate bead-wise, but information must be stored field-wise
    for bead_a_idx in range(scan.num_lbm_beads//2):
        bead_b_idx = bead_a_idx + scan.num_lbm_beads//2
        print(f"\nBeads: ({bead_a_idx}, {bead_b_idx})")
        
        # Extract all fields at a given bead, calculate crosstalk
        if mean_imgs is None:
            bead_a_fields_pre = np.mean(scan[:,bead_a_idx,:,:,:,:],axis=(-1))
            bead_b_fields_pre = np.mean(scan[:,bead_b_idx,:,:,:,:],axis=(-1))
        else:
            bead_a_fields_pre = mean_imgs[:,bead_a_idx,:,:]
            bead_b_fields_pre = mean_imgs[:,bead_b_idx,:,:]
            
        bead_ab_fields_pre = np.array(np.concatenate(np.stack((bead_a_fields_pre,bead_b_fields_pre),axis=-1),axis=0))
        bead_ab_fields_post,P = mosaic.mosaic(bead_ab_fields_pre)
            
        # Split array object into it's two beads
        bead_a_post, bead_b_post = np.split(bead_ab_fields_post, 2, axis=2)

        # Split each bead into it's <num_fields> fields
        bead_a_fields_post = np.split(bead_a_post, scan.num_fields, axis=0)
        bead_b_fields_post = np.split(bead_b_post, scan.num_fields, axis=0)
            
        # Fill correction_matrix, pre_images, and post_images
        for f in np.arange(scan.num_fields):
            lbm_field_a = bead_a_idx+(f*(scan.num_lbm_beads))
            lbm_field_b = bead_b_idx+(f*(scan.num_lbm_beads))
            
            correction_matrix[lbm_field_a, lbm_field_a] = P[0][0]
            correction_matrix[lbm_field_a, lbm_field_b] = P[1][0]
            correction_matrix[lbm_field_b, lbm_field_b] = P[1][1]
            correction_matrix[lbm_field_b, lbm_field_a] = P[0][1]
            
            pre_images[f, bead_a_idx,  :,:]  = bead_a_fields_pre[f].squeeze()
            pre_images[f, bead_b_idx,  :,:]  = bead_b_fields_pre[f].squeeze()
            post_images[f, bead_a_idx, :,:] = bead_a_fields_post[f].squeeze()
            post_images[f, bead_b_idx, :,:] = bead_b_fields_post[f].squeeze()
        
    return correction_matrix, pre_images, post_images


# Function that manages the specific crosstalk calculation methods
def calculate_crosstalk(scan, method, mean_imgs=None):
    
    # Universal pre-allocations
    correction_matrix = np.full((scan.num_lbm_beads * scan.num_fields, scan.num_lbm_beads * scan.num_fields), np.nan)
    
    # Pre-allocate pre_image and post_image structures. Must be able to handle fields of varying widths and heights.
    pre_images = []
    for field_idx in range(scan.num_fields):
        height = scan.field_heights[field_idx]
        width  = scan.field_widths[field_idx]
        field_images = np.empty((scan.num_lbm_beads, height, width))
        pre_images.append(field_images)
    pre_images = np.array(pre_images, dtype=object)
    post_images = copy.deepcopy(pre_images)
          
          
    if method == 'None' or method == 'none':
        analysis_method = none_crosstalk
        method_params = None
        
    elif method == 'mosaic_pairwise_default':
        analysis_method = mosaic_pairwise
        method_params = None
        
    else:
        analysis_method = None
        method_params = None
    
    
    
    if analysis_method:
        return analysis_method(scan, correction_matrix, pre_images, post_images, method_params, mean_imgs)
    else:
        raise ValueError(f"Method not found: {method}")