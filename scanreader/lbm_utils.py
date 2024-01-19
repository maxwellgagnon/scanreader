from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.stats import gamma
import numpy as np
from tqdm import tqdm
from . import core

import sys
sys.path.append('/mnt/lab/users/maxgagnon/src/mosaic-picasso')
import mosaic_picasso.mosaic as mp

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
channel_order = [0,*range(4,9),
                 1,*range(9,17),
                 2,*range(17,22),
                 3,*range(22,30)]

def sum_log_lik_one_line(m, x, y, b = 0, sigma_0 = 10,  c = 1e-10, m_penalty=0):
    mu = m * x + b
    lik_line = gaussian(y, mu, sigma_0)
    lik = lik_line
    
    log_lik = np.log(lik + c - m * m_penalty).sum()
    return -log_lik

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2*np.pi))

# # Specific Methods for calculating crosstalk
# def s2plbm_crosstalk(scan_filename, depths_considered, users_params=None):
#     ''' Crosstalk calculateion method implemented in s2plbm '''

#     print('Reading header...')
#     scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
#     scan = sr.core.read_scan(scan_filename, check_lbm=True) 
    
#     params = {
#             'force_positive': True,
#             'fit_above_percentile': 99.5,
#             'n_proc': 1,
#             'estimate_gamma': True,
#             'peak_width': 1,
#             'sigma': 0.01,
#             'm_penalty': 0}
    
#     # override default params with user params
#     if users_params is not None:
#         params.update(users_params)
        
#     # Confirm scan is lbm
#     if not scan.is_lbm():
#         return []
#     depth_amt = scan.lbm_depth_amt()
    
#     field_sizes = np.asarray([field.shape for field in scan])
    
#     assert len(np.unique(field_sizes[:,1])) == 1, "field stiching not possible"
#     shape_of_stiched_fields = (np.sum(np.unique(field_sizes[:,0])), field_sizes[0,1])
#     all_stiched_fields = np.empty((depth_amt, *shape_of_stiched_fields))

#     for depth_idx in tqdm(range(depth_amt)):
#         fields_in_a_depth = np.arange(depth_idx, depth_amt*field_amt, depth_amt)
#         all_stiched_fields[depth_idx] = np.vstack([np.mean(scan[int(field)], axis=-1) for field in fields_in_a_depth])
        
#     if params['force_positive']:
#         all_stiched_fields = all_stiched_fields - all_stiched_fields.min(axis=(1,2),keepdims=True)

#     assert all_stiched_fields.shape[0] == 30

#     m_opts = [] 
#     m_firsts = []
#     all_liks = []
#     m_opt_liks = []
#     m_first_liks = []
#     ms = np.linspace(0,1,101)

#     for idx, i in enumerate(range(depth_amt//2)):
#         X = all_stiched_fields[i].flatten()
#         Y = all_stiched_fields[i+15].flatten()
#         idxs = X > np.percentile(X, params['fit_above_percentile'])

#         if params['n_proc'] == 1:
#             liks = np.array([sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0 = params['sigma'], m_penalty=params['m_penalty']) for m in ms])
#         else:
#             p = Pool(params['n_proc'])
#             liks = p.starmap(sum_log_lik_one_line,[(m, X[idxs], Y[idxs],0, params['sigma'],1e-10,params['m_penalty']) for m in ms])
#             liks = np.array(liks)

#         m_opt = ms[np.argmin(liks)]
#         pks = find_peaks(-liks, width=params['peak_width'])[0]
#         m_first = ms[pks[0]]

#         m_opts.append(m_opt)
#         m_firsts.append(m_first)
#         all_liks.append(liks)
#         m_opt_liks.append(liks.min())
#         m_first_liks.append(liks[pks[0]])

#     m_opts = np.array(m_opts)
#     m_firsts = np.array(m_firsts)

#     best_ms = m_opts[m_opts==m_firsts]
#     best_m = best_ms.mean()

#     if method_params['estimate_gamma']:
#         gx = gamma.fit(m_opts)
#         x = np.linspace(0,1,1001)
#         gs = gamma.pdf(x, *gx)
#         best_m = x[np.argmax(gs)]
        
#     return m_opts, best_m
    
# def mosaic_crosstalk(scan_filename, depths_considered, users_params=None):
#     ''' Crosstalk calculateion method implemented in mosaic-picasso '''
#     print('Reading header...')
#     scan = core.read_scan(scan_filename, check_lbm=True) 
    
#     params = {
#             'bins': 256,
#             'beta': 0,
#             'gamma': 0.1,
#             'cycles': 20,
#             'nch': 2,
#             'threshold': 50}
    
#     # override default params with user params
#     if users_params is not None:
#         params.update(users_params)
    
def none_crosstalk(scan_filename, depths_considered, users_params=None):
    return np.ones((len(depths_considered), len(depths_considered)))

# Function that manages the specific crosstalk calculation methods
def calculate_crosstalk(scan_filename, method, depths_considered, save_results_path=None, verbose=False):
    
    if method == 'mosaic_default':
        analysis_method = mosaic_crosstalk
        method_params = {'bins': 256,
                        'beta': 0,
                        'gamma': 0.1,
                        'cycles': 20,
                        'nch': len(depths_considered),
                        'threshold': 50}
        
    if method == 's2plbm':
        analysis_method = s2plbm_crosstalk
        method_params = None
        
    elif method == 'None':
        analysis_method = none_crosstalk
        method_params = None
    
    else:
        analysis_method = None
        method_params = None
    
    if analysis_method:
        return analysis_method(scan_filename, depths_considered, method_params)
    else:
        raise ValueError(f"Method {method} not found.")

'''
    if method == 's2plbm':
        # Method Params Required, and default values
        # force_positive = True
        # fit_above_percentile = 99.5
        # n_proc = 1
        # estimate_gamma=True
        # peak_width=1
        # sigma=0.01
        # m_penalty=0
        
        field_sizes = np.asarray([field.shape for field in scan])
        assert len(np.unique(field_sizes[:,1])) == 1, "field stiching not possible"
        shape_of_stiched_fields = (np.sum(np.unique(field_sizes[:,0])), field_sizes[0,1])
        all_stiched_fields = np.empty((depth_amt, *shape_of_stiched_fields))

        for depth_idx in tqdm(range(depth_amt)):
            fields_in_a_depth = np.arange(depth_idx, depth_amt*field_amt, depth_amt)
            all_stiched_fields[depth_idx] = np.vstack([np.mean(scan[int(field)], axis=-1) for field in fields_in_a_depth])
            
        if method_params['force_positive']:
            all_stiched_fields = all_stiched_fields - all_stiched_fields.min(axis=(1,2),keepdims=True)

        assert all_stiched_fields.shape[0] == 30

        m_opts = [] 
        m_firsts = []
        all_liks = []
        m_opt_liks = []
        m_first_liks = []
        ms = np.linspace(0,1,101)

        for idx, i in enumerate(range(depth_amt//2)):
            X = all_stiched_fields[i].flatten()
            Y = all_stiched_fields[i+15].flatten()
            idxs = X > np.percentile(X, method_params['fit_above_percentile'])

            if method_params['n_proc'] == 1:
                liks = np.array([sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0 = method_params['sigma'], m_penalty=method_params['m_penalty']) for m in ms])
            else:
                p = Pool(method_params['n_proc'])
                liks = p.starmap(sum_log_lik_one_line,[(m, X[idxs], Y[idxs],0, method_params['sigma'],1e-10,method_params['m_penalty']) for m in ms])
                liks = np.array(liks)

            m_opt = ms[np.argmin(liks)]
            pks = find_peaks(-liks, width=method_params['peak_width'])[0]
            m_first = ms[pks[0]]

            m_opts.append(m_opt)
            m_firsts.append(m_first)
            all_liks.append(liks)
            m_opt_liks.append(liks.min())
            m_first_liks.append(liks[pks[0]])

        m_opts = np.array(m_opts)
        m_firsts = np.array(m_firsts)

        best_ms = m_opts[m_opts==m_firsts]
        best_m = best_ms.mean()

        if method_params['estimate_gamma']:
            gx = gamma.fit(m_opts)
            x = np.linspace(0,1,1001)
            gs = gamma.pdf(x, *gx)
            best_m = x[np.argmax(gs)]
            
        return m_opts, best_m
    
    elif method == 'mosaic':
        mosaic = mp.MosaicPicasso(bins = method_params['bins'], 
                                  beta = method_params['beta'], 
                                  gamma = method_params['gamma'], 
                                  cycles = method_params['cycles'], 
                                  nch = method_params['nch'], 
                                  threshold  = method_params['threshold'])
        all_tups = []
        for i in np.arange(15):
            tupA = {}
            tupB = {}

            if scan._islbm():
                print('Not Supported Yet')
                return [], []

            print(f"\n      {i}-{i+15} Prepping images...")
            source = np.mean(scan[:,:,:,channel_order[i],:],axis=(-1))
            target = np.mean(scan[:,:,:,channel_order[i+15],:],axis=(-1))
            im = np.array(np.concatenate(np.stack((source,target),axis=-1),axis=0))

            print(f"       {i}-{i+15} Running Mosaic...")
            im_mosaic,P = mosaic.mosaic(im)

            tupA['file'] = tiff_file
            tupA['depth'] = i
            tupA['cont_depth'] = i+15
            tupA['mean_img'] = source
            tupA['self_mod'] = P[0][0]
            tupA['cont_mod'] = P[0][1]
            
            tupA[f"im_mosaic"] = [im_mosaic]
            tupA[f"im_mosaic_pair"] = [i, i+15]
            
            tupB['file'] = tiff_file
            tupB['depth'] = i+15
            tupB['cont_depth'] = i
            tupB['mean_img'] = target
            tupB['self_mod'] = P[1][1]
            tupB['cont_mod'] = P[1][0]        
            
            tupB[f"im_mosaic"] = [im_mosaic]
            tupB[f"im_mosaic_pair"] = [i, i+15]
            
            all_tups.append(tupA)
            all_tups.append(tupB)
                
        print(f"   Saving...")
        my_df = pd.DataFrame(all_tups)
        my_df.to_pickle(results_path+results_file_name)

    else:
        print(f"{method}: Not Implemented")
        return [], []
    # Return entire dataframe??
    
    
def chan_to_fields_at_depth(depth, fields_per_depth, depth_amt=30):

    all_fields_at_depth = []
    for fields in np.arange(fields_per_depth):
        all_fields_at_depth.append(depth + fields*depth_amt)

    return all_fields_at_depth
'''