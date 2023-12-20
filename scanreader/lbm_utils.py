from multiprocessing import shared_memory, Pool
from scipy.signal import find_peaks
from scipy.stats import gamma
import numpy as np
from tqdm import tqdm

def sum_log_lik_one_line(m, x, y, b = 0, sigma_0 = 10,  c = 1e-10, m_penalty=0):
    mu = m * x + b
    lik_line = gaussian(y, mu, sigma_0)
    lik = lik_line
    
    log_lik = np.log(lik + c - m * m_penalty).sum()
    return -log_lik

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2*np.pi))


def calculate_crosstalk_coeff(scan, method, field_amt, method_params=None, depth_amt=30,
                                    force_positive=True, fit_above_percentile=99.5, 
                                    n_proc=1,estimate_gamma=True,peak_width = 1,
                                    sigma=0.01, m_penalty = 0):
    
    if method == 's2plbm':

        field_sizes = np.asarray([field.shape for field in scan])
        assert len(np.unique(field_sizes[:,1])) == 1, "field stiching not possible"
        shape_of_stiched_fields = (np.sum(np.unique(field_sizes[:,0])), field_sizes[0,1])
        all_stiched_fields = np.empty((depth_amt, *shape_of_stiched_fields))

        for depth_idx in tqdm(range(depth_amt)):
            fields_in_a_depth = np.arange(depth_idx, depth_amt*field_amt, depth_amt)
            all_stiched_fields[depth_idx] = np.vstack([np.mean(scan[int(field)], axis=-1) for field in fields_in_a_depth])
            
        if force_positive:
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
            idxs = X > np.percentile(X, fit_above_percentile)

            if n_proc == 1:
                liks = np.array([sum_log_lik_one_line(m, X[idxs], Y[idxs], sigma_0 = sigma, m_penalty=m_penalty) for m in ms])
            else:
                p = Pool(n_proc)
                liks = p.starmap(sum_log_lik_one_line,[(m, X[idxs], Y[idxs],0, sigma,1e-10,m_penalty) for m in ms])
                liks = np.array(liks)

            m_opt = ms[np.argmin(liks)]
            pks = find_peaks(-liks, width=peak_width)[0]
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

        if estimate_gamma:
            gx = gamma.fit(m_opts)
            x = np.linspace(0,1,1001)
            gs = gamma.pdf(x, *gx)
            best_m = x[np.argmax(gs)]
            
        return m_opts, best_m

    else:
        print(f"{method}: Not Implemented")
        return [], []
