import h5py
import numpy as np


with h5py.File('./results/inference_results_multidet.h5', 'r') as f_in:
    samples_flat = {}
    for key in f_in['samples'].keys():
        samples_flat[key] = f_in['samples'][key][:]
    
    tokens_flat = f_in['tokens'][:]
    
    true_params_flat = {}
    if 'true_parameters' in f_in:
        for key in f_in['true_parameters'].keys():
            true_params_flat[key] = f_in['true_parameters'][key][:]
    
    n_events = samples_flat['mass_1'].shape[0]
    n_samples = samples_flat['mass_1'].shape[1]


with h5py.File('./results/inference_results_multidet_restructured.h5', 'w') as f_out:
    for i in range(n_events):
        grp = f_out.create_group(f'event_{i}')
        
        samples_grp = grp.create_group('samples')
        for key, val in samples_flat.items():
            samples_grp.create_dataset(key, data=val[i])
        
        grp.create_dataset('tokens', data=tokens_flat[i])
        
        if true_params_flat:
            true_grp = grp.create_group('true_parameters')
            for key, val in true_params_flat.items():
                true_grp.create_dataset(key, data=val[i])
    
    f_out.attrs['n_events'] = n_events
    f_out.attrs['n_samples_per_event'] = n_samples


import os
os.rename('./results/inference_results_multidet.h5', 
          './results/inference_results_multidet_backup.h5')
os.rename('./results/inference_results_multidet_restructured.h5', 
          './results/inference_results_multidet.h5')
