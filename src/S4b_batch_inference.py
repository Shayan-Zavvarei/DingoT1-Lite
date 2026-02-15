import torch
import numpy as np
import h5py
import os
from tqdm import tqdm

import sys
sys.path.append('.')

from importlib import import_module
try:
    spec = import_module('5_inference_fixed')
    InferenceEngine = spec.InferenceEngine
except Exception as e:
    sys.exit(1)


def run_batch_inference(model_path, data_path, output_path, 
                        n_events=10, n_samples_per_event=5000):
    engine = InferenceEngine(model_path, device='cpu')
    
    with h5py.File(data_path, 'r') as f:
        total_samples = f['tokens'].shape[0]
        n_events = min(n_events, total_samples)
        
        tokens_all = f['tokens'][:n_events]
        
        true_params_all = {}
        if 'parameters' in f:
            for key in engine.param_names:
                if key in f['parameters']:
                    true_params_all[key] = f['parameters'][key][:n_events]
    
    all_samples = {key: [] for key in engine.param_names}
    all_tokens = []
    
    for i in tqdm(range(n_events), desc="Processing events"):
        tokens = tokens_all[i:i+1]
        
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            samples_dict = engine.run_inference(tokens, n_samples=n_samples_per_event)
        
        for key in engine.param_names:
            all_samples[key].append(samples_dict[key])
        
        all_tokens.append(tokens[0])
    
    for key in all_samples:
        all_samples[key] = np.array(all_samples[key])
    
    all_tokens = np.array(all_tokens)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        samples_grp = f.create_group('samples')
        for key, val in all_samples.items():
            samples_grp.create_dataset(key, data=val)
        
        f.create_dataset('tokens', data=all_tokens)
        
        if true_params_all:
            true_grp = f.create_group('true_parameters')
            for key, val in true_params_all.items():
                true_grp.create_dataset(key, data=val)
        
        f.attrs['n_events'] = n_events
        f.attrs['n_samples_per_event'] = n_samples_per_event
        f.attrs['n_params'] = len(engine.param_names)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch inference for importance sampling')
    parser.add_argument('--model', type=str, default='./models/light_model_v2.pt',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, default='./data/val_tokenized_multidet.h5',
                        help='Path to validation data')
    parser.add_argument('--output', type=str, default='./results/inference_results_multidet.h5',
                        help='Output file')
    parser.add_argument('--n_events', type=int, default=10,
                        help='Number of events to process')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Posterior samples per event')
    
    args = parser.parse_args()
    
    run_batch_inference(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        n_events=args.n_events,
        n_samples_per_event=args.n_samples
    )
