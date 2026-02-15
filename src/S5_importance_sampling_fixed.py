import torch
import numpy as np
import h5py
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pycbc.waveform as wf
from pycbc.detector import Detector


class MultiDetectorLikelihoodComputer:
    
    def __init__(self, f_min=20.0, f_max=2048.0, delta_f=0.125):
        self.f_min = f_min
        self.f_max = f_max
        self.delta_f = delta_f
        
        self.detectors = {
            'H1': Detector('H1'),
            'L1': Detector('L1'),
            'V1': Detector('V1')
        }
    
    def compute_inner_product(self, a, b, psd):
        min_len = min(len(a), len(b), len(psd))
        a = a[:min_len]
        b = b[:min_len]
        psd = np.clip(psd[:min_len], 1e-48, None)
        
        integrand = np.conj(a) * b / psd
        inner_prod = 4.0 * self.delta_f * np.sum(np.real(integrand))
        
        return inner_prod
    
    def generate_waveform(self, params):
        try:
            hp, hc = wf.get_fd_waveform(
                approximant='IMRPhenomXPHM',
                mass1=params['mass_1'],
                mass2=params['mass_2'],
                spin1z=params['spin_1z'],
                spin2z=params['spin_2z'],
                distance=1000.0,
                inclination=0.0,
                coa_phase=0.0,
                delta_f=self.delta_f,
                f_lower=self.f_min,
                f_final=self.f_max
            )
            
            return np.array(hp, dtype=np.complex128), np.array(hc, dtype=np.complex128)
        
        except Exception as e:
            return None, None
    
    def project_to_detector(self, hp, hc, params, detector_name):
        det = self.detectors[detector_name]
        
        fp, fc = det.antenna_pattern(
            params['ra'],
            params['dec'],
            params['psi'],
            params['geocent_time']
        )
        
        h_det = fp * hp + fc * hc
        
        dt = det.time_delay_from_earth_center(
            params['ra'],
            params['dec'],
            params['geocent_time']
        )
        
        freqs = np.arange(len(h_det)) * self.delta_f
        phase_shift = np.exp(-2j * np.pi * freqs * dt)
        h_det = h_det * phase_shift
        
        h_det = h_det * (1000.0 / params['luminosity_distance'])
        
        return h_det
    
    def compute_multi_detector_log_likelihood(self, data_dict, psd_dict, params):
        try:
            hp, hc = self.generate_waveform(params)
            
            if hp is None:
                return -1e10
            
            log_L_total = 0.0
            
            for det_name in ['H1', 'L1', 'V1']:
                if det_name not in data_dict or det_name not in psd_dict:
                    continue
                
                h = self.project_to_detector(hp, hc, params, det_name)
                
                d = data_dict[det_name].astype(np.complex128)
                p = psd_dict[det_name].astype(np.float64)
                
                d_h = self.compute_inner_product(d, h, p)
                h_h = self.compute_inner_product(h, h, p)
                d_d = self.compute_inner_product(d, d, p)
                
                log_L_det = -0.5 * (d_d - 2*d_h + h_h)
                log_L_total += log_L_det
            
            return log_L_total
        
        except Exception as e:
            return -1e10
    
    def compute_log_prior(self, params, prior_ranges):
        log_prior = 0.0
        
        for key, (min_val, max_val) in prior_ranges.items():
            if key in params:
                val = params[key]
                if val < min_val or val > max_val:
                    return -np.inf
                log_prior += -np.log(max_val - min_val)
        
        return log_prior


class ImprovedImportanceSamplerMultiDet:
    
    def __init__(self):
        self.likelihood_computer = MultiDetectorLikelihoodComputer()
        
        from config_fixed import PRIOR_RANGES
        self.prior_ranges = PRIOR_RANGES
    
    def importance_sample(self, samples_dict, data_dict, psd_dict,
                         n_max_samples=10000, keep_top_percent=30):
        n_samples = len(samples_dict['mass_1'])
        n_compute = min(n_samples, n_max_samples)
        
        log_L = np.zeros(n_compute)
        log_prior = np.zeros(n_compute)
        
        for i in tqdm(range(n_compute), desc="Computing likelihoods"):
            params = {k: float(v[i]) for k, v in samples_dict.items()}
            
            log_L[i] = self.likelihood_computer.compute_multi_detector_log_likelihood(
                data_dict, psd_dict, params
            )
            log_prior[i] = self.likelihood_computer.compute_log_prior(
                params, self.prior_ranges
            )
        
        n_failed = (log_L == -1e10).sum()
        valid_log_L = log_L[log_L > -1e9]
        
        valid_mask = (log_L > -1e9) & (log_prior > -np.inf)
        log_L_valid = log_L[valid_mask]
        
        if len(log_L_valid) < 10:
            keep_mask = valid_mask
        else:
            threshold = np.percentile(log_L_valid, 100 - keep_top_percent)
            keep_mask = (log_L >= threshold) & valid_mask
        
        n_kept = keep_mask.sum()
        
        if n_kept < 10:
            sorted_idx = np.argsort(log_L)[::-1]
            keep_mask = np.zeros(n_compute, dtype=bool)
            keep_mask[sorted_idx[:min(100, n_compute)]] = True
            n_kept = keep_mask.sum()
        
        samples_filtered = {k: v[:n_compute][keep_mask] for k, v in samples_dict.items()}
        log_L_filtered = log_L[keep_mask]
        log_prior_filtered = log_prior[keep_mask]
        
        log_w = log_prior_filtered + log_L_filtered
        log_w = log_w - log_w.max()
        
        w = np.exp(log_w)
        w = w / w.sum()
        
        ess = 1.0 / np.sum(w**2)
        efficiency = ess / n_kept * 100
        
        return w, samples_filtered, ess, efficiency
    
    def resample(self, samples_filtered, weights, n_output=5000):
        n_valid = len(weights)
        
        if n_valid == 0:
            raise ValueError("No valid samples to resample!")
        
        indices = np.random.choice(n_valid, size=n_output, replace=True, p=weights)
        
        resampled = {k: v[indices] for k, v in samples_filtered.items()}
        
        return resampled


def extract_detector_data(tokens_all_detectors):
    data_dict = {}
    psd_dict = {}
    
    detector_names = ['H1', 'L1', 'V1']
    
    for det_idx, det_name in enumerate(detector_names):
        det_mask = tokens_all_detectors[:, 0, 5] == det_idx
        tokens_det = tokens_all_detectors[det_mask]
        
        if len(tokens_det) == 0:
            continue
        
        strain_segments = []
        psd_segments = []
        
        for token in tokens_det:
            real_part = token[:, 0]
            imag_part = token[:, 1]
            psd_part = token[:, 2]
            
            strain_seg = real_part + 1j * imag_part
            strain_segments.append(strain_seg)
            psd_segments.append(psd_part)
        
        strain_full = np.concatenate(strain_segments)
        psd_full = np.concatenate(psd_segments)
        
        data_dict[det_name] = strain_full
        psd_dict[det_name] = psd_full
    
    return data_dict, psd_dict


def run_importance_sampling_pipeline_multidet(n_events=10):
    sampler = ImprovedImportanceSamplerMultiDet()
    
    with h5py.File('./results/inference_results_multidet.h5', 'r') as f:
        n_available = min(n_events, f.attrs['n_events'])
        
        all_samples = []
        for i in range(n_available):
            grp = f[f'event_{i}/samples']
            all_samples.append({k: grp[k][:] for k in grp.keys()})
    
    with h5py.File('./data/val_tokenized_multidet.h5', 'r') as f:
        tokens_all = f['tokens'][:n_available]
        
        from config_fixed import PARAMETER_NAMES
        true_params = []
        for i in range(n_available):
            tp = {k: float(f['parameters'][k][i]) for k in PARAMETER_NAMES}
            true_params.append(tp)
    
    results = []
    
    for i in range(n_available):
        samples = all_samples[i]
        tokens = tokens_all[i]
        
        data_dict, psd_dict = extract_detector_data(tokens)
        
        try:
            w, samples_filt, ess, eff = sampler.importance_sample(
                samples, data_dict, psd_dict,
                n_max_samples=10000,
                keep_top_percent=30
            )
        except Exception as e:
            continue
        
        resampled = sampler.resample(samples_filt, w, n_output=5000)
        
        improvements = 0
        
        key_params = ['mass_1', 'mass_2', 'spin_1z', 'spin_2z', 'luminosity_distance']
        
        for key in key_params:
            if key in true_params[i]:
                true_val = true_params[i][key]
                
                orig_mean = samples[key].mean()
                orig_err = abs(orig_mean - true_val) / (abs(true_val) + 1e-10) * 100
                
                res_mean = resampled[key].mean()
                res_err = abs(res_mean - true_val) / (abs(true_val) + 1e-10) * 100
                
                improvement = orig_err - res_err
                
                if improvement > 0:
                    improvements += 1
        
        results.append({
            'event_id': i,
            'resampled': resampled,
            'ess': ess,
            'efficiency': eff,
            'improvements': improvements
        })
    
    output_file = './results/importance_sampling_multidet.h5'
    
    with h5py.File(output_file, 'w') as f:
        f.attrs['n_events'] = len(results)
        
        for r in results:
            grp = f.create_group(f"event_{r['event_id']}")
            
            res_grp = grp.create_group('resampled')
            for k, v in r['resampled'].items():
                res_grp.create_dataset(k, data=v, compression='gzip')
            
            grp.attrs['ESS'] = r['ess']
            grp.attrs['efficiency'] = r['efficiency']
            grp.attrs['improvements'] = r['improvements']


if __name__ == '__main__':
    run_importance_sampling_pipeline_multidet(n_events=10)
