import numpy as np
import h5py
from tqdm import tqdm
from scipy import interpolate
from pycbc.detector import Detector


class MultiDetectorPreprocessor:
    
    def __init__(self, config):
        self.f_min = config['f_min']
        self.f_max = config['f_max']
        self.delta_f = config['delta_f']
        self.n_bins_per_token = 16
        
        self.detectors = {
            'H1': Detector('H1'),
            'L1': Detector('L1'),
            'V1': Detector('V1')
        }
        
        self.mb_nodes = None
    
    def design_multibanding_nodes(self, waveform_file, n_design=100):
        with h5py.File(waveform_file, 'r') as f:
            hp_samples = f['hp'][:n_design]
        
        n_nodes = 150
        
        f_low_end = 200
        n_low = int(n_nodes * 0.6)
        n_high = n_nodes - n_low
        
        nodes_low = np.linspace(self.f_min, f_low_end, n_low)
        nodes_high = np.linspace(f_low_end, self.f_max, n_high)
        
        self.mb_nodes = np.concatenate([nodes_low, nodes_high[1:]])
        
        n_segments = len(self.mb_nodes) // self.n_bins_per_token
        self.mb_nodes = self.mb_nodes[:n_segments * self.n_bins_per_token]
        
        compression = (self.f_max - self.f_min) / self.delta_f / len(self.mb_nodes)
        
        return self.mb_nodes
    
    def apply_multibanding(self, strain_fd, psd_fd, freqs):
        if self.mb_nodes is None:
            raise ValueError("Multibanding nodes not designed!")
        
        strain_real = np.interp(self.mb_nodes, freqs, np.real(strain_fd))
        strain_imag = np.interp(self.mb_nodes, freqs, np.imag(strain_fd))
        strain_mb = strain_real + 1j * strain_imag
        
        psd_mb = np.interp(self.mb_nodes, freqs, psd_fd)
        
        return strain_mb, psd_mb
    
    def tokenize_single_detector(self, strain_mb, psd_mb, detector_id):
        n_nodes = len(self.mb_nodes)
        n_tokens = n_nodes // self.n_bins_per_token
        
        tokens = []
        
        for k in range(n_tokens):
            start = k * self.n_bins_per_token
            end = start + self.n_bins_per_token
            
            segment_strain = strain_mb[start:end]
            segment_psd = psd_mb[start:end]
            f_start = self.mb_nodes[start]
            f_end = self.mb_nodes[end-1]
            
            token = np.zeros((self.n_bins_per_token, 6))
            token[:, 0] = np.real(segment_strain)
            token[:, 1] = np.imag(segment_strain)
            token[:, 2] = segment_psd
            token[:, 3] = f_start
            token[:, 4] = f_end
            token[:, 5] = detector_id
            
            tokens.append(token)
        
        return np.array(tokens)
    
    def project_and_tokenize_all_detectors(self, hp, hc, extrinsic_params, psd_library):
        freqs = np.arange(self.f_min, self.f_max, self.delta_f)
        
        all_tokens = []
        
        detector_names = ['H1', 'L1', 'V1']
        
        for det_idx, det_name in enumerate(detector_names):
            det = self.detectors[det_name]
            
            fp, fc = det.antenna_pattern(
                extrinsic_params['ra'],
                extrinsic_params['dec'],
                extrinsic_params['psi'],
                extrinsic_params['geocent_time']
            )
            
            h_det = fp * hp + fc * hc
            
            dt = det.time_delay_from_earth_center(
                extrinsic_params['ra'],
                extrinsic_params['dec'],
                extrinsic_params['geocent_time']
            )
            
            min_len = min(len(h_det), len(freqs))
            h_det = h_det[:min_len]
            freqs_trunc = freqs[:min_len]
            
            phase_shift = np.exp(-2j * np.pi * freqs_trunc * dt)
            h_det = h_det * phase_shift
            
            h_det = h_det * (1000.0 / extrinsic_params['luminosity_distance'])
            
            psd_idx = np.random.randint(0, len(psd_library[det_name]))
            psd = psd_library[det_name][psd_idx][:min_len]
            
            noise_re = np.random.normal(0, 1, min_len)
            noise_im = np.random.normal(0, 1, min_len)
            noise_std = np.sqrt(psd * self.delta_f / 4.0)
            noise = (noise_re + 1j * noise_im) * noise_std
            
            strain = h_det + noise
            
            strain_mb, psd_mb = self.apply_multibanding(strain, psd, freqs_trunc)
            
            tokens_det = self.tokenize_single_detector(strain_mb, psd_mb, det_idx)
            
            all_tokens.append(tokens_det)
        
        all_tokens = np.concatenate(all_tokens, axis=0)
        
        return all_tokens
    
    def process_dataset_with_extrinsics(self, input_file, output_file, 
                                        psd_library_file, n_extrinsic=5):
        with h5py.File(psd_library_file, 'r') as f:
            psd_library = {
                'H1': f['H1'][:],
                'L1': f['L1'][:],
                'V1': f['V1'][:]
            }
        
        with h5py.File(input_file, 'r') as f:
            hp_all = f['hp'][:]
            hc_all = f['hc'][:]
            
            intrinsic_params = {}
            for key in f['intrinsic_parameters'].keys():
                intrinsic_params[key] = f['intrinsic_parameters'][key][:]
        
        n_intrinsic = len(hp_all)
        n_total = n_intrinsic * n_extrinsic
        
        all_tokens = []
        all_params = {k: [] for k in (list(intrinsic_params.keys()) + 
                                      ['luminosity_distance', 'inclination', 'ra', 
                                       'dec', 'phase', 'psi', 'geocent_time'])}
        
        for i in tqdm(range(n_intrinsic), desc="Processing waveforms"):
            hp = hp_all[i]
            hc = hc_all[i]
            
            intr_p = {k: intrinsic_params[k][i] for k in intrinsic_params.keys()}
            
            for j in range(n_extrinsic):
                extr_p = {
                    'luminosity_distance': np.random.uniform(100.0, 6000.0),
                    'inclination': np.arccos(np.random.uniform(-1, 1)),
                    'ra': np.random.uniform(0, 2*np.pi),
                    'dec': np.arcsin(np.random.uniform(-1, 1)),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'psi': np.random.uniform(0, np.pi),
                    'geocent_time': np.random.uniform(-0.1, 0.1)
                }
                
                try:
                    tokens = self.project_and_tokenize_all_detectors(
                        hp, hc, extr_p, psd_library
                    )
                    
                    all_tokens.append(tokens)
                    
                    for k, v in intr_p.items():
                        all_params[k].append(v)
                    for k, v in extr_p.items():
                        all_params[k].append(v)
                
                except Exception as e:
                    continue
        
        n_successful = len(all_tokens)
        
        max_tokens = max(len(t) for t in all_tokens)
        
        tokens_padded = np.zeros((n_successful, max_tokens, 16, 6), dtype=np.float32)
        for i, tokens in enumerate(all_tokens):
            tokens_padded[i, :len(tokens)] = tokens
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('tokens', data=tokens_padded, compression='gzip')
            
            params_grp = f.create_group('parameters')
            for k, v in all_params.items():
                params_grp.create_dataset(k, data=np.array(v), compression='gzip')
            
            f.attrs['n_samples'] = n_successful
            f.attrs['max_tokens'] = max_tokens
            f.attrs['n_detectors'] = 3
            f.attrs['multibanding_nodes'] = self.mb_nodes


def run_multidet_preprocessing():
    from config_fixed import DATA_CONFIG
    
    config = DATA_CONFIG
    
    preprocessor = MultiDetectorPreprocessor(config)
    
    preprocessor.design_multibanding_nodes(
        './data/train_waveforms_fixed.h5',
        n_design=100
    )
    
    preprocessor.process_dataset_with_extrinsics(
        input_file='./data/train_waveforms_fixed.h5',
        output_file='./data/train_tokenized_multidet.h5',
        psd_library_file='./data/psd_library.h5',
        n_extrinsic=5
    )
    
    preprocessor.process_dataset_with_extrinsics(
        input_file='./data/val_waveforms_fixed.h5',
        output_file='./data/val_tokenized_multidet.h5',
        psd_library_file='./data/psd_library.h5',
        n_extrinsic=3
    )


if __name__ == '__main__':
    run_multidet_preprocessing()
