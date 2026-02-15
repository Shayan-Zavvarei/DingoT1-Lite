"""
load_real_gwosc_data.py
================================================================================
تبدیل دیتای واقعی GWOSC به فرمت قابل استفاده برای model
"""

import numpy as np
import h5py
from gwpy.timeseries import TimeSeries
from pycbc.psd import welch
from pycbc.types import TimeSeries as PyCBCTimeSeries
import matplotlib.pyplot as plt
import os


class GWOSCDataLoader:
    """
    Load و preprocess دیتای واقعی GWOSC
    """
    
    def __init__(self, f_min=20.0, f_max=2048.0, delta_f=0.125):
        self.f_min = f_min
        self.f_max = f_max
        self.delta_f = delta_f
        
        print("✓ GWOSCDataLoader initialized")
    
    def load_strain_from_gwosc_file(self, file_path, detector='H1'):
        """
        Load strain data از HDF5 file واقعی GWOSC
        
        Args:
            file_path: path to GWOSC HDF5 file
            detector: 'H1', 'L1', or 'V1'
        
        Returns:
            strain_td: time-domain strain
            gps_start: GPS start time
            sample_rate: sampling rate
        """
        print(f"\n📂 Loading {detector} strain from: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            # GWOSC format
            if 'strain' in f:
                strain_td = f['strain/Strain'][:]
                
                # Metadata
                if 'meta' in f:
                    gps_start = f['meta'].attrs.get('GPSstart', 0)
                    sample_rate = f['meta'].attrs.get('SamplingRate', 4096)
                else:
                    gps_start = 0
                    sample_rate = 4096
            
            # یا فرمت دیگر
            elif 'data' in f:
                strain_td = f['data'][:]
                gps_start = f.attrs.get('start_time', 0)
                sample_rate = f.attrs.get('sample_rate', 4096)
            
            else:
                raise ValueError("Unknown HDF5 format! Check file structure.")
        
        print(f"  ✓ Loaded {len(strain_td)} samples")
        print(f"    GPS start: {gps_start}")
        print(f"    Sample rate: {sample_rate} Hz")
        print(f"    Duration: {len(strain_td)/sample_rate:.1f} seconds")
        
        return strain_td, gps_start, sample_rate
    
    def download_event_data_gwosc(self, event_name='GW150914', duration=32, detectors=['H1', 'L1']):
        """
        دانلود مستقیم از GWOSC API (alternative)
        
        Args:
            event_name: نام event (e.g., 'GW150914', 'GW170817')
            duration: مدت زمان data (seconds)
            detectors: list of detectors
        
        Returns:
            data_dict: {'H1': strain, 'L1': strain, ...}
        """
        try:
            from gwosc.datasets import event_gps
            from gwosc import datasets
        except ImportError:
            print("❌ gwosc package not installed! Install: pip install gwosc")
            return None
        
        print(f"\n🌐 Downloading {event_name} data from GWOSC...")
        
        # Get event GPS time
        try:
            gps_time = event_gps(event_name)
            print(f"  Event GPS time: {gps_time}")
        except:
            print(f"  ⚠️ Event {event_name} not found in catalog")
            return None
        
        data_dict = {}
        
        for det in detectors:
            print(f"\n  Downloading {det}...")
            
            try:
                # Load strain data
                strain = TimeSeries.fetch_open_data(
                    det,
                    gps_time - duration/2,
                    gps_time + duration/2,
                    sample_rate=4096,
                    cache=True
                )
                
                data_dict[det] = {
                    'strain': np.array(strain.value),
                    'times': np.array(strain.times.value),
                    'gps_start': strain.t0.value,
                    'sample_rate': strain.sample_rate.value
                }
                
                print(f"    ✓ Downloaded {len(strain)} samples")
            
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                continue
        
        return data_dict
    
    def whiten_and_bandpass(self, strain_td, sample_rate, fmin=20, fmax=500):
        """
        Whiten و bandpass filter
        """
        print("\n🔧 Whitening and bandpassing...")
        
        # Convert to PyCBC TimeSeries
        strain_pycbc = PyCBCTimeSeries(strain_td, delta_t=1.0/sample_rate)
        
        # Compute PSD با Welch method
        psd = welch(strain_pycbc, seg_len=int(4*sample_rate))
        
        # Whiten
        from pycbc.filter import highpass, lowpass_fir
        
        # Highpass
        strain_hp = highpass(strain_pycbc, fmin)
        
        # Lowpass
        strain_bp = lowpass_fir(strain_hp, fmax, 8)
        
        # Whiten
        from pycbc.types import FrequencySeries
        psd_interp = psd.interpolate(strain_bp.delta_f)
        
        strain_fd = strain_bp.to_frequencyseries()
        strain_whitened_fd = strain_fd / (psd_interp**0.5)
        
        strain_whitened_td = strain_whitened_fd.to_timeseries()
        
        print(f"  ✓ Whitened")
        
        return np.array(strain_whitened_td), np.array(psd)
    
    def convert_to_frequency_domain(self, strain_td, sample_rate):
        """
        تبدیل به frequency domain
        """
        print("\n🔄 Converting to frequency domain...")
        
        # FFT
        strain_fd = np.fft.rfft(strain_td) / sample_rate
        freqs = np.fft.rfftfreq(len(strain_td), 1.0/sample_rate)
        
        # Crop to desired range
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        strain_fd_crop = strain_fd[mask]
        freqs_crop = freqs[mask]
        
        print(f"  ✓ {len(freqs_crop)} frequency bins")
        
        return strain_fd_crop, freqs_crop
    
    def estimate_psd(self, strain_td, sample_rate, seg_duration=4):
        """
        تخمین PSD با Welch method
        """
        print("\n📊 Estimating PSD...")
        
        # PyCBC Welch
        strain_pycbc = PyCBCTimeSeries(strain_td, delta_t=1.0/sample_rate)
        psd = welch(strain_pycbc, seg_len=int(seg_duration * sample_rate))
        
        # Interpolate to desired delta_f
        psd_freqs = np.arange(self.f_min, self.f_max, self.delta_f)
        psd_interp = np.interp(psd_freqs, psd.sample_frequencies.numpy(), psd.numpy())
        
        print(f"  ✓ PSD estimated ({len(psd_interp)} bins)")
        
        return psd_interp, psd_freqs
    
    def process_real_event_to_tokens(self, event_data_dict, mb_nodes):
        """
        تبدیل کامل real event → tokens
        
        Args:
            event_data_dict: output از download_event_data_gwosc
            mb_nodes: multibanding nodes (load از training)
        
        Returns:
            tokens: [L, 16, 6] array
        """
        from preprocessing_multidet_fixed import MultiDetectorPreprocessor
        
        print("\n" + "="*70)
        print("PROCESSING REAL EVENT TO TOKENS")
        print("="*70)
        
        # Load config
        from config_fixed import DATA_CONFIG
        preprocessor = MultiDetectorPreprocessor(DATA_CONFIG)
        preprocessor.mb_nodes = mb_nodes
        
        all_tokens = []
        
        for det_idx, det_name in enumerate(['H1', 'L1', 'V1']):
            if det_name not in event_data_dict:
                print(f"\n⚠️ {det_name} data not available, skipping...")
                continue
            
            print(f"\n🔧 Processing {det_name}...")
            
            det_data = event_data_dict[det_name]
            strain_td = det_data['strain']
            sample_rate = det_data['sample_rate']
            
            # Convert to FD
            strain_fd, freqs = self.convert_to_frequency_domain(strain_td, sample_rate)
            
            # Estimate PSD
            psd, psd_freqs = self.estimate_psd(strain_td, sample_rate)
            
            # Multibanding
            strain_mb, psd_mb = preprocessor.apply_multibanding(strain_fd, psd, freqs)
            
            # Tokenize
            tokens_det = preprocessor.tokenize_single_detector(strain_mb, psd_mb, det_idx)
            
            all_tokens.append(tokens_det)
            
            print(f"  ✓ {len(tokens_det)} tokens generated")
        
        # Concatenate all detectors
        tokens_combined = np.concatenate(all_tokens, axis=0)
        
        print(f"\n✓ Total tokens: {len(tokens_combined)} (across {len(all_tokens)} detectors)")
        
        return tokens_combined


def example_usage_gwosc_file():
    """
    مثال 1: Load از HDF5 file محلی
    """
    print("="*70)
    print("EXAMPLE 1: LOADING FROM LOCAL GWOSC FILE")
    print("="*70)
    
    loader = GWOSCDataLoader()
    
    # فرض: شما یک file دانلود کرده‌اید
    file_path = './data/H-H1_LOSC_4_V1-1126259446-32.hdf5'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("   Download from: https://gwosc.org/events/")
        return
    
    # Load strain
    strain_td, gps_start, sample_rate = loader.load_strain_from_gwosc_file(file_path, 'H1')
    
    # Convert to FD
    strain_fd, freqs = loader.convert_to_frequency_domain(strain_td, sample_rate)
    
    # Estimate PSD
    psd, psd_freqs = loader.estimate_psd(strain_td, sample_rate)
    
    print("\n✓ Data loaded and processed!")


def example_usage_gwosc_api():
    """
    مثال 2: دانلود مستقیم از GWOSC
    """
    print("="*70)
    print("EXAMPLE 2: DOWNLOADING FROM GWOSC API")
    print("="*70)
    
    loader = GWOSCDataLoader()
    
    # Download GW150914
    event_data = loader.download_event_data_gwosc(
        event_name='GW150914',
        duration=32,
        detectors=['H1', 'L1']
    )
    
    if event_data is None:
        return
    
    # Load multibanding nodes from training
    print("\n📂 Loading multibanding nodes from training...")
    
    try:
        with h5py.File('./data/train_tokenized_multidet.h5', 'r') as f:
            mb_nodes = f.attrs['multibanding_nodes']
        print(f"  ✓ Loaded {len(mb_nodes)} multibanding nodes")
    except:
        print("  ⚠️ Training data not found, using default nodes")
        mb_nodes = np.linspace(20, 2048, 150)
    
    # Process to tokens
    tokens = loader.process_real_event_to_tokens(event_data, mb_nodes)
    
    # Save
    output_file = './data/GW150914_tokens.h5'
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('tokens', data=tokens)
        f.attrs['event_name'] = 'GW150914'
    
    print(f"\n✓ Tokens saved to: {output_file}")
    
    return tokens


def run_inference_on_real_event(tokens, model_path):
    """
    اجرای inference روی real event
    """
    print("\n" + "="*70)
    print("RUNNING INFERENCE ON REAL EVENT")
    print("="*70)
    
    from inference_multidet_fixed import InferenceEngineMultiDet
    
    # Initialize inference
    inference = InferenceEngineMultiDet(model_path, device='cuda')
    
    # Generate samples
    samples = inference.generate_samples(tokens, n_samples=10000)
    
    # Print results
    print("\n📊 POSTERIOR STATISTICS:")
    print("-" * 70)
    
    from config_fixed import PARAMETER_NAMES
    
    for param in PARAMETER_NAMES[:8]:
        mean = samples[param].mean()
        std = samples[param].std()
        median = np.median(samples[param])
        
        print(f"{param:25s}: {mean:9.3f} ± {std:7.3f} (median: {median:9.3f})")
    
    return samples


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        # Download and process
        tokens = example_usage_gwosc_api()
        
        if tokens is not None and os.path.exists('./models/best_model_multidet.pt'):
            # Run inference
            samples = run_inference_on_real_event(
                tokens,
                './models/best_model_multidet.pt'
            )
    
    else:
        print("\nUsage:")
        print("  python load_real_gwosc_data.py download  # Download GW150914 and run inference")
        print("\nOr modify the script to load your own HDF5 files")
