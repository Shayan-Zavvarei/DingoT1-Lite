import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats, signal
from scipy.ndimage import gaussian_filter
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_strain_signal_noise(event_id=0, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./data/detector_strain_full.h5', 'r') as f:
            signal_H1 = f['signals/H1'][event_id]
            signal_L1 = f['signals/L1'][event_id]
            signal_V1 = f['signals/V1'][event_id]
            
            noise_H1 = f['noise/H1'][event_id]
            noise_L1 = f['noise/L1'][event_id]
            noise_V1 = f['noise/V1'][event_id]
            
            strain_H1 = f['strain/H1'][event_id]
            strain_L1 = f['strain/L1'][event_id]
            strain_V1 = f['strain/V1'][event_id]
            
            sample_rate = f.attrs['sample_rate']
            duration = f.attrs['duration']
            
            params = {}
            for key in f['parameters'].keys():
                params[key] = f['parameters'][key][event_id]
    except:
        return
    
    dt = 1.0 / sample_rate
    times = np.arange(0, duration, dt)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    
    detectors = ['H1 (Hanford)', 'L1 (Livingston)', 'V1 (Virgo)']
    signals = [signal_H1, signal_L1, signal_V1]
    noises = [noise_H1, noise_L1, noise_V1]
    strains = [strain_H1, strain_L1, strain_V1]
    colors = ['blue', 'green', 'red']
    
    for i, (det, sig, noi, str_data, color) in enumerate(zip(detectors, signals, noises, strains, colors)):
        ax = axes[i, 0]
        ax.plot(times, sig, color=color, alpha=0.9, linewidth=0.8)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'{det} - Clean Signal', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.5, 2.5)
        if i == 2:
            ax.set_xlabel('Time (s)', fontsize=11)
        
        ax = axes[i, 1]
        ax.plot(times, noi, color='gray', alpha=0.7, linewidth=0.5)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'{det} - Detector Noise', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.5, 2.5)
        if i == 2:
            ax.set_xlabel('Time (s)', fontsize=11)
        
        ax = axes[i, 2]
        ax.plot(times, str_data, color=color, alpha=0.7, linewidth=0.6)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'{det} - Total Strain', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.5, 2.5)
        if i == 2:
            ax.set_xlabel('Time (s)', fontsize=11)
    
    plt.suptitle(f'Event {event_id} - Signal + Noise = Strain Decomposition\n' +
                 f'M1={params.get("mass_1", 0):.1f} M☉, M2={params.get("mass_2", 0):.1f} M☉, ' +
                 f'DL={params.get("luminosity_distance", 0):.0f} Mpc',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'strain_decomposition_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_frequency_domain(event_id=0, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./data/detector_strain_full.h5', 'r') as f:
            signal_H1 = f['signals/H1'][event_id]
            noise_H1 = f['noise/H1'][event_id]
            strain_H1 = f['strain/H1'][event_id]
            
            signal_L1 = f['signals/L1'][event_id]
            noise_L1 = f['noise/L1'][event_id]
            strain_L1 = f['strain/L1'][event_id]
            
            signal_V1 = f['signals/V1'][event_id]
            noise_V1 = f['noise/V1'][event_id]
            strain_V1 = f['strain/V1'][event_id]
            
            sample_rate = f.attrs['sample_rate']
    except:
        return
    
    dt = 1.0 / sample_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    detectors = ['H1', 'L1', 'V1']
    signals = [signal_H1, signal_L1, signal_V1]
    noises = [noise_H1, noise_L1, noise_V1]
    strains = [strain_H1, strain_L1, strain_V1]
    colors = ['blue', 'green', 'red']
    
    for ax, det, sig, noi, strain, color in zip(axes, detectors, signals, noises, strains, colors):
        freq = np.fft.rfftfreq(len(sig), dt)
        
        fft_sig = np.abs(np.fft.rfft(sig))
        fft_noi = np.abs(np.fft.rfft(noi))
        fft_strain = np.abs(np.fft.rfft(strain))
        
        ax.loglog(freq[1:], fft_sig[1:], color=color, alpha=0.9, linewidth=2, label='Signal')
        ax.loglog(freq[1:], fft_noi[1:], color='gray', alpha=0.6, linewidth=1.5, label='Noise')
        ax.loglog(freq[1:], fft_strain[1:], color='black', alpha=0.7, linewidth=1, 
                  linestyle='--', label='Total Strain')
        
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(f'{det} - Frequency Domain', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(20, 1024)
    
    axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    
    plt.suptitle(f'Event {event_id} - Frequency Domain Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'frequency_domain_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_qtransforms(event_id=0, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./data/detector_strain_full.h5', 'r') as f:
            strains = [f['strain/H1'][event_id], 
                      f['strain/L1'][event_id], 
                      f['strain/V1'][event_id]]
            sample_rate = f.attrs['sample_rate']
    except:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    detectors = ['H1 (Hanford)', 'L1 (Livingston)', 'V1 (Virgo)']
    
    for ax, strain, det in zip(axes, strains, detectors):
        f, t, Sxx = signal.spectrogram(strain, fs=sample_rate, nperseg=256, 
                                        noverlap=250, mode='magnitude')
        
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-20), 
                           shading='gouraud', cmap='viridis', vmin=-60, vmax=-20)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_title(f'{det} - Q-Transform', fontsize=12, fontweight='bold')
        ax.set_ylim(20, 512)
        ax.set_xlim(1.0, 3.0)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)', fontsize=10)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.suptitle(f'Event {event_id} - Time-Frequency Analysis (Q-Transforms)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'qtransform_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_whitened_data(event_id=0, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./data/detector_strain_full.h5', 'r') as f:
            strains = [f['strain/H1'][event_id], 
                      f['strain/L1'][event_id], 
                      f['strain/V1'][event_id]]
            sample_rate = f.attrs['sample_rate']
            duration = f.attrs['duration']
    except:
        return
    
    dt = 1.0 / sample_rate
    times = np.arange(0, duration, dt)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    detectors = ['H1', 'L1', 'V1']
    colors = ['blue', 'green', 'red']
    
    for ax, strain, det, color in zip(axes, strains, detectors, colors):
        sos = signal.butter(4, [20, 512], btype='bandpass', fs=sample_rate, output='sos')
        whitened = signal.sosfilt(sos, strain)
        
        ax.plot(times, whitened, color=color, alpha=0.8, linewidth=0.8)
        ax.set_ylabel('Whitened Amplitude', fontsize=11)
        ax.set_title(f'{det} - Whitened Strain', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1.5, 2.5)
        ax.axvline(2.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Merger time')
        ax.legend(fontsize=10)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.suptitle(f'Event {event_id} - Whitened Strain Data',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'whitened_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_snr_analysis(save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./data/detector_strain_full.h5', 'r') as f:
            n_events = f['signals/H1'].shape[0]
            
            snrs_H1 = []
            snrs_L1 = []
            snrs_V1 = []
            snrs_network = []
            
            for i in range(n_events):
                sig_H1 = f['signals/H1'][i]
                noi_H1 = f['noise/H1'][i]
                snr_H1 = np.sqrt(np.sum(sig_H1**2) / (np.sum(noi_H1**2) + 1e-10))
                snrs_H1.append(snr_H1)
                
                sig_L1 = f['signals/L1'][i]
                noi_L1 = f['noise/L1'][i]
                snr_L1 = np.sqrt(np.sum(sig_L1**2) / (np.sum(noi_L1**2) + 1e-10))
                snrs_L1.append(snr_L1)
                
                sig_V1 = f['signals/V1'][i]
                noi_V1 = f['noise/V1'][i]
                snr_V1 = np.sqrt(np.sum(sig_V1**2) / (np.sum(noi_V1**2) + 1e-10))
                snrs_V1.append(snr_V1)
                
                snr_net = np.sqrt(snr_H1**2 + snr_L1**2 + snr_V1**2)
                snrs_network.append(snr_net)
    except:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax = axes[0, 0]
    ax.hist(snrs_H1, bins=20, alpha=0.5, label='H1', color='blue', edgecolor='black')
    ax.hist(snrs_L1, bins=20, alpha=0.5, label='L1', color='green', edgecolor='black')
    ax.hist(snrs_V1, bins=20, alpha=0.5, label='V1', color='red', edgecolor='black')
    ax.set_xlabel('SNR', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('SNR Distribution per Detector', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(snrs_network, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(np.median(snrs_network), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(snrs_network):.1f}')
    ax.set_xlabel('Network SNR', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Network SNR Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(snrs_H1, 'o-', label='H1', color='blue', alpha=0.7, markersize=5)
    ax.plot(snrs_L1, 's-', label='L1', color='green', alpha=0.7, markersize=5)
    ax.plot(snrs_V1, '^-', label='V1', color='red', alpha=0.7, markersize=5)
    ax.set_xlabel('Event ID', fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_title('SNR per Event (Individual Detectors)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(snrs_network, 'D-', color='purple', alpha=0.7, markersize=6, linewidth=2)
    ax.axhline(np.median(snrs_network), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(snrs_network):.1f}')
    ax.set_xlabel('Event ID', fontsize=12)
    ax.set_ylabel('Network SNR', fontsize=12)
    ax.set_title('Network SNR per Event', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Signal-to-Noise Ratio (SNR) Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'snr_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_tokens(event_id=0, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./data/val_tokenized_multidet.h5', 'r') as f:
            tokens = f['tokens'][event_id]
    except:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    detector_names = ['H1', 'L1', 'V1']
    
    for i in range(3):
        ax = axes[0, i]
        im = ax.imshow(tokens[:, :, i*2].T, aspect='auto', cmap='RdBu_r', 
                       interpolation='nearest')
        ax.set_title(f'{detector_names[i]} - Real Part', fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Index', fontsize=10)
        ax.set_ylabel('Frequency Bin', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax = axes[1, i]
        im = ax.imshow(tokens[:, :, i*2+1].T, aspect='auto', cmap='RdBu_r',
                       interpolation='nearest')
        ax.set_title(f'{detector_names[i]} - Imaginary Part', fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Index', fontsize=10)
        ax.set_ylabel('Frequency Bin', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Event {event_id} - Tokenized Representation ({tokens.shape[0]} tokens × 16 bins × 6 channels)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'tokens_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_posterior_comparison(event_id=0, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    param_names = ['mass_1', 'mass_2', 'spin_1z', 'spin_2z', 'luminosity_distance',
                   'inclination', 'ra', 'dec']
    
    try:
        with h5py.File('./results/inference_results_multidet.h5', 'r') as f:
            npe_samples = {}
            for key in param_names:
                npe_samples[key] = f['samples'][key][event_id]
            
            true_params = {}
            if 'true_parameters' in f:
                for key in param_names:
                    true_params[key] = f['true_parameters'][key][event_id]
    except:
        return
    
    try:
        with h5py.File('./results/importance_sampling_multidet.h5', 'r') as f:
            is_samples = {}
            for key in param_names:
                is_samples[key] = f[f'event_{event_id}/resampled/{key}'][:]
            has_is = True
    except:
        has_is = False
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        ax.hist(npe_samples[param], bins=50, alpha=0.5, color='blue',
               density=True, label='NPE', edgecolor='black', linewidth=0.5)
        
        if has_is and param in is_samples:
            ax.hist(is_samples[param], bins=50, alpha=0.5, color='green',
                   density=True, label='IS', edgecolor='black', linewidth=0.5)
        
        if param in true_params:
            true_val = true_params[param]
            ax.axvline(true_val, color='red', linestyle='--',
                      linewidth=2.5, label='True', alpha=0.9)
            
            npe_lower = np.percentile(npe_samples[param], 5)
            npe_upper = np.percentile(npe_samples[param], 95)
            in_ci = npe_lower <= true_val <= npe_upper
            ci_color = 'green' if in_ci else 'red'
            ax.axvspan(npe_lower, npe_upper, alpha=0.1, color=ci_color)
        
        ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Event {event_id}: NPE vs Importance Sampling Posteriors',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'posterior_comparison_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_corner_plot(event_id=0, use_is=False, save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    suffix = 'IS' if use_is else 'NPE'
    
    param_names = ['mass_1', 'mass_2', 'spin_1z', 'spin_2z', 'luminosity_distance']
    
    try:
        if use_is:
            with h5py.File('./results/importance_sampling_multidet.h5', 'r') as f:
                samples_dict = {}
                for key in param_names:
                    samples_dict[key] = f[f'event_{event_id}/resampled/{key}'][:]
        else:
            with h5py.File('./results/inference_results_multidet.h5', 'r') as f:
                samples_dict = {}
                for key in param_names:
                    samples_dict[key] = f['samples'][key][event_id]
    except:
        return
    
    try:
        with h5py.File('./results/inference_results_multidet.h5', 'r') as f:
            true_params = {}
            if 'true_parameters' in f:
                for key in param_names:
                    true_params[key] = f['true_parameters'][key][event_id]
    except:
        true_params = {}
    
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_params, figsize=(15, 15))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                data = samples_dict[param_names[i]]
                ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue',
                       edgecolor='black', linewidth=0.5)
                
                if param_names[i] in true_params:
                    ax.axvline(true_params[param_names[i]], color='red',
                              linestyle='--', linewidth=2)
                
                ax.set_ylabel('Density', fontsize=9)
                ax.set_yticks([])
                
            elif i > j:
                x = samples_dict[param_names[j]]
                y = samples_dict[param_names[i]]
                
                H, xedges, yedges = np.histogram2d(x, y, bins=40)
                H = gaussian_filter(H, sigma=0.8)
                
                ax.contourf(xedges[:-1], yedges[:-1], H.T, levels=10,
                           cmap='Blues', alpha=0.8)
                
                if param_names[i] in true_params and param_names[j] in true_params:
                    ax.plot(true_params[param_names[j]], true_params[param_names[i]],
                           'r*', markersize=15, markeredgewidth=2, markeredgecolor='darkred')
            else:
                ax.axis('off')
            
            if i == n_params - 1:
                ax.set_xlabel(param_names[j].replace('_', ' ').title(), fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i].replace('_', ' ').title(), fontsize=10)
            else:
                if i != j:
                    ax.set_yticklabels([])
            
            ax.tick_params(labelsize=7)
    
    plt.suptitle(f'Event {event_id} - Corner Plot ({suffix})',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'corner_{suffix.lower()}_event_{event_id}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_efficiency_summary(save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        with h5py.File('./results/importance_sampling_multidet.h5', 'r') as f:
            n_events = f.attrs['n_events']
            
            event_ids = []
            efficiencies = []
            ess_values = []
            improvements = []
            
            for i in range(n_events):
                event_ids.append(i)
                efficiencies.append(f[f'event_{i}'].attrs['efficiency'])
                ess_values.append(f[f'event_{i}'].attrs['ESS'])
                improvements.append(f[f'event_{i}'].attrs['improvements'])
    except:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes[0]
    bars = ax.bar(event_ids, efficiencies, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(y=np.mean(efficiencies), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(efficiencies):.2f}%')
    ax.set_xlabel('Event ID', fontsize=12)
    ax.set_ylabel('Efficiency (%)', fontsize=12)
    ax.set_title('Importance Sampling Efficiency', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    ax = axes[1]
    bars = ax.bar(event_ids, ess_values, color='green', alpha=0.8, edgecolor='black')
    ax.axhline(y=np.mean(ess_values), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(ess_values):.1f}')
    ax.set_xlabel('Event ID', fontsize=12)
    ax.set_ylabel('ESS', fontsize=12)
    ax.set_title('Effective Sample Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[2]
    bars = ax.bar(event_ids, improvements, color='orange', alpha=0.8, edgecolor='black')
    ax.axhline(y=np.mean(improvements), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(improvements):.1f}/5')
    ax.set_xlabel('Event ID', fontsize=12)
    ax.set_ylabel('Improved Parameters (out of 5)', fontsize=12)
    ax.set_title('Parameter Improvements', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 5.5)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'efficiency_summary.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_training_history(save_dir='./plots/'):
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        import torch
        checkpoint = torch.load('./models/light_model.pt', map_location='cpu')
        
        if 'history' in checkpoint:
            history = checkpoint['history']
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            
            ax = axes[0]
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, 
                   label='Train Loss', marker='o', markersize=4)
            ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, 
                   label='Val Loss', marker='s', markersize=4)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training History', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            ax = axes[1]
            ax.semilogy(epochs, history['train_loss'], 'b-', linewidth=2, 
                       label='Train Loss', marker='o', markersize=4)
            ax.semilogy(epochs, history['val_loss'], 'r-', linewidth=2, 
                       label='Val Loss', marker='s', markersize=4)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss (log scale)', fontsize=12)
            ax.set_title('Training History (Log Scale)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, 'training_history.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
    except:
        pass


def main():
    os.makedirs('./plots/', exist_ok=True)
    
    has_detector_data = os.path.exists('./data/detector_strain_full.h5')
    has_tokens = os.path.exists('./data/val_tokenized_multidet.h5')
    has_inference = os.path.exists('./results/inference_results_multidet.h5')
    has_is = os.path.exists('./results/importance_sampling_multidet.h5')
    
    if not has_inference:
        return
    
    with h5py.File('./results/inference_results_multidet.h5', 'r') as f:
        n_events = f.attrs['n_events']
    
    if has_detector_data:
        n_plot = min(3, n_events)
        for i in range(n_plot):
            plot_strain_signal_noise(i)
            plot_frequency_domain(i)
            plot_whitened_data(i)
            plot_qtransforms(i)
        
        plot_snr_analysis()
    
    if has_tokens:
        n_plot = min(3, n_events)
        for i in range(n_plot):
            plot_tokens(i)
    
    n_plot = min(3, n_events)
    for i in range(n_plot):
        plot_posterior_comparison(i)
        plot_corner_plot(i, use_is=False)
        if has_is:
            plot_corner_plot(i, use_is=True)
    
    if has_is:
        plot_efficiency_summary()
    
    plot_training_history()


if __name__ == '__main__':
    main()
