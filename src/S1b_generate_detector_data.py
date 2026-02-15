import numpy as np
import h5py
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower, AdvVirgo
from pycbc.types import TimeSeries
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def antenna_response(detector, ra, dec, psi, gps_time):
    return detector.antenna_pattern(ra, dec, psi, gps_time)


def project_waveform_manual(hp, hc, detector, ra, dec, psi, gps_time):
    fp, fc = antenna_response(detector, ra, dec, psi, gps_time)
    
    h_det = fp * np.array(hp) + fc * np.array(hc)
    
    return h_det


def generate_detector_data(n_samples=100, output_file='./data/detector_strain_full.h5'):
    sample_rate = 2048
    duration = 4.0
    f_lower = 20.0
    target_len = int(duration * sample_rate)
    
    det_H1 = Detector('H1')
    det_L1 = Detector('L1')
    det_V1 = Detector('V1')
    
    delta_f = 1.0 / duration
    flen = int(sample_rate / (2 * delta_f)) + 1
    
    psd_H1 = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
    psd_L1 = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
    psd_V1 = AdvVirgo(flen, delta_f, f_lower)
    
    all_hp = []
    all_hc = []
    all_signal_H1 = []
    all_signal_L1 = []
    all_signal_V1 = []
    all_noise_H1 = []
    all_noise_L1 = []
    all_noise_V1 = []
    all_strain_H1 = []
    all_strain_L1 = []
    all_strain_V1 = []
    
    all_params = {
        'mass_1': [], 'mass_2': [],
        'spin_1z': [], 'spin_2z': [],
        'luminosity_distance': [],
        'inclination': [], 'ra': [], 'dec': [],
        'phase': [], 'psi': [], 'geocent_time': []
    }
    
    success_count = 0
    geocent_time = 1126259462.0
    
    for i in tqdm(range(n_samples)):
        try:
            mass_1 = np.random.uniform(15, 60)
            mass_2 = np.random.uniform(15, mass_1)
            spin_1z = np.random.uniform(-0.8, 0.8)
            spin_2z = np.random.uniform(-0.8, 0.8)
            distance = np.random.uniform(200, 3000)
            inclination = np.arccos(np.random.uniform(-1, 1))
            ra = np.random.uniform(0, 2*np.pi)
            dec = np.arcsin(np.random.uniform(-1, 1))
            phase = np.random.uniform(0, 2*np.pi)
            psi = np.random.uniform(0, np.pi)
            
            hp, hc = get_td_waveform(
                approximant='IMRPhenomPv2',
                mass1=mass_1,
                mass2=mass_2,
                spin1z=spin_1z,
                spin2z=spin_2z,
                distance=distance,
                inclination=inclination,
                coa_phase=phase,
                delta_t=1.0/sample_rate,
                f_lower=f_lower
            )
            
            hp_array = np.array(hp)
            hc_array = np.array(hc)
            
            if len(hp_array) < target_len:
                pad_len = target_len - len(hp_array)
                hp_array = np.pad(hp_array, (pad_len, 0), mode='constant')
                hc_array = np.pad(hc_array, (pad_len, 0), mode='constant')
            else:
                hp_array = hp_array[-target_len:]
                hc_array = hc_array[-target_len:]
            
            signal_H1 = project_waveform_manual(hp_array, hc_array, det_H1, ra, dec, psi, geocent_time)
            signal_L1 = project_waveform_manual(hp_array, hc_array, det_L1, ra, dec, psi, geocent_time)
            signal_V1 = project_waveform_manual(hp_array, hc_array, det_V1, ra, dec, psi, geocent_time)
            
            noise_H1 = noise_from_psd(target_len, 1.0/sample_rate, psd_H1, seed=i*3)
            noise_L1 = noise_from_psd(target_len, 1.0/sample_rate, psd_L1, seed=i*3+1)
            noise_V1 = noise_from_psd(target_len, 1.0/sample_rate, psd_V1, seed=i*3+2)
            
            noise_H1_array = np.array(noise_H1)
            noise_L1_array = np.array(noise_L1)
            noise_V1_array = np.array(noise_V1)
            
            strain_H1 = signal_H1 + noise_H1_array
            strain_L1 = signal_L1 + noise_L1_array
            strain_V1 = signal_V1 + noise_V1_array
            
            all_hp.append(hp_array)
            all_hc.append(hc_array)
            all_signal_H1.append(signal_H1)
            all_signal_L1.append(signal_L1)
            all_signal_V1.append(signal_V1)
            all_noise_H1.append(noise_H1_array)
            all_noise_L1.append(noise_L1_array)
            all_noise_V1.append(noise_V1_array)
            all_strain_H1.append(strain_H1)
            all_strain_L1.append(strain_L1)
            all_strain_V1.append(strain_V1)
            
            all_params['mass_1'].append(mass_1)
            all_params['mass_2'].append(mass_2)
            all_params['spin_1z'].append(spin_1z)
            all_params['spin_2z'].append(spin_2z)
            all_params['luminosity_distance'].append(distance)
            all_params['inclination'].append(inclination)
            all_params['ra'].append(ra)
            all_params['dec'].append(dec)
            all_params['phase'].append(phase)
            all_params['psi'].append(psi)
            all_params['geocent_time'].append(geocent_time)
            
            success_count += 1
            
        except Exception as e:
            continue
    
    if success_count == 0:
        return
    
    with h5py.File(output_file, 'w') as f:
        wf_grp = f.create_group('waveforms')
        wf_grp.create_dataset('hp', data=np.array(all_hp))
        wf_grp.create_dataset('hc', data=np.array(all_hc))
        
        sig_grp = f.create_group('signals')
        sig_grp.create_dataset('H1', data=np.array(all_signal_H1))
        sig_grp.create_dataset('L1', data=np.array(all_signal_L1))
        sig_grp.create_dataset('V1', data=np.array(all_signal_V1))
        
        noise_grp = f.create_group('noise')
        noise_grp.create_dataset('H1', data=np.array(all_noise_H1))
        noise_grp.create_dataset('L1', data=np.array(all_noise_L1))
        noise_grp.create_dataset('V1', data=np.array(all_noise_V1))
        
        strain_grp = f.create_group('strain')
        strain_grp.create_dataset('H1', data=np.array(all_strain_H1))
        strain_grp.create_dataset('L1', data=np.array(all_strain_L1))
        strain_grp.create_dataset('V1', data=np.array(all_strain_V1))
        
        params_grp = f.create_group('parameters')
        for key, val in all_params.items():
            params_grp.create_dataset(key, data=np.array(val))
        
        f.attrs['n_samples'] = success_count
        f.attrs['sample_rate'] = sample_rate
        f.attrs['duration'] = duration
        f.attrs['f_lower'] = f_lower


if __name__ == '__main__':
    import argparse, os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output', type=str, default='./data/detector_strain_full.h5')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    generate_detector_data(n_samples=args.n_samples, output_file=args.output)
