"""
Debug version - shows exact errors
"""

import numpy as np
import h5py
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.noise import noise_from_psd
from pycbc.psd import aLIGOZeroDetHighPower, AdvVirgo

# Test single sample
print("Testing single waveform generation...")

sample_rate = 2048
duration = 4.0
f_lower = 20.0
target_len = int(duration * sample_rate)

# Simple parameters
mass_1 = 30.0
mass_2 = 25.0
spin_1z = 0.0
spin_2z = 0.0
distance = 500.0
inclination = 0.0
phase = 0.0
ra = 0.0
dec = 0.0
psi = 0.0
geocent_time = 1126259462.0

print(f"\nGenerating waveform...")
print(f"  M1={mass_1}, M2={mass_2}, DL={distance}")

try:
    # Generate waveform
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
    
    print(f"✓ Waveform generated: len={len(hp)}")
    
    # Resize
    hp.resize(target_len)
    hc.resize(target_len)
    print(f"✓ Resized to {target_len}")
    
    # Set time
    hp.start_time = geocent_time - duration + 2.0
    hc.start_time = geocent_time - duration + 2.0
    print(f"✓ Time set: {hp.start_time}")
    
    # Project to detector
    print(f"\nProjecting to H1...")
    det_H1 = Detector('H1')
    signal_H1 = det_H1.project_wave(hp, hc, ra, dec, psi)
    print(f"✓ Projected: len={len(signal_H1)}")
    
    signal_H1.resize(target_len)
    print(f"✓ Resized signal: {target_len}")
    
    # Generate noise
    print(f"\nGenerating noise...")
    delta_f = 1.0 / duration
    flen = int(sample_rate / (2 * delta_f)) + 1
    psd_H1 = aLIGOZeroDetHighPower(flen, delta_f, f_lower)
    
    noise_H1 = noise_from_psd(target_len, 1.0/sample_rate, psd_H1, seed=0)
    noise_H1.resize(target_len)
    print(f"✓ Noise generated: len={len(noise_H1)}")
    
    # Combine
    print(f"\nCombining signal + noise...")
    strain_H1 = signal_H1.numpy() + noise_H1.numpy()
    print(f"✓ Strain: shape={strain_H1.shape}")
    
    print("\n✅ SUCCESS! All steps worked!")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()
