"""
plot_batch_results.py
Plot corner plots for batch inference results
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load results
with h5py.File('./results/inference_results_multidet.h5', 'r') as f:
    samples = {}
    for key in f['samples'].keys():
        samples[key] = f['samples'][key][:]  # [10, 5000]
    
    true_params = {}
    if 'true_parameters' in f:
        for key in f['true_parameters'].keys():
            true_params[key] = f['true_parameters'][key][:]

print(f"Loaded {samples['mass_1'].shape[0]} events")

# Plot first event
event_idx = 0

plot_params = ['mass_1', 'mass_2', 'luminosity_distance', 'spin_1z', 'spin_2z']
n_params = len(plot_params)

fig, axes = plt.subplots(n_params, n_params, figsize=(15, 15))

for i in range(n_params):
    for j in range(n_params):
        ax = axes[i, j]
        
        if i == j:
            # 1D histogram
            data = samples[plot_params[i]][event_idx]
            ax.hist(data, bins=50, density=True, alpha=0.7, color='blue')
            
            # True value
            if plot_params[i] in true_params:
                true_val = true_params[plot_params[i]][event_idx]
                ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='True')
            
            ax.set_ylabel('Density', fontsize=10)
            
        elif i > j:
            # 2D histogram
            x = samples[plot_params[j]][event_idx]
            y = samples[plot_params[i]][event_idx]
            ax.hist2d(x, y, bins=40, cmap='Blues')
            
            # True value point
            if plot_params[i] in true_params and plot_params[j] in true_params:
                true_x = true_params[plot_params[j]][event_idx]
                true_y = true_params[plot_params[i]][event_idx]
                ax.plot(true_x, true_y, 'r*', markersize=15, markeredgewidth=2)
        else:
            ax.axis('off')
        
        if i == n_params - 1:
            ax.set_xlabel(plot_params[j], fontsize=10)
        if j == 0 and i > 0:
            ax.set_ylabel(plot_params[i], fontsize=10)
        
        ax.tick_params(labelsize=8)

plt.suptitle(f'Event {event_idx} - Posterior from Trained Model', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(f'corner_plot_event_{event_idx}_trained.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: corner_plot_event_{event_idx}_trained.png")
plt.close()

print("\n✓ Corner plot created!")
