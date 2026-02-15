INTRINSIC_PARAMS = ['mass_1', 'mass_2', 'spin_1z', 'spin_2z']
EXTRINSIC_PARAMS = [
    'luminosity_distance',
    'inclination',
    'ra',
    'dec',
    'phase',
    'psi',
    'geocent_time'
]
PARAMETER_NAMES = INTRINSIC_PARAMS + EXTRINSIC_PARAMS


PRIOR_RANGES = {
    'mass_1': (5.0, 50.0),
    'mass_2': (5.0, 50.0),
    'spin_1z': (-0.99, 0.99),
    'spin_2z': (-0.99, 0.99),
    'luminosity_distance': (100.0, 3000.0),
    'inclination': (0.0, 3.14159),
    'ra': (0.0, 6.28318),
    'dec': (-1.5708, 1.5708),
    'phase': (0.0, 6.28318),
    'psi': (0.0, 3.14159),
    'geocent_time': (-0.1, 0.1)
}


DATA_CONFIG = {
    'f_min': 30.0,
    'f_max': 1024.0,
    'delta_f': 0.125,
    'duration': 8.0,
    'sample_rate': 2048,
    
    'detectors': ['H1', 'L1'],
    
    'n_train': 3000,
    'n_val': 500,
    'n_test': 100,
    
    'output_dir': './data/',
}


MODEL_CONFIG = {
    'n_bins': 16,
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 4,
    'd_ff': 512,
    'context_dim': 64,
    'n_params': 11,
    'n_flow_layers': 4,
    'dropout': 0.1,
}


TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'patience': 10,
    
    'mask_prob': 0.1,
    'n_extrinsic_per_intrinsic': 1,
    
    'model_save_path': './models/light_model_v2.pt',
    'use_amp': True,
    'gradient_accumulation': 1,
}


def print_comparison():
    full_params = 160_000_000
    light_params = (
        MODEL_CONFIG['d_model'] * MODEL_CONFIG['n_layers'] * 
        MODEL_CONFIG['d_ff'] * 4
    )


if __name__ == '__main__':
    print_comparison()
