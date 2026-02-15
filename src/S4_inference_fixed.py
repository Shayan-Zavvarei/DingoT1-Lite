import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
import os


class TokenEmbedding(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        
        hidden = d_model * 2
        
        self.embedding = nn.Sequential(
            nn.Linear(16 * 6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model)
        )
    
    def forward(self, tokens):
        B, L, _, _ = tokens.shape
        tokens_flat = tokens.view(B, L, -1)
        return self.embedding(tokens_flat)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=2, n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, embeddings, mask=None):
        B = embeddings.shape[0]
        summary = self.summary_token.expand(B, -1, -1)
        embeddings = torch.cat([summary, embeddings], dim=1)
        
        if mask is not None:
            summary_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([summary_mask, mask], dim=1)
            attn_mask = ~mask
        else:
            attn_mask = None
        
        output = self.transformer(embeddings, src_key_padding_mask=attn_mask)
        return output[:, 0, :]


class NormalizingFlow(nn.Module):
    def __init__(self, n_params=11, context_dim=32, n_layers=2):
        super().__init__()
        self.n_params = n_params
        self.n_split = n_params // 2
        self.n_rest = n_params - self.n_split
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                in_dim, out_dim = self.n_split, self.n_rest
            else:
                in_dim, out_dim = self.n_rest, self.n_split
            
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim + context_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim * 2)
            ))
    
    def forward(self, z, context):
        theta = z
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                theta1, theta2 = theta[:, :self.n_split], theta[:, self.n_split:]
            else:
                theta1, theta2 = theta[:, :self.n_rest], theta[:, self.n_rest:]
            
            h = layer(torch.cat([theta1, context], dim=1))
            s, t = torch.chunk(h, 2, dim=1)
            s = torch.tanh(s)
            theta2 = theta2 * torch.exp(s) + t
            
            if i % 2 == 0:
                theta = torch.cat([theta2, theta1], dim=1)
            else:
                theta = torch.cat([theta1, theta2], dim=1)
        
        return theta, None


class DingoT1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(d_model=config['d_model'])
        self.transformer = TransformerEncoder(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout']
        )
        self.context_projection = nn.Linear(config['d_model'], config['context_dim'])
        self.flow = NormalizingFlow(
            n_params=config['n_params'],
            context_dim=config['context_dim'],
            n_layers=config['n_flow_layers']
        )
    
    def sample(self, tokens, mask=None, n_samples=1000):
        B = tokens.shape[0]
        
        with torch.no_grad():
            embeddings = self.token_embedding(tokens)
            context = self.transformer(embeddings, mask)
            context = self.context_projection(context)
            
            samples = []
            for _ in range(n_samples):
                z = torch.randn(B, self.config['n_params'], device=tokens.device)
                theta, _ = self.flow(z, context)
                samples.append(theta)
            
            samples = torch.stack(samples, dim=1)
        
        return samples


class InferenceEngine:
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        from config_fixed import MODEL_CONFIG
        self.config = MODEL_CONFIG
        
        self.model = DingoT1Model(self.config)
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            pass
        
        self.model.to(device)
        self.model.eval()
        
        self.param_names = ['dec', 'geocent_time', 'inclination', 'luminosity_distance', 
                            'mass_1', 'mass_2', 'phase', 'psi', 'ra', 'spin_1z', 'spin_2z']
        
        self.load_normalization_stats()
    
    def load_normalization_stats(self):
        try:
            with h5py.File('./data/train_tokenized_multidet.h5', 'r') as f:
                self.param_stats = {}
                for key in self.param_names:
                    if key in f['parameters']:
                        data = f['parameters'][key][:]
                        self.param_stats[key] = {
                            'mean': np.mean(data),
                            'std': np.std(data) + 1e-6
                        }
        except:
            self.param_stats = {k: {'mean': 0.0, 'std': 1.0} for k in self.param_names}
    
    def denormalize_samples(self, samples_normalized):
        samples_physical = np.zeros_like(samples_normalized)
        
        for i, key in enumerate(self.param_names):
            if key in self.param_stats:
                samples_physical[:, i] = (
                    samples_normalized[:, i] * self.param_stats[key]['std'] + 
                    self.param_stats[key]['mean']
                )
            else:
                samples_physical[:, i] = samples_normalized[:, i]
        
        return samples_physical
    
    def run_inference(self, tokens, n_samples=5000):
        tokens_tensor = torch.FloatTensor(tokens).to(self.device)
        
        samples_normalized = self.model.sample(tokens_tensor, n_samples=n_samples)
        samples_normalized = samples_normalized[0].cpu().numpy()
        
        samples_physical = self.denormalize_samples(samples_normalized)
        
        samples_dict = {}
        for i, key in enumerate(self.param_names):
            samples_dict[key] = samples_physical[:, i]
        
        return samples_dict
    
    def plot_corner(self, samples_dict, true_params=None, output_file='corner_plot.png'):
        plot_params = ['mass_1', 'mass_2', 'luminosity_distance', 
                       'spin_1z', 'spin_2z']
        
        n_params = len(plot_params)
        fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
        
        for i in range(n_params):
            for j in range(n_params):
                ax = axes[i, j]
                
                if i == j:
                    data = samples_dict[plot_params[i]]
                    ax.hist(data, bins=50, density=True, alpha=0.7, color='blue')
                    
                    if true_params and plot_params[i] in true_params:
                        ax.axvline(true_params[plot_params[i]], color='red', 
                                   linestyle='--', linewidth=2, label='True')
                    
                    ax.set_ylabel('Density', fontsize=8)
                elif i > j:
                    x = samples_dict[plot_params[j]]
                    y = samples_dict[plot_params[i]]
                    ax.hist2d(x, y, bins=40, cmap='Blues')
                    
                    if true_params and plot_params[i] in true_params and plot_params[j] in true_params:
                        ax.plot(true_params[plot_params[j]], true_params[plot_params[i]], 
                                'r*', markersize=10)
                else:
                    ax.axis('off')
                
                if i == n_params - 1:
                    ax.set_xlabel(plot_params[j], fontsize=8)
                if j == 0 and i > 0:
                    ax.set_ylabel(plot_params[i], fontsize=8)
                
                ax.tick_params(labelsize=6)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, samples_dict, tokens, output_file='./results/inference_results_multidet.h5'):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            grp = f.create_group('samples')
            for key, val in samples_dict.items():
                grp.create_dataset(key, data=val)
            
            f.create_dataset('tokens', data=tokens)
            
            f.attrs['n_samples'] = len(next(iter(samples_dict.values())))
            f.attrs['n_params'] = len(samples_dict)


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model', type=str, default='./models/light_model_v2.pt',
                        help='Path to trained model')
    parser.add_argument('--data', type=str, default='./data/val_tokenized_multidet.h5',
                        help='Path to tokenized data')
    parser.add_argument('--index', type=int, default=0,
                        help='Sample index to analyze')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of posterior samples')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    
    args = parser.parse_args()
    
    engine = InferenceEngine(args.model, device=args.device)
    
    with h5py.File(args.data, 'r') as f:
        tokens = f['tokens'][args.index:args.index+1]
        
        true_params = {}
        if 'parameters' in f:
            for key in engine.param_names:
                if key in f['parameters']:
                    true_params[key] = f['parameters'][key][args.index]
    
    samples_dict = engine.run_inference(tokens, n_samples=args.n_samples)
    
    engine.plot_corner(samples_dict, true_params=true_params, 
                       output_file='inference_corner_plot.png')


if __name__ == '__main__':
    main()
