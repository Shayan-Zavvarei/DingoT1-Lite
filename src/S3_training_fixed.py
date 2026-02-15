import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
from tqdm import tqdm
import json
from datetime import datetime


class GWTokenDataset(Dataset):
    
    def __init__(self, token_file, n_extrinsic=3):
        self.token_file = token_file
        self.n_extrinsic = n_extrinsic
        
        with h5py.File(token_file, 'r') as f:
            self.tokens = f['tokens'][:]
            
            self.params = {}
            if 'parameters' in f:
                for key in f['parameters'].keys():
                    self.params[key] = f[f'parameters/{key}'][:]
            else:
                raise ValueError("No parameters found!")
            
            self.n_samples = f.attrs.get('n_samples', self.tokens.shape[0])
            self.n_tokens = f.attrs.get('max_tokens', self.tokens.shape[1])
            self.n_detectors = f.attrs.get('n_detectors', 3)
        
        self.param_stats = {}
        param_order = ['dec', 'geocent_time', 'inclination', 'luminosity_distance', 
                       'mass_1', 'mass_2', 'phase', 'psi', 'ra', 'spin_1z', 'spin_2z']
        
        for key in param_order:
            if key in self.params:
                self.param_stats[key] = {
                    'mean': np.mean(self.params[key]),
                    'std': np.std(self.params[key]) + 1e-6
                }
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        
        param_order = ['dec', 'geocent_time', 'inclination', 'luminosity_distance', 
                       'mass_1', 'mass_2', 'phase', 'psi', 'ra', 'spin_1z', 'spin_2z']
        
        params_list = []
        for key in param_order:
            if key in self.params:
                val = self.params[key][idx]
                
                normalized_val = (val - self.param_stats[key]['mean']) / self.param_stats[key]['std']
                params_list.append(normalized_val)
            else:
                params_list.append(0.0)
        
        tokens_tensor = torch.FloatTensor(tokens)
        params_tensor = torch.FloatTensor(params_list)
        
        return tokens_tensor, params_tensor


def apply_data_based_masking(tokens, mask_prob=0.2):
    B, L, _, _ = tokens.shape
    
    mask = torch.ones(B, L, dtype=torch.bool)
    
    for b in range(B):
        if np.random.rand() > mask_prob:
            continue
        
        mask_type = np.random.choice(['detector', 'frequency', 'notch'])
        
        if mask_type == 'detector':
            detector_to_mask = np.random.randint(0, 3)
            tokens_per_det = L // 3
            start_idx = detector_to_mask * tokens_per_det
            end_idx = start_idx + tokens_per_det
            mask[b, start_idx:end_idx] = False
        
        elif mask_type == 'frequency':
            if np.random.rand() < 0.7:
                cutoff = int(L * 0.7)
                mask[b, cutoff:] = False
            else:
                cutoff = int(L * 0.2)
                mask[b, :cutoff] = False
        
        elif mask_type == 'notch':
            n_mask = np.random.randint(5, 11)
            start = np.random.randint(0, L - n_mask)
            mask[b, start:start+n_mask] = False
    
    masked_tokens = tokens.clone()
    masked_tokens[~mask] = 0.0
    
    return masked_tokens, mask


class TokenEmbedding(nn.Module):
    
    def __init__(self, d_model=256):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(16 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
    
    def forward(self, tokens):
        B, L, _, _ = tokens.shape
        
        tokens_flat = tokens.view(B, L, -1)
        
        embeddings = self.embedding(tokens_flat)
        
        return embeddings


class TransformerEncoder(nn.Module):
    
    def __init__(self, d_model=256, n_heads=4, n_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.summary_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, embeddings, mask=None):
        B, L, _ = embeddings.shape
        
        summary = self.summary_token.expand(B, -1, -1)
        embeddings = torch.cat([summary, embeddings], dim=1)
        
        if mask is not None:
            summary_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([summary_mask, mask], dim=1)
            
            attn_mask = ~mask
        else:
            attn_mask = None
        
        output = self.transformer(embeddings, src_key_padding_mask=attn_mask)
        
        context = output[:, 0, :]
        
        return context


class NormalizingFlow(nn.Module):
    
    def __init__(self, n_params=11, context_dim=64, n_layers=4):
        super().__init__()
        
        self.n_params = n_params
        self.context_dim = context_dim
        
        self.n_split = n_params // 2
        self.n_rest = n_params - self.n_split
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                in_dim = self.n_split
                out_dim = self.n_rest
            else:
                in_dim = self.n_rest
                out_dim = self.n_split
            
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim + context_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, out_dim * 2)
                )
            )
    
    def forward(self, z, context):
        theta = z
        log_det = torch.zeros(z.shape[0], device=z.device)
        
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                theta1 = theta[:, :self.n_split]
                theta2 = theta[:, self.n_split:]
            else:
                theta1 = theta[:, :self.n_rest]
                theta2 = theta[:, self.n_rest:]
            
            h = layer(torch.cat([theta1, context], dim=1))
            s, t = torch.chunk(h, 2, dim=1)
            
            s = torch.tanh(s)
            theta2 = theta2 * torch.exp(s) + t
            
            log_det += s.sum(dim=1)
            
            if i % 2 == 0:
                theta = torch.cat([theta2, theta1], dim=1)
            else:
                theta = torch.cat([theta1, theta2], dim=1)
        
        return theta, log_det
    
    def log_prob(self, theta, context):
        z = theta
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i
            
            if layer_idx % 2 == 0:
                z2 = z[:, :self.n_rest]
                z1 = z[:, self.n_rest:]
            else:
                z2 = z[:, :self.n_split]
                z1 = z[:, self.n_split:]
            
            h = layer(torch.cat([z1, context], dim=1))
            s, t = torch.chunk(h, 2, dim=1)
            s = torch.tanh(s)
            
            z2 = (z2 - t) * torch.exp(-s)
            log_det -= s.sum(dim=1)
            
            if layer_idx % 2 == 0:
                z = torch.cat([z1, z2], dim=1)
            else:
                z = torch.cat([z2, z1], dim=1)
        
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * theta.shape[1] * np.log(2 * np.pi)
        
        log_q = log_pz + log_det
        
        return log_q


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
    
    def forward(self, tokens, params, mask=None):
        embeddings = self.token_embedding(tokens)
        
        context = self.transformer(embeddings, mask)
        
        context = self.context_projection(context)
        
        log_q = self.flow.log_prob(params, context)
        
        return log_q
    
    def sample(self, tokens, mask=None, n_samples=1):
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


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['patience']
        )
        
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for tokens, params in pbar:
            tokens = tokens.to(self.device)
            params = params.to(self.device)
            
            tokens_masked, mask = apply_data_based_masking(
                tokens,
                mask_prob=self.config.get('mask_prob', 0.2)
            )
            mask = mask.to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    log_q = self.model(tokens_masked, params, mask)
                    loss = -log_q.mean()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                log_q = self.model(tokens_masked, params, mask)
                loss = -log_q.mean()
                
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for tokens, params in tqdm(self.val_loader, desc="Validation"):
                tokens = tokens.to(self.device)
                params = params.to(self.device)
                
                log_q = self.model(tokens, params, mask=None)
                loss = -log_q.mean()
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint()
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['patience']:
                break
    
    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.config['model_save_path']), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, self.config['model_save_path'])


def print_config():
    from config_fixed import MODEL_CONFIG, TRAINING_CONFIG, PARAMETER_NAMES


def main():
    from config_fixed import MODEL_CONFIG, TRAINING_CONFIG, PARAMETER_NAMES
    
    print_config()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_dataset = GWTokenDataset(
        './data/train_tokenized_multidet.h5',
        n_extrinsic=TRAINING_CONFIG.get('n_extrinsic_per_intrinsic', 3)
    )
    
    val_dataset = GWTokenDataset(
        './data/val_tokenized_multidet.h5',
        n_extrinsic=TRAINING_CONFIG.get('n_extrinsic_per_intrinsic', 3)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    model = DingoT1Model(MODEL_CONFIG)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TRAINING_CONFIG,
        device=device
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
