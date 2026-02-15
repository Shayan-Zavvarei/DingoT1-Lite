import torch
import torch.nn as nn
import math


class MultiDetTokenizer(nn.Module):
    
    def __init__(self, n_bins=16, d_model=1024):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model
        
        in_features = n_bins * 3
        
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        
        self.freq_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 512)
        )
        
        self.detector_embed = nn.Embedding(3, 64)
        
        self.glu = nn.Linear(512 + 64, d_model * 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B, L, n_bins, n_feat = x.shape
        
        strain_psd = x[..., :3]
        f_bounds = x[:, :, 0, 3:5]
        detector_ids = x[:, :, 0, 5].long()
        
        x_flat = strain_psd.reshape(B, L, -1)
        
        h = self.net(x_flat)
        
        freq_cond = self.freq_embed(f_bounds)
        det_cond = self.detector_embed(detector_ids)
        
        h = h + freq_cond
        h = torch.cat([h, det_cond], dim=-1)
        
        out = self.glu(h)
        out, gate = out.chunk(2, dim=-1)
        out = out * torch.sigmoid(gate)
        
        return out


class SummaryToken(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(self, x):
        B = x.size(0)
        summary = self.token.expand(B, -1, -1)
        return torch.cat([summary, x], dim=1)


class PreLNTransformerLayer(nn.Module):
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        h = self.norm1(x)
        
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
            attn_mask = attn_mask.float() * -1e9
        
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + ffn_out
        
        return x


class RobustNormalizingFlow(nn.Module):
    
    def __init__(self, n_params, context_dim, n_layers=8, hidden_dim=256):
        super().__init__()
        self.n_params = n_params
        self.n1 = n_params // 2
        self.n2 = n_params - self.n1
        
        self.transforms = nn.ModuleList()
        
        for i in range(n_layers):
            if i % 2 == 0:
                in_dim = self.n1 + context_dim
                out_dim = self.n2 * 2
            else:
                in_dim = self.n2 + context_dim
                out_dim = self.n1 * 2
            
            self.transforms.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim)
            ))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.transforms:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)
    
    def forward(self, z, context):
        x = z
        log_det = torch.zeros(z.size(0), device=z.device)
        
        for i, transform in enumerate(self.transforms):
            x1, x2 = x[:, :self.n1], x[:, self.n1:]
            
            if i % 2 == 0:
                params = transform(torch.cat([x1, context], dim=-1))
            else:
                params = transform(torch.cat([x2, context], dim=-1))
            
            shift, log_scale = params.chunk(2, dim=-1)
            log_scale = torch.clamp(log_scale, -7, 3)
            scale = torch.exp(log_scale)
            
            if i % 2 == 0:
                x2 = x2 * scale + shift
                log_det = log_det + log_scale.sum(-1)
            else:
                x1 = x1 * scale + shift
                log_det = log_det + log_scale.sum(-1)
            
            x = torch.cat([x1, x2], dim=-1)
        
        return x, log_det
    
    def inverse(self, x, context):
        z = x
        log_det = torch.zeros(x.size(0), device=x.device)
        
        for i, transform in enumerate(reversed(self.transforms)):
            layer_idx = len(self.transforms) - 1 - i
            
            z1, z2 = z[:, :self.n1], z[:, self.n1:]
            
            if layer_idx % 2 == 0:
                params = transform(torch.cat([z1, context], dim=-1))
            else:
                params = transform(torch.cat([z2, context], dim=-1))
            
            shift, log_scale = params.chunk(2, dim=-1)
            log_scale = torch.clamp(log_scale, -7, 3)
            scale = torch.exp(log_scale)
            
            if layer_idx % 2 == 0:
                z2 = (z2 - shift) / (scale + 1e-8)
                log_det = log_det - log_scale.sum(-1)
            else:
                z1 = (z1 - shift) / (scale + 1e-8)
                log_det = log_det - log_scale.sum(-1)
            
            z = torch.cat([z1, z2], dim=-1)
        
        return z, log_det


class DingoT1MultiDet(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        d_model = config['d_model']
        n_heads = config['n_heads']
        n_layers = config['n_layers']
        d_ff = config['d_ff']
        context_dim = config['context_dim']
        n_params = config['n_params']
        dropout = config.get('dropout', 0.1)
        
        self.tokenizer = MultiDetTokenizer(16, d_model)
        self.summary_token = SummaryToken(d_model)
        
        self.encoder_layers = nn.ModuleList([
            PreLNTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, context_dim),
            nn.Tanh()
        )
        
        self.flow = RobustNormalizingFlow(
            n_params, context_dim,
            n_layers=config.get('n_flow_layers', 8),
            hidden_dim=256
        )
        
        total_params = sum(p.numel() for p in self.parameters())
    
    def encode(self, tokens, mask=None):
        x = self.tokenizer(tokens)
        
        x = self.summary_token(x)
        
        if mask is not None:
            summary_mask = torch.zeros(mask.size(0), 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([summary_mask, mask], dim=1)
        
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        
        summary = x[:, 0, :]
        context = self.context_proj(summary)
        
        return context
    
    def log_prob(self, params, tokens, mask=None):
        context = self.encode(tokens, mask)
        
        z, log_det = self.flow.inverse(params, context)
        
        log_prior = -0.5 * (z**2).sum(-1) - 0.5 * self.config['n_params'] * math.log(2 * math.pi)
        
        log_prob = log_prior + log_det
        
        return log_prob
    
    def sample(self, tokens, n_samples=1, mask=None):
        B = tokens.size(0)
        
        context = self.encode(tokens, mask)
        
        z = torch.randn(B * n_samples, self.config['n_params'], device=tokens.device)
        
        context_expanded = context.repeat_interleave(n_samples, dim=0)
        
        params, _ = self.flow(z, context_expanded)
        
        params = torch.clamp(params, 0.0, 1.0)
        
        params = params.reshape(B, n_samples, -1)
        
        return params


if __name__ == '__main__':
    from config_fixed import MODEL_CONFIG, print_config
    
    print_config()
    
    model = DingoT1MultiDet(MODEL_CONFIG)
    
    B, L = 2, 60
    tokens = torch.randn(B, L, 16, 6)
    params = torch.rand(B, 11)
    
    log_p = model.log_prob(params, tokens)
    
    samples = model.sample(tokens, n_samples=10) 
    