import torch
import torch.nn as nn
import pickle
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, channels=16, freq_bins, time_steps)
        x = self.proj(x)  # (batch, emb_dim, H', W') where H', W' = freq_bins//ph, time_steps//pw
        x = x.flatten(2)  # (batch, emb_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, emb_dim)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTRegression(nn.Module):
    def __init__(self, in_channels=16, patch_size=(4, 4), emb_dim=128, num_heads=4, mlp_dim=256, num_layers=4, dropout=0.1, device="cuda", load=False, model_name=None):
        super(ViTRegression, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.load = load
        self.model_name = model_name
        if load: # load pos_embedding
            with open(f'./models/ViT position emb/{model_name}.pkl', 'rb') as file:
                self.pos_embedding = pickle.load(file)
        else:
            self.pos_embedding = None  # Will be created dynamically in forward()

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, 1)
        self.device = device
    
    # def to_device(self, device):
    #     self.patch_embed.to(device)

    def forward(self, x):
        x = self.patch_embed(x)  # (batch_size, num_patches, emb_dim)
        B, N, D = x.shape

        # print(f'x.shape:{x.shape}')

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + num_patches, emb_dim)

        # print(f'x.shape:{x.shape}')
        # print(f'x.shape[1]:{x.shape[1]}')
        # exit()

        # Positional encoding
        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1] and not self.load:
            self.pos_embedding = nn.Parameter(torch.zeros(1, x.shape[1], D)).to(self.device)
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
            with open(f'./models/ViT position emb/{self.model_name}.pkl', 'wb') as file: # save pos_embedding
                pickle.dump(self.pos_embedding, file)

        # print(f'x.type:{x.type}')
        # print(f'self.pos_embedding.type:{self.pos_embedding.type}')
        # print(f'x.shape:{x.shape}')
        # print(f'self.pos_embedding.shape:{self.pos_embedding.shape}')
        x = x + self.pos_embedding

        # exit()

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        cls_output = x[:, 0]  # Use CLS token representation
        out = self.head(cls_output)  # (batch_size, 1)
        return out