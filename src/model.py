# Copyright 2026 Theodore Maselko
# Licensed under the MIT license. See LICENSE file in the project root for details.

import os, json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Required if we want to keep track of dtype casts sanely
import torch._C._nn as spooky_nn  # type: ignore



DEFAULT_BACKENDS = [nn.attention.SDPBackend.CUDNN_ATTENTION, nn.attention.SDPBackend.FLASH_ATTENTION, nn.attention.SDPBackend.EFFICIENT_ATTENTION, nn.attention.SDPBackend.MATH]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, d_qk: int, d_v: int, tau:float, force_precision:bool, device, dtype):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.head_shape = (d_qk, d_v)
        self.qkv_proj = nn.Linear(d_model, 2*d_qk + d_v, **factory_kwargs)
        self.out_proj = nn.Linear(d_v, d_model, **factory_kwargs)
        self.tau = tau
        self.force_precision = force_precision
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        B, S, E = hidden.shape
        d, v = self.head_shape

        qkv = self.qkv_proj(hidden)
        query, key = qkv[..., :2*d].chunk(2, dim=-1)  # [B, S, d]
        value = qkv[..., 2*d:]  # [B, S, v]

        # [B, H, S, d/v]
        query = query.view(B, 1, S, d)
        key = key.view(B, 1, S, d)
        value = value.view(B, 1, S, v)

        # [B, H, S, v] (Song-and-dance to prevent F.sdpa from dying)
        query = F.pad(query, [0, (8 - (d%8))%8], 'constant', 0)
        key = F.pad(key, [0, (8 - (d%8))%8], 'constant', 0)
        value = F.pad(value, [0, (8 - (v%8))%8], 'constant', 0)
        with nn.attention.sdpa_kernel([nn.attention.SDPBackend.MATH] if self.force_precision else DEFAULT_BACKENDS):
            attn_output = F.scaled_dot_product_attention(query, key, value,
                                                            is_causal=False,
                                                            scale=1/self.tau)
        attn_output = attn_output[..., :v]
        attn_output = attn_output.view(B, S, v)

        attn_output = self.out_proj(attn_output)
        return attn_output


class SimpleAttentionModel(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, d_qk:int, d_v:int, tau:float, features:dict, dtype, device):
        super().__init__()
        self.name = f'{self.__class__.__name__}'
        self.args = dict(kv for kv in locals().items() if kv[0] not in ['self', '__class__', 'device'])
        kwargs = {'dtype': dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype.split('.', 1)[-1]), 'device': device}

        self.embed = nn.Embedding(vocab_size, d_model//2, **kwargs)
        self.outproj = nn.Linear(d_model, vocab_size, **kwargs)

        self.attn_norm = nn.RMSNorm(d_model, elementwise_affine=True, **kwargs)
        self.attn = SelfAttention(d_model=d_model, d_qk=d_qk, d_v=d_v, tau=tau,
                                  force_precision=features.get('force_precision', False), **kwargs)
        
        self.mlp = nn.Sequential(
            nn.RMSNorm(d_model, **kwargs),
            nn.Linear(d_model, d_model*4, **kwargs),
            nn.GELU(),
            nn.Linear(d_model*4, d_model, **kwargs),
        ) if features.get('with_mlp', False) else None

    def to(self, *args, **kwargs):
        _, dtype, _, _ = spooky_nn._parse_to(*args, **kwargs)
        if dtype is not None:
            self.args['dtype'] = dtype
        return super().to(*args, **kwargs)
    
    @classmethod
    def load(cls, model_folder, device, weights_file='model.pth'):
        try:
            with open(os.path.join(model_folder, 'args.json'), 'rt', -1, 'utf-8') as f:
                args = json.load(f)

            model = SimpleAttentionModel(**args, device=device)
            state_dict = torch.load(os.path.join(model_folder, weights_file), map_location=device)
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            print(f"Error loading state dict for '{model_folder}': {e}")
            raise e

    def save(self, model_folder, untrained=False):
        os.makedirs(model_folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_folder, 'untrained.pth' if untrained else 'model.pth'))
        with open(os.path.join(model_folder, 'args.json'), 'wt', -1, 'utf-8') as f:
            json.dump(self.args, f, indent='  ', default=str)
        maxlname = 1 + max(len(x) for x,_ in self.named_parameters())
        maxlshape = 1 + max(len(str(p.shape)) for _,p in self.named_parameters())
        shapetext = '\n'.join(f'{n.ljust(maxlname)} {str(p.shape).ljust(maxlshape)} {p.numel(): 12,}'
                                for n, p in self.named_parameters())
        with open(os.path.join(model_folder, 'shapes.txt'), 'wt', -1, 'utf-8') as f:
            f.write(f"Params (total): {sum(p.numel() for p in self.parameters()):,}\n"
                    f"Params (train): {sum(p.numel() for p in self.parameters() if p.requires_grad):,}\n"
                    f"{shapetext}\n")

    def forward(self, in_toks:Tensor):
        """in_toks: [B, S, 2] -> [B, S, N]"""
        emb:Tensor = self.embed(in_toks).flatten(-2, -1)
        emb = emb + self.attn(self.attn_norm(emb))
        if self.mlp is not None:
            emb = emb + self.mlp(emb)
        logits = self.outproj(emb)
        return logits