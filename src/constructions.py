# Copyright 2026 Theodore Maselko
# Licensed under the MIT license. See LICENSE file in the project root for details.

import math
from functools import reduce
from itertools import product


import torch
from torch import Tensor

from .model import SimpleAttentionModel
from .tsar import TSARSynthDataset






def _set_projections(model:SimpleAttentionModel):
	model.attn_norm.weight.data.fill_(1)
	d_qk = model.args['d_qk']
	w_qkv = model.attn.qkv_proj.weight.data
	model.attn.qkv_proj.bias.data.fill_(0)
	wq = w_qkv[:d_qk, :].fill_(0)
	wk = w_qkv[d_qk:2*d_qk, :].fill_(0)
	wv = w_qkv[2*d_qk:, :].fill_(0)
	wq[:, d_qk:].fill_diagonal_(1)
	wk[:, :d_qk].fill_diagonal_(1)
	wv[:, d_qk:].fill_diagonal_(1)
	model.attn.out_proj.weight.data.fill_(0).fill_diagonal_(1)
	model.attn.out_proj.bias.data.fill_(0)
	model.outproj.weight.data.fill_(0)
	model.outproj.weight.data[:, :d_qk].copy_(model.embed.weight.data)
	model.outproj.bias.data.fill_(0)



def calc_magnitude(theta_min, N):
    # as seen in paper
	req_sep = math.sin(theta_min/2)
	return math.log(req_sep/(N-1))/(math.cos(theta_min) - 1)



def construct_model_randomsphere(model:SimpleAttentionModel, magnitude:float):
    with torch.no_grad():
        w_emb = model.embed.weight.data
        w_emb[2:].normal_()
        w_emb[2:].div_(w_emb[2:].norm(p=2, dim=-1, keepdim=True))
        w_emb[:2] = 0
        model.attn.tau = 1/magnitude
        model.args['tau'] = model.attn.tau

        _set_projections(model)



def construct_model_unitcircle(model:SimpleAttentionModel):
    N = model.embed.num_embeddings - 2
    d_qk = model.args['d_qk']
    assert d_qk == 2, '2d initialization only possible in... 2d.'
    with torch.no_grad():
        # Construct an optimal spherical code
        w_emb = model.embed.weight.data
        angles = 2 * math.pi * torch.arange(0, N, device=w_emb.device, dtype=w_emb.dtype) / N
        w_emb[2:] = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        w_emb[:2] = 0

        magnitude = calc_magnitude(2 * math.pi/N, N)
        model.attn.tau = 1/magnitude
        model.args['tau'] = model.attn.tau
        print(f'Score Magnitude: {magnitude:.02f}')

        _set_projections(model)



def hypergrid_N(d, L):
    eff_L = L + L%2
    max_val = eff_L - 1  # lattice levels are in the range [-max_val, max_val]. One is popped if L is odd.

    def _divisible(k):
        # how many (odd!) multiples of k are there in max_val?
        # really, the question is 'how many odd numbers are between 1 and max_val//k?'
        return (max_val//k + 1)//2 * 2

    def _mobius_ish(n):  # for n>1, (-1)^k if all k prime factors are unique, 0 otherwise.
        nfactors = 0
        prime = 3
        # This only works because the check will always fail when `prime` is a technically-not-prime composite number,
        # since the lower primes it is made from will have already been checked and removed from `n`. sneaky sneaky.
        while prime * prime <= n:
            if n % prime == 0:  # factor of n?
                nfactors += 1
                n //= prime
                if n % prime == 0:  # multiplicity!
                    return 0
            prime += 2  # only care about odds...
        # if we've reached `prime*prime > n` (see: prime > sqrt(n)) and n isn't 1, that means whatever's left is another factor and there definitely isn't two of them.
        if n > 1:
            nfactors += 1
        return -1 if nfactors%2 else 1
    
    total = L**d  # What we start from
    for k in range(3, max_val + 1, 2):  # only odd k
        mu = _mobius_ish(k)
        if mu != 0:
            total += mu * _divisible(k) ** d
    return total


def _create_code(d, L):
    # make the code using integer arithmetic instead of torch.cartesian_product to avoid float shenanigans when checking alignment 
    def _is_primitive(v):
        return reduce(math.gcd, map(abs, v)) == 1

    eff_L = L + L%2
    vals = list(range(-(eff_L-1), eff_L, 2))  # odd integers
    if L != eff_L:
        vals.pop(len(vals)//2)
    vecs = [v for v in product(vals, repeat=d) if _is_primitive(v)]
    return vecs

def calculate_maximum_alignment(w:Tensor):
    w_norm = w / w.norm(p=2, dim=-1, keepdim=True)
    cosine_sim = w_norm @ w_norm.T
    cosine_sim.fill_diagonal_(-1)
    max_cosine = torch.clamp(cosine_sim.max(), -1.0, 1.0)
    return max_cosine.item()

def construct_model_hypergrid(model:SimpleAttentionModel, L:int):
    N = TSARSynthDataset.count_symbols(model.embed.num_embeddings)
    control_tokens = model.embed.num_embeddings - N
    d_qk = model.args['d_qk']
    assert L >= 2, 'L must be >= 2.'
    with torch.no_grad():
        # Construct embedding from hypercube lattice
        w_emb = model.embed.weight.data
        w_code = torch.tensor(_create_code(d_qk, L), dtype=torch.float)
        w_code.div_(w_code.norm(p=2, dim=-1, keepdim=True))
        assert len(w_code) == N
        
        w_emb[control_tokens:] = w_code
        w_emb[:control_tokens] = 0

        theta_min = math.acos(calculate_maximum_alignment(w_code))
        magnitude = calc_magnitude(theta_min, N)
        model.attn.tau = 1/(magnitude)
        model.args['tau'] = model.attn.tau
        print(f'Score Magnitude: {magnitude:.02f}, theta_min: {theta_min:.04f}')
        
        _set_projections(model)

def hypergrid_align_mag(d, L):
    N = hypergrid_N(d, L)
    code = torch.tensor(_create_code(d, L), dtype=torch.float, device='cpu')
    code.div_(code.norm(p=2, dim=-1, keepdim=True))
    align = calculate_maximum_alignment(code)
    return align, calc_magnitude(math.acos(align), N)
