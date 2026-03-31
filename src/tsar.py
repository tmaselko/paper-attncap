# Copyright 2026 Theodore Maselko
# Licensed under the MIT license. See LICENSE file in the project root for details.

import torch
import torch.nn.functional as F
import math


class TSARSynthDataset():
    t_bos = 0
    t_query = 1
    zero_symbol = 2
    
    def __init__(self, num_symbols:int, keys_per_seq:int, num_batches:int, batch_size:int,
                 full_shuffle:bool, device, seed=None):
        assert keys_per_seq <= num_symbols
        self.num_symbols = num_symbols
        self.keys_per_seq = keys_per_seq
        self.full_shuffle = full_shuffle

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.device = device
        
        # Should be on the GPU to go fast, but that does make determinism iffy.
        self.trand = torch.Generator(device=device)
        if seed is not None:
            self.trand.manual_seed(seed)
        
    @classmethod
    def vocab_size(cls, num_symbols):
        return 2 + num_symbols
    
    @classmethod
    def count_symbols(cls, vocab_size):
        return vocab_size - 2
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        self.cursor = self.num_batches
        return self
    
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cursor:
            raise StopIteration()
        self.cursor -= 1
        return self.get_batch()

    def get_batch(self):
        with torch.no_grad():
            key_count = self.num_symbols
            val_offset = 0
            B, N, n, S = self.batch_size, key_count, self.keys_per_seq, self.keys_per_seq*2
            gen_kwargs = {'generator': self.trand, 'device': self.device}

            seqs_per_symrange = int(math.floor(N / n))
            sym_repeats = int(math.ceil(B / seqs_per_symrange))

            required_from_perm = min(seqs_per_symrange, B)*n
            keys = torch.concat([
                torch.randperm(key_count, **gen_kwargs)[:required_from_perm]
                for i in range(sym_repeats)
            ], dim=-1)[:B*n].view(B, n)  # [B, unique_keys]
            keys.add_(self.zero_symbol)
            values = torch.randint(self.zero_symbol+val_offset, self.zero_symbol+self.num_symbols,
                                  (B, n), **gen_kwargs)


            key_indices = torch.arange(n, device=self.device).repeat([2]).unsqueeze(0).expand(B, -1)
            if key_indices.shape[-1] < S:
                remaining = S - key_indices.shape[-1]
                extras = torch.randint(0, n, (B, remaining), **gen_kwargs)
                key_indices = torch.concat([key_indices, extras], dim=-1)
                
            if self.full_shuffle:
                perm = torch.argsort(torch.rand_like(key_indices.float()), dim=-1)
                key_indices = torch.gather(key_indices, -1, perm)
            else:
                # Assignments are shuffled by default, but retrieval section still should be shuffled.
                perm = torch.argsort(torch.rand_like(key_indices[..., n:].float()), dim=-1)
                key_indices = key_indices.clone()
                key_indices[..., n:] = torch.gather(key_indices[...,n:], -1, perm)

            keys = torch.gather(keys, -1, key_indices)
            values = torch.gather(values, -1, key_indices)

            positions = torch.arange(S, device=self.device).view(1, -1).expand(B, -1)
            first_occurrence = torch.full((B, n), S, dtype=torch.long, device=self.device)
            first_occurrence.scatter_reduce_(
                dim=1,
                index=key_indices,
                src=positions,
                reduce="amin",
                include_self=False,
            )
            first_occ_per_pos = torch.gather(first_occurrence, dim=1, index=key_indices)
            first_mask = positions==first_occ_per_pos
            
            inputs = torch.stack([
                torch.where(first_mask, keys, self.t_query),
                torch.where(first_mask, values, keys),
            ], dim=-1)
            labels = torch.where(first_mask, -100, values)

            inputs = F.pad(inputs, (0, 0, 1, 0), 'constant', self.t_bos)
            labels = F.pad(labels, (1, 0), 'constant', -100)

            # print(f'inputs: {''.join([str(x).rjust(6) for x in inputs[0].tolist()])[-200:]}\n'
            #       f'labels: {''.join([str(x).rjust(6) for x in labels[0].tolist()])[-200:]}\n')
            return inputs, labels
    
