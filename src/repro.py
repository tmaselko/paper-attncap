# Copyright 2026 Theodore Maselko
# Licensed under the MIT license. See LICENSE file in the project root for details.

import math, time, os, json, copy, gc
from pathlib import Path


import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from . import chartmaker
from . import constructions
from .model import SimpleAttentionModel
from .tsar import TSARSynthDataset

REPORT_INTERVAL = 10

PathLike = Path|str

def _load_json(path:PathLike):  # yep
    with open(path, 'rt', -1, 'utf-8') as f:
        return json.load(f)
def _save_json(path:PathLike, blob):  # yyyyyyep
    with open(path, 'wt', -1, 'utf-8') as f:
        json.dump(blob, f, indent='\t')
def _save_text(path:PathLike, literature):  # they dont think it be like it is but it do
    with open(path, 'wt', -1, 'utf-8') as f:
        f.write(literature)


def _is_model_finished(folder:PathLike):
    folder = Path(folder)
    if (folder / 'accuracy.json').exists():
        return True
    return False

def _load_model_if_finished(folder:PathLike, device):
    folder = Path(folder)
    if (folder / 'accuracy.json').exists():
        return SimpleAttentionModel.load(str(folder), device)
    return None

def _clean_model_folder(folder:PathLike):
    """cleans all json files from model folder, like training logs and accuracy reports"""
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith('.json'):
                os.remove(os.path.join(folder, file))









def _train_model(model:SimpleAttentionModel, dataset:TSARSynthDataset, lr:float, weight_decay:float, freeze_token_range:int, saveto:Path):
    progress_file = os.path.join(saveto, 'training.json')
    timeline = _load_json(progress_file) if os.path.exists(progress_file) else []
    progress_offset = timeline[-1]['progress'] if timeline else 0

    wall_time = last_report = time.perf_counter()
    total_samples = samples_last_report = 0

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=lr, fused=True, weight_decay=weight_decay)

    model.train()
    if freeze_token_range:
        frozen_embeds = model.embed.weight[-freeze_token_range:].clone().detach()

    for cur_batch, (inputs, labels) in enumerate(dataset):
        total_samples += inputs.size(0) * inputs.size(1)

        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, -2), labels.reshape(-1))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            if freeze_token_range:
                model.embed.weight.data[-freeze_token_range:] = frozen_embeds

            progress = (cur_batch+1) / len(dataset)
            wrap_time = time.perf_counter() - last_report
            if cur_batch&1 == 0 or cur_batch == len(dataset)-1:
                timeline.append({
                    'progress': progress_offset + progress,
                    'batch_size': dataset.batch_size,
                    'seq_size': dataset.keys_per_seq,
                    'lr': next(group['lr'] for group in optimizer.param_groups),
                    'wall': time.perf_counter() - wall_time,
                    'samples_per_sec': (total_samples - samples_last_report) / wrap_time,
                    'loss': loss.detach()
                })
                samples_last_report = total_samples

    for t in timeline:
        for k in t:
            if isinstance(t[k], Tensor):
                t[k] = t[k].item()

    model.save(saveto)
    with open(progress_file, 'wt', -1, 'utf-8') as f:
        json.dump(timeline, f, indent='\t')

    return timeline

def run_training_curriculum(model:SimpleAttentionModel, model_folder:Path, training_steps:list[tuple],
                             weight_decay:float, freeze_symbols:bool):
    num_symbols = TSARSynthDataset.count_symbols(model.args['vocab_size'])
    model.save(model_folder, untrained=True)
    
    for idx, (unique_keys, lr, num_batches) in enumerate(training_steps):
        if isinstance(num_batches, tuple):
            num_batches, batch_size = num_batches
        else:
            batch_size = num_symbols//unique_keys
        dataset = TSARSynthDataset(num_symbols=num_symbols, keys_per_seq=unique_keys, full_shuffle=False, seed=None,
                                   num_batches=num_batches, batch_size=batch_size, device=model.embed.weight.data.device)
        timeline = _train_model(model=model, dataset=dataset, lr=lr, weight_decay=weight_decay,
                                freeze_token_range=num_symbols if freeze_symbols else 0, saveto=model_folder)
        
    chartmaker.generate_stacked_graphs(
        chart_path=os.path.join(model_folder, 'stats.jpg'),
        chart_title=f"Training Stats for Model({TSARSynthDataset.count_symbols(model.args['vocab_size'])}, {model.args['d_qk']})",
        x_vals=[x['progress'] for x in timeline],
        plot_axis_labels=['Keys', 'Loss', 'Learning Rate'],
        plot_logscale=[False, True, True],
        plot_extrema=['max', 'min', 'both'],
        series_names=['Keys per Sequence', 'Loss', 'Learning Rate'],
        series_data=[
            [x['seq_size'] for x in timeline],
            [x['loss'] for x in timeline],
            [x['lr'] for x in timeline],
        ],
        series_plot=[0, 1, 2],
    )

def _test_model(model:SimpleAttentionModel, dataset:TSARSynthDataset):
    def _per_position_accuracy(logits: Tensor, labels: Tensor):
        """logits: [B, S, L], labels: [B, S]"""
        correct = (logits.argmax(dim=-1) == labels)  # [B, S]
        valid = (labels != -100)  # [B, S]
        
        # Compute seen_keys per example: cumsum of invalid positions
        seen_keys = (torch.cumsum(~valid, dim=-1) - 2).clamp(min=0)  # [B, S]
        
        max_seen = int(seen_keys.max().item())
        zargs = {'dtype': labels.dtype, 'device': labels.device}
        
        # Flatten and scatter
        retrievals = torch.zeros((max_seen + 1,), **zargs)
        correct_counts = torch.zeros((max_seen + 1,), **zargs)
        
        retrievals.scatter_add_(0, seen_keys[valid], torch.ones_like(seen_keys[valid]))
        correct_counts.scatter_add_(0, seen_keys[valid], correct[valid].to(labels.dtype))
        
        return correct_counts, retrievals

    all_position_corrects = torch.zeros((dataset.keys_per_seq), dtype=torch.int, device=dataset.device)
    all_position_counts = torch.zeros((dataset.keys_per_seq), dtype=torch.int, device=dataset.device)
    
    model.eval()
    with torch.no_grad():
        wall_time = time.perf_counter()
        last_report = wall_time
        total_samples = samples_last_report = 0

        for cur_batch, (inputs, labels) in enumerate(dataset):
            total_samples += inputs.size(0) * inputs.size(1)
            logits = model(inputs)
        
            correct_per_pos, retrievals_per_pos = _per_position_accuracy(logits, labels)
            all_position_corrects += correct_per_pos
            all_position_counts += retrievals_per_pos

            progress = (cur_batch+1) / len(dataset)
        
            if time.perf_counter() - last_report > REPORT_INTERVAL:
                samples_per_sec = (total_samples - samples_last_report) / (time.perf_counter() - last_report)
                samples_last_report = total_samples
                print(f'  Progress [{progress:.1%}] {time.perf_counter() - wall_time:4.01f}s {samples_per_sec/1000:.02f}k/s')
                last_report = time.perf_counter()

        accuracies = all_position_corrects.float() / all_position_counts.clamp(min=1)
        nzmask = all_position_counts > 0
        positions = torch.arange(1, 1+all_position_counts.shape[-1], device=all_position_counts.device)
    return {
        'accuracies': accuracies[nzmask].tolist(),
        'num_correct': all_position_corrects.sum().item(),
        'counts': all_position_counts[nzmask].tolist(),
        'positions': positions[nzmask].tolist(),
    }

def run_test_battery(model:SimpleAttentionModel|None, model_folder:Path, report_filename:str, device, extra_data={}):
    filename = os.path.join(model_folder, report_filename)
    accuracy_report = _load_json(filename) if os.path.exists(filename) else {}

    if len(set(['native', 'fp32', 'fp16']) - set(accuracy_report.keys())) == 0:
        return accuracy_report  # No new tests to run.
    
    model = model or SimpleAttentionModel.load(model_folder, device)
    num_symbols = TSARSynthDataset.count_symbols(model.args['vocab_size'])
    dataset = TSARSynthDataset(num_symbols=num_symbols, keys_per_seq=num_symbols, full_shuffle=False,
                               num_batches=64, batch_size=max(1, min(16384//num_symbols, 16)), device=device)
    
    def _run_test(model):
        try:
            results = _test_model(model, dataset)
            return {
                'accuracy': results['accuracies'][0],
                'num_correct': results['num_correct'],
                'count': results['counts'][0],
                'n': results['positions'][0],  # added for convenience
                'd_qk': model.args['d_qk'],  # added for convenience
                **extra_data,
            }
        except RuntimeError  as e:
            if "CUDA" in str(e):  # OOM happens
                print('Test ran out of memory, skipping...')
                gc.collect()
                torch.cuda.empty_cache()
                return {
                    'accuracy': -1,
                    'num_correct': 0,
                    'count': 0,
                    'n': 0,
                    'd_qk': model.args['d_qk'],
                    **extra_data,
                }
            raise e

    
    if 'native' not in accuracy_report:
        accuracy_report['native'] = _run_test(model)
    if 'fp32' not in accuracy_report:
        accuracy_report['fp32'] = _run_test(copy.deepcopy(model).to(torch.float32))
    if 'fp16' not in accuracy_report:
        accuracy_report['fp16'] = _run_test(copy.deepcopy(model).to(torch.float16))

    _save_json(filename, accuracy_report)

    return accuracy_report


def _report_accuracy(print_name:str, accuracy_report:dict):
    best = accuracy_report['native']
    if best['accuracy'] == 1.0:
        print(f'  Tested {print_name}: NATIVE=100%, (FP16={accuracy_report['fp16']['accuracy']:.04%})')
    else:
        print(f'  Tested {print_name}: NATIVE={best['accuracy']:.04%} ({best['count'] - best['num_correct']} incorrect of {best['count']})')



def save_mechinterp_data(model_folder:Path, native_accuracy:float):
    mech_file = model_folder / 'mech.json'
    if mech_file.exists():
        return
    state_dict = torch.load(model_folder / 'model.pth', map_location='cpu')
    symbol_embeddings:Tensor = state_dict['embed.weight'][TSARSynthDataset.zero_symbol:].cuda()
    qkv_proj:Tensor = state_dict['attn.qkv_proj.weight'].cuda()
    qkv_bias:Tensor = state_dict['attn.qkv_proj.bias'].cuda()
    d = qkv_proj.shape[0]//3
    wqv, wqn = qkv_proj[:d].chunk(2, dim=-1)
    wkv, wkn = qkv_proj[d:2*d].chunk(2, dim=-1)
    wvv, wvn = qkv_proj[2*d:].chunk(2, dim=-1)

    Tr_embeddings = state_dict['embed.weight'][TSARSynthDataset.t_query].cuda().repeat(symbol_embeddings.shape[0], 1)
    MeanSym_embeddings = symbol_embeddings.mean(dim=0).repeat(symbol_embeddings.shape[0], 1)

    q = F.linear(torch.concat([Tr_embeddings, symbol_embeddings], dim=-1), qkv_proj[:d], qkv_bias[:d])
    k = F.linear(torch.concat([symbol_embeddings, MeanSym_embeddings], dim=-1), qkv_proj[d:2*d], qkv_bias[d:2*d])
    v = F.linear(torch.concat([MeanSym_embeddings, symbol_embeddings], dim=-1), qkv_proj[2*d:], qkv_bias[2*d:])
    
    qk_aligns = (q * k).sum(dim=-1) / (q.norm(p=2, dim=-1)*k.norm(p=2, dim=-1))

    W_tensors = [wqv, wqn, wkv, wkn, wvv, wvn]
    W_colmags = [x.norm(dim=0, keepdim=False) for x in W_tensors]
    W_svds = [torch.linalg.svdvals(x) for x in W_tensors]
    W_names = ['wqv', 'wqn', 'wkv', 'wkn', 'wvv', 'wvn']
    W_stableranks = [(x**2).sum()/(svd[0] ** 2) for x, svd in zip(W_tensors, W_svds)]
    # W_entropyrank = [torch.linalg.svd(x)[1] for x in W_tensors]
    # W_entropyrank = [(x/x.sum()) for x in W_entropyrank]
    # W_entropyrank = [-(x * x.log()).sum() for x in W_entropyrank]
    # W_entropyrank = [x.exp() for x in W_entropyrank]

    untrained_state_dict = torch.load(model_folder / 'untrained.pth', map_location='cpu')
    untrained_symbol_embeddings = untrained_state_dict['embed.weight'][TSARSynthDataset.t_query:].cuda()

    def _inplace_max_align(w:Tensor):
        w.div_(w.norm(p=2, dim=-1, keepdim=True))
        w2 = (w @ w.T).fill_diagonal_(-1)
        return torch.clamp(w2.max(), -1.0, 1.0)
    max_align = _inplace_max_align(symbol_embeddings)
    untrained_max_align = _inplace_max_align(untrained_symbol_embeddings)
    q_maxalign = _inplace_max_align(q)
    k_maxalign = _inplace_max_align(k)
    v_maxalign = _inplace_max_align(v)


    gv, gn = state_dict['attn_norm.weight'].cuda().chunk(2, dim=-1)

    _save_json(mech_file, {
        'model': {
            'args': {'d_qk': d, 'n': symbol_embeddings.shape[0], 'd_e': symbol_embeddings.shape[1]},
            'success': native_accuracy == 1.0,
            'accuracy': native_accuracy,
            'max_align': max_align.item(),
            'untrained_max_align': untrained_max_align.item(),
            'q_maxalign': q_maxalign.item(),
            'k_maxalign': k_maxalign.item(),
            'v_maxalign': v_maxalign.item(),
            'qk_minalign': qk_aligns.min().item(),
            'qk_meanalign': qk_aligns.mean().item(),
        },
        'w': {
            name: {
                'wmean': mags.mean().item(),
                'wstd': mags.std().item(),
                'svdmean': svd.mean().item(),
                'svdratio': svd.max().item()/svd.min().item(),
                'rank': rank.item(),
            }
            for name, mags, svd, rank in zip(W_names, W_colmags, W_svds, W_stableranks)
        },
        'g': {
            name: {
                'norm': g.norm(p=2).item(),
                'mean': g.mean().item(),
                'std': g.std().item(),
            }
            for name, g in [('gv', gv), ('gn', gn)]
        }
    })

def generate_mechinterp_tables(model_folder:Path, out_path:Path):
    W_names = ['wqv', 'wqn', 'wkv', 'wkn', 'wvv', 'wvn']
    W_row_names = [
        r'$W_{Q \text{verb}}$', r'$W_{Q \text{noun}}$',
        r'$W_{K \text{verb}}$', r'$W_{K \text{noun}}$',
        r'$W_{V \text{verb}}$', r'$W_{V \text{noun}}$',
    ]

    mdata = _load_json(os.path.join(model_folder, 'mech.json'))
    wdata = [mdata['w'][x] for x in W_names]
    gv, gn = mdata['g']['gv'], mdata['g']['gn']
    dim = mdata['model']['args']['d_qk']

    all_rows = []
    all_rows.extend([
        f'$d_k={dim}$' + r' & $\text{Mean}(\Vert W_i \Vert)$ & $\text{Std}(\Vert W_i \Vert)$ & SVD $\text{mean}$ & SVD $\frac{\max}{\min}$ \\',
        r'\midrule',
        *[f'{name} & {row['wmean']:.04f} & {row['wstd']:.04f} & {row['svdmean']:.04f} & {row['svdratio']:.04f} \\\\'
        for name, row in zip(W_row_names, wdata)]
    ])
    all_rows.extend([''])

    all_rows.extend([
        f'$d_k={dim}$' + r' & $\Vert G \Vert$ & $\text{Mean}(g_i)$ & $\text{Std}(g_i)$ \\',
        r'\midrule',
        r'$G_\text{verb}$' + f' & {gv['norm']:.04f} & {gv['mean']:.04f} & {gv['std']:.04f} \\\\',
        r'$G_\text{noun}$' + f' & {gn['norm']:.04f} & {gn['mean']:.04f} & {gn['std']:.04f} \\\\',
    ])
    all_rows.extend([''])

    g_vn_ratio = gv['norm'] / gn['norm']
    wq_snr = mdata['w']['wqn']['wmean'] / mdata['w']['wqv']['wmean']
    wk_snr = mdata['w']['wkv']['wmean'] / mdata['w']['wkn']['wmean']
    wv_snr = mdata['w']['wvn']['wmean'] / mdata['w']['wvv']['wmean']

    all_rows.extend([
        f'$d_k={dim}$' + r' & $\text{Mean}(\Vert W_i \Vert)$ SNR & With $G$ \\',
        r'\midrule',
        r'$W_Q$' + f' & {wq_snr:.02f} & {wq_snr/g_vn_ratio:.02f} \\\\',
        r'$W_K$' + f' & {wk_snr:.02f} & {wk_snr*g_vn_ratio:.02f} \\\\',
        r'$W_V$' + f' & {wv_snr:.02f} & {wv_snr/g_vn_ratio:.02f} \\\\',
    ])
    all_rows.extend([''])
    
    with open(out_path, 'wt', -1, 'utf-8') as f:
        f.write('\n'.join(all_rows) + '\n\n')



def run_base_training(models_dir:Path, charts_dir:Path, repeats:int, base_lr:float, weight_decay:float,
                      Nd_vals:dict[int,list[int]], needs_pca, device, emb_mult=1, permit_early_stopping=False):
    T = 16
    needs_pca = set(needs_pca)

    for N, dims in Nd_vals.items():
        folder_Ngroup = models_dir / f'KV{N:05d}' / f'trained-{T:02d}'
        training_steps = [
            (2, 1, 64),
            (min(16, N//(4*T)), 1, 64),
            (N//(2*T), 1, 64),
            (N//T, 0.5, 64),
            (N//T, 0.25, 64),
            (N//T, 0.125, 64),
            (N//T, 0.0625, 64),
            (N//T, 0.03125, 64),
        ]

        for d in dims:
            folder_Dgroup = folder_Ngroup / f'dim-{d:04}'
            for i in range(repeats):
                model_folder = folder_Dgroup / f'model-{i:02d}'
                model = None
                if not _is_model_finished(model_folder):
                    print(f'Creating new model ({models_dir.name}, {N}/{T} d={d} v{i}) in {model_folder}')
                    _clean_model_folder(model_folder)
                    new_model_args = {
                        'vocab_size': TSARSynthDataset.vocab_size(N),
                        'd_model': d*emb_mult*2, 'd_qk': d, 'd_v': d, 'tau': d**0.5,
                        'features': {
                            'with_mlp': False
                        },
                        'dtype': torch.float32,
                        'device': device,
                    }
                    this_training = [(x[0], x[1] * base_lr / d, *x[2:]) for x in training_steps]
                    model = SimpleAttentionModel(**new_model_args)
                    run_training_curriculum(model, model_folder, this_training,
                                            weight_decay=weight_decay, freeze_symbols=False)
                    
                report = run_test_battery(model, model_folder, 'accuracy.json', device, extra_data={'t': T})
                save_mechinterp_data(model_folder, report['native']['accuracy'])
                _report_accuracy(f'({models_dir.name}, {N}/{T} d={d} v{i})', report)

                if d in needs_pca and report['native']['accuracy'] == 1.0:
                    chartmaker.generate_model_pca_charts(str(charts_dir / f'embed_d{d:03d}'), str(model_folder), 'embed.weight')
                    generate_mechinterp_tables(model_folder, charts_dir / f'mechinterp_tables_{d:03d}.txt')
                    needs_pca.remove(d)
                if report['native']['accuracy'] == 1.0 and permit_early_stopping:
                    break

            gc.collect()
            torch.cuda.empty_cache()

    for d in needs_pca:
        print(f'Could not find a 16K d={d} model with 100% accuracy!')

def run_micromodel_training(models_dir:Path, charts_dir:Path, dims:list, repeats:int, device, permit_early_stopping=False):
    model_args = [
        # N, d_k, T_frac, lr_base, lr_mult, extra_epochs, weight_decay
        (32, 2, 1, 0.005, 0.85, 20, 0.05),
        (512, 3, 2, 0.01, 0.8, 16, 0.1),
        (1024, 4, 4, 0.01, 0.8, 16, 0.15),
        (8192, 5, 16, 0.01, 0.85, 16, 0.15),
        (16384, 6, 16, 0.01, 0.9, 20, 0.15),
    ]
    if dims:
        model_args = [x for x in model_args if x[1] in dims]

    needs_pca = set([3])
    batches = 64
    bsz_budget = 16384

    for N, d, T, lr_base, lr_mult, extra_epochs, weight_decay in model_args:
        folder_group = models_dir / f'KV{N:05d}' / f'trained-{T:02d}' / f'dim-{d:04}'
        max_train_seq = (N//T)
        n_sizes = [int((i+1)/4 * max_train_seq) for i in range(4)]
        training_steps = [
            (2, lr_mult, (batches, bsz_budget//2)),
            *[(n, lr_mult, (batches, bsz_budget//n))
              for n in n_sizes],
            *[(max_train_seq, lr_mult**(1+i), (batches, bsz_budget//max_train_seq))
              for i in range(extra_epochs)],
        ]

        for i in range(repeats):
            model_folder = folder_group / f'model-{i:02d}'
            model = _load_model_if_finished(model_folder, device)
            if model is None:
                print(f'Creating new model (Micro, {N}/{T} d={d} v{i}) in {model_folder}')
                _clean_model_folder(model_folder)
                new_model_args = {
                    'vocab_size': TSARSynthDataset.vocab_size(N),
                    'd_model': d*2, 'd_qk': d, 'd_v': d, 'tau': d**0.5,
                    'features': {
                        'with_mlp': False,
                        'force_precision': False,
                    },
                    'dtype': torch.float32,
                    'device': device,
                }
                this_training = [(x[0], x[1] * lr_base, *x[2:]) for x in training_steps]
                model = SimpleAttentionModel(**new_model_args)
                run_training_curriculum(model, model_folder, this_training,
                                        weight_decay=weight_decay, freeze_symbols=False)
                
            report = run_test_battery(model, model_folder, 'accuracy.json', device, extra_data={'t': T})
            save_mechinterp_data(model_folder, report['native']['accuracy'])
            _report_accuracy(f'(Micro, {N}/{T} d={d} v{i})', report)

            if d in needs_pca and report['native']['accuracy'] == 1.0:
                chartmaker.generate_model_pca_charts(str(charts_dir / f'embed_d{d:03d}'), str(model_folder), 'embed.weight')
                needs_pca.remove(d)
            if report['native']['accuracy'] == 1.0 and permit_early_stopping:
                break

        gc.collect()
        torch.cuda.empty_cache()

    for d in needs_pca:
        print(f'Could not find a d={d} model with 100% accuracy!')

def run_frozen_embed_training(models_dir:Path, charts_dir:Path, repeats:int, device):
    D_vals = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14]
    N_vals = {
        256:   D_vals,
        512:   D_vals,
        1024:   D_vals,
        2048:   D_vals,
        4096:   D_vals,
        8192:   D_vals,
        16384:  D_vals,
    }
    T = 16
    
    Nk_folders = []
    for N, dims in N_vals.items():
        folder_Ngroup = models_dir / f'KV{N:05d}'
        Nk_folders.append(folder_Ngroup)
        folder_Tgroup = folder_Ngroup / f'trained-{T:02d}'
        base_lr = 5e-2
        training_steps = [
            (2, 1, 64),
            (min(16, N//T), 1, 64),
            (min(64, N//T), 1, 64),
            (N//T, 0.5, 64),
            (N//T, 0.25, 64),
            (N//T, 0.125, 64),
            (N//T, 0.0625, 64),
            (N//T, 0.03125, 64),
        ]

        for d in dims:
            folder_Dgroup = folder_Tgroup / f'dim-{d:04}'

            for i in range(repeats):
                model_folder = folder_Dgroup / f'model-{i:02d}'
                model = _load_model_if_finished(model_folder, device)
                if model is None:
                    print(f'Creating new model (Frozen, {N}/{T} d={d} v{i}) in {model_folder}')
                    _clean_model_folder(model_folder)
                    new_model_args = {
                        'vocab_size': TSARSynthDataset.vocab_size(N),
                        'd_model': d*2, 'd_qk': d, 'd_v': d, 'tau': d**0.5,
                        'features': {
                            'with_mlp': False,
                        },
                        'dtype': torch.float32,
                        'device': device,
                    }
                    this_training = [(x[0], x[1] * base_lr * (d*2)**-0.5, *x[2:]) for x in training_steps]
                    model = SimpleAttentionModel(**new_model_args)
                    model.embed.weight.data[2:].div_(model.embed.weight.data[2:].norm(p=2, dim=-1, keepdim=True))
                    run_training_curriculum(model, model_folder, this_training,
                                            weight_decay=0.1, freeze_symbols=True)

                report = run_test_battery(model, model_folder, 'accuracy.json', device, extra_data={'t': T})
                save_mechinterp_data(model_folder, report['native']['accuracy'] == 1.0)
                generate_mechinterp_tables(model_folder, model_folder / 'mechinterp_tables.txt')
                _report_accuracy(f'(Frozen, {N}/{T} d={d} v{i})', report)
                
        gc.collect()
        torch.cuda.empty_cache()

    reports = [[_load_json(os.path.join(root, 'accuracy.json'))
                for root, dirs, files in os.walk(group_folder) if 'accuracy.json' in files]
                for group_folder in Nk_folders]
    
    chartmaker.generate_graph_accuracy_by_dim(charts_dir / f'acc_by_dim_frozen.png',
                                              [f'N={x}' for x in N_vals.keys()],
                                              [[r['native'] for r in x] for x in reports],
                                              mode='max')

def run_constructed_d2(models_dir:Path, charts_dir:Path, device):
    N_vals = [2**x + sub * 2**(x-2) for x in range(7, 14) for sub in range(4)] + [16*1024, 18*1024, 20*1024]
    d = 2
    
    Nk_folders = []
    for N in N_vals:
        model_folder = models_dir / f'KV{N:06d}'
        Nk_folders.append(model_folder)
        new_model_args = {
            'vocab_size': TSARSynthDataset.vocab_size(N),
            'd_model': d*2, 'd_qk': d, 'd_v': d, 'tau': 1,
            'features': {
                'with_mlp': False,
                'force_precision': True,
            },
            'dtype': torch.float64,
            'device': device,
        }
        print(f"Constructed (UnitCircle, {N}, d=2)")
        model = SimpleAttentionModel(**new_model_args)
        constructions.construct_model_unitcircle(model)
        model.save(model_folder)
        report = run_test_battery(model, model_folder, 'accuracy.json', device)
        print(f'  Tested ({N} d={d}): NATIVE={report['native']['accuracy']:.04%}, FP32={report['fp32']['accuracy']:.04%}, FP16={report['fp16']['accuracy']:.04%}')
        
    reports = [_load_json(os.path.join(root, 'accuracy.json'))
               for group_folder in Nk_folders
               for root, dirs, files in os.walk(group_folder) if 'accuracy.json' in files]
    
    reports_native = {N: [r['native']['accuracy'] for r in reports if r['native']['n'] == N] for N in N_vals}
    reports_native = {k:v for k,v in reports_native.items() if v}
    reports_fp32 = {N: [r['fp32']['accuracy'] for r in reports if r['fp32']['n'] == N] for N in N_vals}
    reports_fp32 = {k:v for k,v in reports_fp32.items() if v}
    reports_fp16 = {N: [r['fp16']['accuracy'] for r in reports if r['fp16']['n'] == N] for N in N_vals}
    reports_fp16 = {k:v for k,v in reports_fp16.items() if v}
    
    chartmaker.generate_graph_accuracy(charts_dir / f'acc_by_n_d2.png',
                                       ['FP64', 'FP32', 'FP16'],
                                       [reports_native, reports_fp32, reports_fp16],
                                       xlabel='N', xlog=True, mode='max')
    
    magtable_rows = [f'{N} & \\num{{{math.ceil(constructions.calc_magnitude(2*math.pi/N, N))}}} \\\\'
                     for N in [32,256,1024,4096,16384]]
    _save_text(charts_dir / 'd2_mag_table.txt',
               '\n'.join(magtable_rows))

def run_constructed_dx(models_dir:Path, charts_dir:Path, device):
    D_vals = [2, 3, 4, 5, 6, 7, 8]
    L_vals = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]

    for d in D_vals:
        for L in L_vals:
            N = constructions.hypergrid_N(d, L)
            if N > 40000:  # my vram is smol
                continue
            model_folder = models_dir / f'dim-{d:02d}/L-{L:02d}'
            new_model_args = {
                'vocab_size': TSARSynthDataset.vocab_size(N),
                'd_model': d*2, 'd_qk': d, 'd_v': d, 'tau': 1,
                'features': {
                    'with_mlp': False,
                },
                'dtype': torch.float32,
                'device': device,
            }
            model = SimpleAttentionModel(**new_model_args)
            constructions.construct_model_hypergrid(model, L)
            model.save(model_folder)
            report = run_test_battery(model, model_folder, 'accuracy.json', device)
            _report_accuracy(f'Constructed (Lattice, d={d}, L={L}, N={N})', report)

    
    
    print('Building Tables')
    chart_D = [2, 8, 16, 64]
    chart_L = [2, 4, 6, 8]
    dx_table_text = []
    def _fmt(x):
        return f'${x}$' if x < 1e7 else f'$\\num{{{x:.02e}}}$'
    for d in chart_D:
        real_N = [constructions.hypergrid_N(d, L) for L in chart_L]
        est_N = [L**d for L in chart_L]
        dx_table_text.append(f'$d_k={d}$ & ' + ' & '.join(_fmt(rN) for rN, eN in zip(real_N, est_N)) + ' \\\\')

    _save_text(charts_dir / 'dx_n_table.txt',
               '\n'.join(dx_table_text))
    

    chart_B = [4, 8, 16, 32]
    chart_D = [2, 4, 8, 16, 32]
    b_table_text = []
    def _fmtpct(rN, eN):
        return f'{100*rN/eN:.04f}\\%' if rN != eN else '100\\%'
    for b in chart_B:
        real_N = [constructions.hypergrid_N(d, 2**(b//d)) if (b%d)==0 else 0 for d in chart_D]
        est_N = [(2**(b//d))**d if (b%d)==0 else 0 for d in chart_D]
        b_table_text.append(f'$2^{{{b}}}$ & ' + ' & '.join(_fmtpct(rN, eN) if rN else '' for rN, eN in zip(real_N, est_N)) + ' \\\\')
    _save_text(charts_dir / 'dx_b_table.txt',
               '\n'.join(b_table_text))



    
def run_constructed_randsphere(models_dir:Path, charts_dir:Path, repeats:int, device):
    D_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    N_vals = [256, 512, 1024, 2048, 4096, 8192, 16384]

    Nk_folders = []
    for N in N_vals:
        Nk_folders.append(models_dir / f'KV{N:06d}')
        for d in D_vals:
            dim_folder = Nk_folders[-1] / f'dim-{d:02d}'
            for i in range(repeats):
                model_folder = dim_folder / f'model-{i:02d}'
                model = _load_model_if_finished(model_folder, device)
                if model is None:
                    print(f'Constructed New (RandSphere, d={d} N={N} v{i}) in {model_folder}')
                    new_model_args = {
                        'vocab_size': TSARSynthDataset.vocab_size(N),
                        'd_model': d*2, 'd_qk': d, 'd_v': d, 'tau': 1,
                        'features': {
                            'with_mlp': False,
                        },
                        'dtype': torch.float32,
                        'device': device,
                    }
                    model = SimpleAttentionModel(**new_model_args)
                    constructions.construct_model_randomsphere(model, 1.0)
                    model.save(model_folder)
                report = run_test_battery(model, model_folder, 'accuracy.json', device)
                _report_accuracy(f'(RandSphere, d={d} N={N} v{i})', report)


    reports = [[_load_json(os.path.join(root, 'accuracy.json'))
                for root, dirs, files in os.walk(group_folder) if 'accuracy.json' in files]
                for group_folder in Nk_folders]
    
    chartmaker.generate_graph_accuracy_by_dim(charts_dir / f'acc_by_dim_randsphere.png',
                                              [f'N={x}' for x in N_vals],
                                              [[r['native'] for r in x] for x in reports],
                                              mode='max')



def _expected_stable_rank(m, n):
    return m*n / ((m**0.5 + n**0.5)**2)

def generate_mechinterp_charts(saveto:Path, model_folders:list, N_val:int=0):
    os.makedirs(saveto, exist_ok=True)
    mdatas = [_load_json(os.path.join(x, 'mech.json')) for x in model_folders]
    
    repeat_groups = {}
    for mdata in mdatas:
        key = (mdata['model']['args']['n'], mdata['model']['args']['d_qk'])
        if N_val and key[0] != N_val:
            continue
        repeat_groups.setdefault(key, []).append(mdata)

    align_charts = [{}, {}, {}, {}, {}]
    mag_chart = [{}, {}]

    snr_chart = [{} for _ in range(6)]

    for key, group in repeat_groups.items():
        def _mark(x):
            if x['model']['accuracy'] < 0.99:
                return 2
            elif x['model']['accuracy'] < 1.0:
                return 3
            return 1
        
        align_charts[0][key[1]] = [(x['model']['max_align'], _mark(x)) for x in group]
        align_charts[1][key[1]] = [(x['model']['q_maxalign'], _mark(x)) for x in group]
        align_charts[2][key[1]] = [(x['model']['k_maxalign'], _mark(x)) for x in group]
        align_charts[3][key[1]] = [(x['model']['v_maxalign'], _mark(x)) for x in group]
        align_charts[4][key[1]] = [(x['model']['qk_minalign'], _mark(x)) for x in group]

        mag_chart[0][key[1]] = [(x['g']['gn']['norm']*x['w']['wqn']['wmean']*x['g']['gv']['norm']*x['w']['wkv']['wmean'], _mark(x)) for x in group]
        mag_chart[1][key[1]] = [(x['g']['gn']['norm']*x['w']['wqn']['wmean']*x['g']['gv']['norm']*x['w']['wkv']['wmean']*(key[1]**-0.5), _mark(x)) for x in group]

        snr_chart[0][key[1]] = [(x['g']['gn']['norm']/x['g']['gv']['norm'] * x['w']['wqn']['wmean']/x['w']['wqv']['wmean'], _mark(x)) for x in group]
        snr_chart[1][key[1]] = [(x['g']['gv']['norm']/x['g']['gn']['norm'] * x['w']['wkv']['wmean']/x['w']['wkn']['wmean'], _mark(x)) for x in group]
        snr_chart[2][key[1]] = [(x['g']['gv']['norm']/x['g']['gn']['norm'] * x['w']['wvn']['wmean']/x['w']['wvv']['wmean'], _mark(x)) for x in group]

        snr_chart[3][key[1]] = [(x['w']['wqn']['rank'], _mark(x)) for x in group]
        snr_chart[4][key[1]] = [(x['w']['wkv']['rank'], _mark(x)) for x in group]
        snr_chart[5][key[1]] = [(x['w']['wvn']['rank'], _mark(x)) for x in group]

    magsep_args = dict(
        plot_titles=[''],
        rowscols=(1, 1),
        figsize=(7, 2.25),
        xylines=[None],
    )
    chartmaker.generate_mechinterp_scatter(
        data=align_charts[:1],
        saveto=os.path.join(saveto, 'mechinterp_trend_align.png'),
        ylabel='Alignment',
        xlabel='Head Dimension (log scale)',
        islog=[(True, False)],
        # ylims=[(None, 1)],
        **magsep_args,  # type: ignore
    )
    chartmaker.generate_mechinterp_scatter(
        data=align_charts[1:],
        saveto=os.path.join(saveto, 'mechinterp_trend_projalign.png'),
        ylabel='Alignment',
        xlabel='Head Dimension (log scale)',
        islog=[(True, False)]*4,
        # ylims=[(None, 1)],
        plot_titles=['Q Max Align', 'K Max Align', 'V Max Align', 'QK Min Align'],
        rowscols=(4, 1),
        figsize=(7, 2.25*4),
        xylines=[None]*4,
    )
    
    chartmaker.generate_mechinterp_scatter(
        data=mag_chart[:1],
        saveto=os.path.join(saveto, 'mechinterp_trend_mag.png'),
        ylabel='Magnitude\n(log scale)',
        xlabel='Head Dimension (log scale)',
        islog=[(True, True)],
        **magsep_args,  # type: ignore
    )
    chartmaker.generate_mechinterp_scatter(
        data=mag_chart[1:],
        saveto=os.path.join(saveto, 'mechinterp_trend_magdiv.png'),
        ylabel='Adjusted Magnitude\n(log scale)',
        xlabel='Head Dimension (log scale)',
        islog=[(True, True)],
        **magsep_args,  # type: ignore
    )
    chartmaker.generate_mechinterp_scatter(
        data=snr_chart[:3],
        saveto=os.path.join(saveto, 'mechinterp_trend_snr.png'),
        ylabel='SNR (log scale)',
        xlabel='Head Dimension (log scale)',
        plot_titles=['$W_Q$ SNR', '$W_K$ SNR', '$W_V$ SNR'],
        rowscols=(1, 3),
        figsize=(9, 2.5),
        islog=[True]*3,
        xylines=[None]*3,
        sharey=True,
        ylocator='log',
    )
    chartmaker.generate_mechinterp_scatter(
        data=snr_chart[3:],
        saveto=os.path.join(saveto, 'mechinterp_trend_rank.png'),
        ylabel='Stable Rank (log scale)',
        xlabel='Head Dimension (log scale)',
        plot_titles=['$W_Q$ Noun Stable Rank', '$W_K$ Verb Stable Rank', '$W_V$ Noun Stable Rank'],
        rowscols=(1, 3),
        figsize=(9, 2.5),
        islog=[True]*3,
        xylines=[[(lambda x:x, 'black')]]*3,
        sharey=True,
        ylocator='log',
    )

def generate_grouped_mechinterp_charts(saveto, group_names:list[str], model_groups:list[list[Path]], emb_mults:list=[]):
    def _mark(x):
        if x['model']['accuracy'] < 0.99:
            return 2
        elif x['model']['accuracy'] < 1.0:
            return 3
        return 1
    
    os.makedirs(saveto, exist_ok=True)
    groups_by_dk = []
    for group in model_groups:
        dks = {}
        for folder in group:
            mdata = _load_json(folder / 'mech.json')
            dks.setdefault(mdata['model']['args']['d_qk'], []).append(mdata)
        groups_by_dk.append(dks)

    chart_embeddings = []
    chart_aligns = []
    chart_projaligns = []
    charts_mags = [[], []]
    chart_snr = []
    chart_rank = []

    for groupidx, group in enumerate(groups_by_dk):
        succeeded = {dk: [x for x in mdatas if x['model']['accuracy'] > 0.99] for dk, mdatas in group.items()}
        chart_embeddings.append({
            dk: [100 * sum(1.0 for x in mdatas if x['model']['untrained_max_align'] < x['model']['max_align']) / len(mdatas)]
            for dk, mdatas in succeeded.items() if mdatas
        })

        chart_projaligns.append([
            {dk: [x['model']['max_align'] for x in mdatas] for dk, mdatas in succeeded.items()},
            {dk: [x['model']['qk_minalign'] for x in mdatas] for dk, mdatas in succeeded.items()},
            {dk: [x['model']['q_maxalign'] for x in mdatas] for dk, mdatas in succeeded.items()},
            {dk: [x['model']['k_maxalign'] for x in mdatas] for dk, mdatas in succeeded.items()},
            {dk: [x['model']['v_maxalign'] for x in mdatas] for dk, mdatas in succeeded.items()},
        ])

        chart_aligns.append({dk: [(x['model']['max_align'], _mark(x)) for x in mdatas]
                             for dk, mdatas in group.items()})
        
        charts_mags[0].append({dk: [(x['g']['gn']['norm']*x['w']['wqn']['wmean']*x['g']['gv']['norm']*x['w']['wkv']['wmean'], _mark(x)) for x in mdatas]
                              for dk, mdatas in group.items()})
        charts_mags[1].append({dk: [(x['g']['gn']['norm']*x['w']['wqn']['wmean']*x['g']['gv']['norm']*x['w']['wkv']['wmean']*(dk**-0.5), _mark(x)) for x in mdatas]
                              for dk, mdatas in group.items()})

        chart_snr.extend([
            {dk: [(x['g']['gn']['norm']/x['g']['gv']['norm'] * x['w']['wqn']['wmean']/x['w']['wqv']['wmean'], _mark(x)) for x in mdatas]
             for dk, mdatas in group.items()},
            {dk: [(x['g']['gv']['norm']/x['g']['gn']['norm'] * x['w']['wkv']['wmean']/x['w']['wkn']['wmean'], _mark(x)) for x in mdatas]
             for dk, mdatas in group.items()},
            {dk: [(x['g']['gv']['norm']/x['g']['gn']['norm'] * x['w']['wvn']['wmean']/x['w']['wvv']['wmean'], _mark(x)) for x in mdatas]
             for dk, mdatas in group.items()}
        ])
        chart_rank.extend([
            {dk: [(x['w']['wqn']['rank'], _mark(x)) for x in mdatas] for dk, mdatas in group.items()},
            {dk: [(x['w']['wkv']['rank'], _mark(x)) for x in mdatas] for dk, mdatas in group.items()},
            {dk: [(x['w']['wvn']['rank'], _mark(x)) for x in mdatas] for dk, mdatas in group.items()},
        ])

    path = Path(saveto)

    chartmaker.generate_graph_accuracy(
        str(path / 'grouped_trend_embedding_quality.png'), group_names, chart_embeddings,
        xlabel='Head Dimension (log scale)', ylabel='Percent', xlog=True, mode='mean'
    )

    for gi, data in enumerate(chart_projaligns):
        chartmaker.generate_graph_accuracy(
            str(path / f'grouped_trend_projalign_{gi}.png'),
            ['Embed Coherence', 'QK Min Align', 'Q Coherence', 'K Coherence', 'V Coherence'],
            data,
            xlabel='Head Dimension', ylabel='Alignment',
            xlog=True, mode='mean'
        )

    chartmaker.generate_mechinterp_scatter(
        data=chart_aligns,
        saveto=str(path / 'grouped_trend_align.png'),
        plot_titles=group_names,
        ylabel='Alignment',
        xlabel='Head Dimension (log scale)',
        islog=[(True, False)]*len(group_names),
        rowscols=(len(group_names), 1),
        figsize=(7, 1.75*len(group_names)),
        xylines=[None]*len(group_names),
        sharex='col',
        sharey=True,
    )

    chartmaker.generate_mechinterp_scatter(
        data=charts_mags[0],
        saveto=str(Path(saveto) / f'grouped_trend_mag.png'),
        plot_titles=group_names,
        ylabel='Magnitude\n(log scale)',
        xlabel='Head Dimension (log scale)',
        islog=[(True, True)]*len(group_names),
        rowscols=(len(group_names), 1),
        figsize=(7, 1.75*len(group_names)),
        xylines=[None]*len(group_names),
        sharex='col',
        sharey=True,
        # ylocator='log',
    )
    chartmaker.generate_mechinterp_scatter(
        data=charts_mags[1],
        saveto=str(Path(saveto) / f'grouped_trend_magdiv.png'),
        plot_titles=group_names,
        ylabel='Adjusted Magnitude\n(log scale)',
        xlabel='Head Dimension (log scale)',
        islog=[(True, True)]*len(group_names),
        rowscols=(len(group_names), 1),
        figsize=(7, 1.75*len(group_names)),
        xylines=[None]*len(group_names),
        sharex='col',
        sharey=True,
        # ylocator='log',
    )

    plotnames = ['{g} $W_Q$ SNR', '{g} $W_K$ SNR', '{g} $W_V$ SNR']
    plotnames = [fmt.format(g=x) for x in group_names for fmt in plotnames]

    chartmaker.generate_mechinterp_scatter(
        data=chart_snr,
        saveto=str(path / 'grouped_trend_snr.png'),
        ylabel='SNR (log scale)',
        xlabel='Head Dimension (log scale)',
        plot_titles=plotnames,
        rowscols=(len(group_names), 3),
        figsize=(9, 2.5*len(group_names)),
        islog=[True]*3*len(group_names),
        xylines=[None]*3*len(group_names),
        sharey=True,
        sharex='col',
        ylocator='log',
    )
    plotnames = ['{g} $W_Q$ Noun S.Rank', '{g} $W_K$ Verb S.Rank', '{g} $W_V$ Noun S.Rank']
    plotnames = [fmt.format(g=x) for x in group_names for fmt in plotnames]

    emb_mults = emb_mults or [1] * len(group_names)

    chartmaker.generate_mechinterp_scatter(
        data=chart_rank,
        saveto=str(path / 'grouped_trend_rank.png'),
        ylabel='Stable Rank (log scale)',
        xlabel='Head Dimension (log scale)',
        plot_titles=plotnames,
        rowscols=(len(group_names), 3),
        figsize=(9, 2.5*len(group_names)),
        islog=[True]*3*len(group_names),
        xylines=[[((lambda x:x), 'black'), ((lambda x, mx=mx: _expected_stable_rank(x, mx*x)), 'red')]
                 for mx in emb_mults for _ in range(3)],
        sharey=True,
        sharex='col',
        ylocator='log',
    )

def collect_model_tables(table_path, models_dirs:list, with_t:bool=False):
    model_groups = {}
    for groupidx, groupdir in enumerate(models_dirs):
        for folder in [root for root, _, files in os.walk(groupdir) if 'accuracy.json' in files]:
            mdata = _load_json(Path(folder) / 'accuracy.json')
            key = (mdata['native']['d_qk'], groupidx)
            model_groups.setdefault(key, []).append(mdata)

    dimensions = sorted(set(x[0] for x in model_groups.keys()))
    
    def _pack(x, y):
        if y == 0:
            return f'& '
        elif x == 0:
            return '& --'
        elif x == y:
            return f'& all {y}'
        return f'& {x} of {y}'

    all_rows = []
    for d in dimensions:
        all_N = set()
        for groupidx in range(len(models_dirs)):
            models = model_groups.get((d, groupidx), [])
            all_N.update(x['native']['n'] for x in models)

        row = [f'Model({d}, {all_N.pop()})'] if len(all_N) == 1 else [f'Model({d}, $N$)']

        for groupidx in range(len(models_dirs)):
            models = model_groups.get((d, groupidx), [])
            accs = [x['native']['accuracy'] for x in models]
            perfected = sum(1 for x in accs if x == 1.0)
            flawed = sum(1 for x in accs if 0.99 <= x < 1.0)
            total = len(accs)
            if with_t:
                T = models[0]['native']['t']
                assert all(x['native']['t'] == T for x in models), 'Models do not all share the same T values.'
                row.append(f'& $N/{T}$')
            row.append(_pack(perfected, total))
            row.append(_pack(perfected+flawed, total))
        all_rows.append(' '.join(row) + r' \\')
        
    with open(table_path, 'wt', -1, 'utf-8') as f:
        f.write('\n'.join(all_rows))



def compare_trained_to_constructed(chart_name:Path, models_dir:Path):
    model_folders = [root for root, _, files in os.walk(models_dir) if 'accuracy.json' in files]
    mdatas = [(_load_json(Path(x) / 'accuracy.json'), x) for x in model_folders]
    
    model_groups = {}
    for mdata, folder in mdatas:
        key = (mdata['native']['d_qk'], mdata['native']['n'], mdata['native']['t'])
        mdata['folder'] = folder
        model_groups.setdefault(key, []).append(mdata)

    all_rows = []
    for (d, N, T), models in model_groups.items():
        alignments = []
        magnitudes = []
        for mdata in models:
            if mdata['native']['accuracy'] != 1.0:
                continue
            mech = _load_json(Path(mdata['folder']) / 'mech.json')
            alignments.append(mech['model']['max_align'])
            magnitudes.append(mech['g']['gn']['norm']*mech['w']['wqn']['wmean']*mech['g']['gv']['norm']*mech['w']['wkv']['wmean'] * (d**-0.5))

        max_align = max(alignments, default=0)
        max_mag = max(magnitudes, default=0)

        # Find the hypergrid_N(d, L) that is nearest to N.
        L = 2
        while constructions.hypergrid_N(d, L+1) < N:
            L += 1
        if (N - constructions.hypergrid_N(d, L)) > (constructions.hypergrid_N(d, L+1) - N):
            L += 1
        
        grid_N = constructions.hypergrid_N(d, L)
        grid_align, grid_mag = constructions.hypergrid_align_mag(d, L)

        row = f'{d} & {L}'
        if N > grid_N:
            row += f' & \\textbf{{{N}}} & {grid_N}'
        else:
            row += f' & {N} & \\textbf{{{grid_N}}}'

        if max_align < grid_align:
            row += f' & \\textbf{{{max_align:.04f}}} & {grid_align:.04f}'
        else:
            row += f' & {max_align:.04f} & \\textbf{{{grid_align:.04f}}}'
        if max_mag < grid_mag:
            row += f' & \\textbf{{{max_mag:.0f}}} ({constructions.calc_magnitude(math.acos(max_align), N):.0f}) & {grid_mag:.0f}'
        else:
            row += f' & {max_mag:.0f} ({constructions.calc_magnitude(math.acos(max_align), N):.0f}) & \\textbf{{{grid_mag:.0f}}}'


        all_rows.extend([
            row + ' \\\\',
        ])
        
    with open(chart_name, 'wt', -1, 'utf-8') as f:
        f.write('\n'.join(all_rows))


def main(models:Path, charts:Path, run_all_tests:bool, run_all_iterations:bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    def _all_models(subdir):
        return [Path(root) for root, _, files in os.walk(subdir) if 'mech.json' in files]
    
    os.makedirs(models, exist_ok=True)
    os.makedirs(charts, exist_ok=True)


    models_n16 = models / 'base-16K'
    models_n8 = models / 'base-8K'
    models_n4 = models / 'base-4K'
    models_n2 = models / 'base-2K'

    models_wd16 = models / 'base-wd-16K'
    models_wd8 = models / 'base-wd-8K'
    models_wd4 = models / 'base-wd-4K'

    models_embed2 = models / 'base-8K-2e'
    models_embed4 = models / 'base-8K-4e'

    charts_base = charts / 'mechinterp_base'
    charts_n = charts / 'mechinterp_ablateN'
    charts_wd = charts / 'mechinterp_ablateWD'
    charts_embed = charts / 'mechinterp_ablateEmbed'
    
    models_micro = models / 'micro'

    PRIMARY_REPEATS = 20 if run_all_iterations else 1
    AUXILIARY_REPEATS = 10 if run_all_iterations else 1

    DIMS_256  = [8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256]
    DIMS_1024 = DIMS_256 + [320, 384, 448, 512, 640, 768, 896, 1024]
    base_args:dict = dict(charts_dir=charts, base_lr=2e-1, weight_decay=0.5, device=device)
    wd_args:dict = dict(charts_dir=charts, base_lr=2e-1, weight_decay=0.01, device=device)



    st = time.monotonic()


    if not run_all_tests:
        run_micromodel_training(models_micro, charts, dims=[3, 6], repeats=10, device=device, permit_early_stopping=True)
        collect_model_tables(charts / 'trained_models_micro.txt', [models_micro], with_t=True)
        compare_trained_to_constructed(charts / 'trained_vs_hypergrid.txt', models_micro)

        run_base_training(models_n16, repeats=10, Nd_vals={16384: [8, 16]}, needs_pca=[8, 16], **base_args, permit_early_stopping=True)
        generate_mechinterp_charts(charts_base, _all_models(models_n16))
        collect_model_tables(charts_base / 'trained_models.txt', [models_n16], with_t=True)

    else:

        run_base_training(models_n16, repeats=PRIMARY_REPEATS, Nd_vals={16384: DIMS_256}, needs_pca=[8, 16], **base_args)
        run_base_training(models_n8, repeats=AUXILIARY_REPEATS, Nd_vals={8192: DIMS_1024}, needs_pca=[], **base_args)
        run_base_training(models_n4, repeats=AUXILIARY_REPEATS, Nd_vals={4096: DIMS_1024}, needs_pca=[], **base_args)
        run_base_training(models_n2, repeats=AUXILIARY_REPEATS, Nd_vals={2048: DIMS_1024}, needs_pca=[], **base_args)
        generate_mechinterp_charts(charts_base, _all_models(models_n16))
        collect_model_tables(charts_base / 'trained_models.txt', [models_n16], with_t=True)
        
        generate_grouped_mechinterp_charts(charts_n, ['N=16384', 'N=8192', 'N=4096', 'N=2048'],
                                        [_all_models(models_n16), _all_models(models_n8), _all_models(models_n4), _all_models(models_n2)])
        collect_model_tables(charts_n / 'trained_models.txt', [models_n8, models_n4, models_n2])

        run_base_training(models_wd16, repeats=AUXILIARY_REPEATS, Nd_vals={16384: DIMS_256}, needs_pca=[], **wd_args)
        run_base_training(models_wd8, repeats=AUXILIARY_REPEATS, Nd_vals={8192: DIMS_1024}, needs_pca=[], **wd_args)
        run_base_training(models_wd4, repeats=AUXILIARY_REPEATS, Nd_vals={4096: DIMS_1024}, needs_pca=[], **wd_args)
        generate_grouped_mechinterp_charts(charts_wd, ['N=16384', 'N=8192', 'N=4096'],
                                        [_all_models(models_wd16), _all_models(models_wd8), _all_models(models_wd4)])
        collect_model_tables(charts_wd / 'trained_models.txt', [models_wd16, models_wd8, models_wd4])

        run_base_training(models_embed2, repeats=AUXILIARY_REPEATS, emb_mult=2, Nd_vals={8192: DIMS_1024}, needs_pca=[], **base_args)
        run_base_training(models_embed4, repeats=AUXILIARY_REPEATS, emb_mult=4, Nd_vals={8192: DIMS_1024}, needs_pca=[], **base_args)
        generate_grouped_mechinterp_charts(charts_embed, ['1x Embed', '2x Embed', '4x Embed'],
                                        [_all_models(models_n8), _all_models(models_embed2), _all_models(models_embed4)],
                                        emb_mults=[1, 2, 4])
        collect_model_tables(charts_embed / 'trained_models.txt', [models_embed2, models_embed4])


        run_frozen_embed_training(models / 'frozen', charts, repeats=AUXILIARY_REPEATS, device=device)

        run_micromodel_training(models_micro, charts, dims=[],  repeats=PRIMARY_REPEATS, device=device)
        collect_model_tables(charts / 'trained_models_micro.txt', [models_micro], with_t=True)
        compare_trained_to_constructed(charts / 'trained_vs_hypergrid.txt', models_micro)
        
        run_constructed_randsphere(models / 'rand', charts, repeats=AUXILIARY_REPEATS, device=device)

    run_constructed_d2(models / 'unit', charts, device)
    run_constructed_dx(models / 'grid', charts, device)

    print(f'Finished in {time.monotonic() - st:.01f}')
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce charts and tables from the paper")
    parser.add_argument("--short", action="store_true", help="Minimally test all model variants in the paper, including ablations.")
    parser.add_argument("--all", action="store_true", help="Reproduce all figures in the paper, including ablations, with all repeats.")
    args = parser.parse_args()

    if args.short:
        run_all_tests = True
        run_all_iterations = False
    elif args.all:
        run_all_tests = run_all_iterations = True
    else:
        run_all_tests = run_all_iterations = False

    models = Path('models-min')
    charts = Path('charts-min')
    main(models, charts, run_all_tests, run_all_iterations)


