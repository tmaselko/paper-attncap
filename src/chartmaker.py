# Copyright 2026 Theodore Maselko
# Licensed under the MIT license. See LICENSE file in the project root for details.

import os
from collections import defaultdict
from typing import Sequence, Any

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mplcolors

def _clip_labels(fig, ax, axis):
    renderer = fig.canvas.get_renderer()
    labels = ax.yaxis.get_ticklabels() if axis == 'y' else ax.xaxis.get_ticklabels()
    if not labels:
        return
    
    for l in labels:
        l.set_visible(True)
    items = [l for l in labels if l.get_text()]
    bb_prev = items.pop(0).get_window_extent(renderer)
    bb_last = items.pop(-1).get_window_extent(renderer)
    for label in items:
        bbox = label.get_window_extent(renderer)
        if bbox.overlaps(bb_prev) or bbox.overlaps(bb_last):
            label.set_visible(False)
        else:
            bb_prev = bbox



def generate_graph_accuracy(chart_path, group_names: list[str], group_results: list[dict[Any, list]],
                            xlabel:str, ylabel='Accuracy', xlog=False, ylog=False, mode: str = 'max'):
    """
    mode: 'max' - plot best accuracy with error bar to worst
          'mean' - plot mean accuracy with error bars to max and min
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    marker_styles = ['o', 's', 'D', '^', 'v', '*', 'p']
    colors = list(mplcolors.TABLEAU_COLORS.keys())
    
    all_x = set()

    for group_idx, (group_name, results) in enumerate(zip(group_names, group_results)):
        xvals = sorted(set(k for k,v in results.items() if v))
        all_x.update(xvals)
        results = [results[x] for x in xvals if results[x]]
        ymeans = [sum(ys) / len(ys) if ys else 0 for ys in results]
        ymins = [min(ys) for ys in results]
        ymaxs = [max(ys) for ys in results]

        plot_values = ymaxs if mode == 'max' else ymeans
        yerr_lower = [m - w for m, w in zip(plot_values, ymins)]
        yerr_upper = [b - m for b, m in zip(ymaxs, plot_values)]
        
        if any(a!=b for a,b in zip(ymins, ymaxs)):
            ax.errorbar(xvals, plot_values,
                        yerr=[yerr_lower, yerr_upper],
                        label=group_name,
                        marker=marker_styles[group_idx % len(marker_styles)],
                        markersize=3,
                        linewidth=1,
                        linestyle=':',
                        color=colors[group_idx % len(colors)],
                        capsize=2)
        else:
            ax.plot(xvals, plot_values,
                    label=group_name,
                    marker=marker_styles[group_idx % len(marker_styles)],
                    markersize=3,
                    linewidth=1,
                    linestyle=':',
                    color=colors[group_idx % len(colors)])
    
    ax.set_xticks(sorted(all_x))
    if xlog:
        ax.set_xscale('log')
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.LogLocator(base=2, subs='all'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    if ylog:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        # ax.yaxis.set_major_locator(ticker.LogLocator(base=2, subs='all'))
    # ax.set_yticks(sorted(set(ax.get_yticks())))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))

    fig.canvas.draw()
    _clip_labels(fig, ax, 'y')
    _clip_labels(fig, ax, 'x')


    # ax.set_xticklabels(sorted(all_x))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



def generate_graph_accuracy_by_dim(chart_path, group_names: list[str], group_results: list[list[dict]], mode: str = 'max'):
    """
    mode: 'max' - plot best accuracy with error bar to worst
          'mean' - plot mean accuracy with error bars to max and min
    """
    fig, ax = plt.subplots(figsize=(7, 2.5))
    
    marker_styles = ['o', 's', 'D', '^', 'v', '*', 'p']
    colors = list(mplcolors.TABLEAU_COLORS.keys())
    
    all_dims = set()

    min_sample_count = 999
    
    for group_idx, (group_name, results) in enumerate(zip(group_names, group_results)):
        # Aggregate by d_qk: d_qk -> list of per-model accuracies
        dim_data = {}
        
        for result in results:
            d_qk = result['d_qk']
            model_accuracy = result['accuracy']
            
            if d_qk not in dim_data:
                dim_data[d_qk] = []
            dim_data[d_qk].append(model_accuracy)

        min_sample_count = min(min_sample_count, min(len(x) for x in dim_data.values()))
        
        # Compute stats per dimension
        dims = sorted(dim_data.keys())
        all_dims.update(dims)
        
        bests = []
        worsts = []
        means = []
        for d in dims:
            acc_list = dim_data[d]
            bests.append(max(acc_list))
            worsts.append(min(acc_list))
            means.append(np.mean(acc_list))
        
        plot_values = bests if mode == 'max' else means
        yerr_lower = [m - w for m, w in zip(plot_values, worsts)]
        yerr_upper = [b - m for b, m in zip(bests, plot_values)]
        
        ax.errorbar(dims, plot_values,
                    yerr=[yerr_lower, yerr_upper],
                    label=group_name,
                    marker=marker_styles[group_idx % len(marker_styles)],
                    markersize=4,
                    linewidth=1,
                    color=colors[group_idx % len(colors)],
                    capsize=4)
    
    ax.set_xticks(sorted(all_dims))
    
    ax.set_xlabel('Head Dimension', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=1)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_stacked_graphs(chart_path, chart_title:str, x_vals:list,
                            series_names:list[str], series_plot:list, series_data:list[list],
                            plot_axis_labels:list[str], plot_logscale:list[bool], plot_extrema=None):
    n_series = len(series_names)

    # ── defaults ────────────────────────────────────────────────
    if series_plot is None:
        series_plot = list(range(n_series))

    # Build ordered list of unique subplot ids (preserving first-seen order)
    seen = set()
    plot_ids = []
    for pid in series_plot:
        if pid not in seen:
            seen.add(pid)
            plot_ids.append(pid)
    n_plots = len(plot_ids)
    pid_to_row = {pid: row for row, pid in enumerate(plot_ids)}

    if plot_axis_labels is None:
        plot_axis_labels = [None] * n_plots
    if plot_logscale is None:
        plot_logscale = [False] * n_plots
    if plot_extrema is None:
        plot_extrema = ["max"] * n_plots

    # Group series indices by subplot row
    groups = defaultdict(list)  # row -> [series indices]
    for si, pid in enumerate(series_plot):
        groups[pid_to_row[pid]].append(si)

    # ── colours: high-contrast palette, cycling if >8 series ───
    _PALETTE = [
        "#2274A5",  # blue
        "#D64933",  # red-orange
        "#3A9679",  # teal
        "#9B59B6",  # purple
        "#E8913A",  # amber
        "#17809F",  # cerulean
        "#C2375A",  # raspberry
        "#5D8C3E",  # olive green
    ]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_series)]

    # ── figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(10, 2.4 * n_plots),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    if n_plots == 1:
        axes = [axes]

    for row, ax in enumerate(axes):
        members = groups[row]
        is_multi = len(members) > 1

        if plot_logscale[row]:
            ax.set_yscale("log")

        for si in members:
            ax.plot(x_vals, series_data[si], color=colors[si],
                    linewidth=1.8, label=series_names[si])

            # Fill only when a series has a subplot to itself
            if not is_multi:
                ax.fill_between(x_vals, series_data[si],
                                alpha=0.10, color=colors[si])

        # ── labelling ───────────────────────────────────────────
        if is_multi:
            leg = ax.legend(fontsize=9, loc="center left",
                            framealpha=0.9, edgecolor="#cccccc",
                            handlelength=1.8, labelspacing=0.3)
            for text, si in zip(leg.get_texts(), members):
                text.set_color(colors[si])
                text.set_fontweight("bold")
        else:
            si = members[0]
            ax.text(
                0.01, 0.95, series_names[si],
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", color=colors[si],
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                          pad=2),
            )

        # ── Y limits (skip for log — let matplotlib auto-range) ─
        if not plot_logscale[row]:
            all_y = np.concatenate([np.asarray(series_data[si])
                                    for si in members])
            ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
            margin = (ymax - ymin) * 0.1 or 1
            ax.set_ylim(ymin - margin, ymax + margin)

        # ── extrema reference lines per series ─────────────────────
        extrema = (plot_extrema[row] or "").lower()
        show_max = extrema in ("max", "both")
        show_min = extrema in ("min", "both")

        for si in members:
            data_arr = np.asarray(series_data[si])
            for val, show in [(np.nanmax(data_arr), show_max),
                              (np.nanmin(data_arr), show_min)]:
                if not show:
                    continue
                ax.axhline(val, color=colors[si], linewidth=0.8,
                            linestyle=":", alpha=0.6)
                label = f"{val:.4g}"
                ax.text(1.01, val, label,
                        transform=ax.get_yaxis_transform(),
                        fontsize=7.5, color=colors[si], va="center",
                        fontweight="bold")

        if plot_axis_labels[row]:
            ax.set_ylabel(plot_axis_labels[row], fontsize=9)

        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.tick_params(labelsize=9)

        if plot_logscale[row]:
            # Show ticks at 1, 2, 5 subdivisions within each decade
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
            ax.yaxis.set_minor_locator(ticker.LogLocator(
                base=10, subs=list(i*0.1 for i in range(2, 10)), numticks=12))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda v, _: f"{v:g}"))
            ax.tick_params(axis="y", which="minor", length=3, width=0.5)
        else:
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(nbins=4, integer=False))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].tick_params(axis="x", labelsize=9)
    axes[-1].set_xlim(x_vals[0], x_vals[-1])
    fig.suptitle(chart_title, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.94)
    fig.align_ylabels(axes)
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pca_torch(x, n_components=3):
    centered = x - x.mean(dim=0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vh[:n_components].T
    explained_var = (S[:n_components] ** 2) / (S ** 2).sum()
    return projected, explained_var, Vh[:n_components]

def generate_pca_chart(chart_path, labels, embeddings_list):
    n = len(embeddings_list)
    fig = plt.figure(figsize=(7, 4))

    combined = torch.cat([e / e.norm(dim=1, keepdim=True).mean() for e in embeddings_list], dim=0)
    _, _, components = _pca_torch(combined, n_components=3)
    combined_mean = combined.mean(dim=0)

    for idx, (embeddings, label) in enumerate(zip(embeddings_list, labels)):
        centered = embeddings - combined_mean
        projected = (centered @ components.T).cpu().numpy()

        ax_3d = fig.add_subplot(1, n, idx + 1, projection='3d')

        # Compute screen-depth from the view angle
        elev, azim = ax_3d.elev, ax_3d.azim
        elev_r, azim_r = np.radians(elev), np.radians(azim)
        view_dir = np.array([
            np.cos(elev_r) * np.cos(azim_r),
            np.cos(elev_r) * np.sin(azim_r),
            np.sin(elev_r),
        ])
        depth_cmap = mplcolors.LinearSegmentedColormap.from_list('depth', ['#1e88e5', '#7b1fa2', '#d32f2f'])
        # Normalize per axis to match matplotlib's independent axis scaling
        normed = projected.copy()
        for i in range(3):
            mn, mx = normed[:, i].min(), normed[:, i].max()
            if mx > mn:
                normed[:, i] = (normed[:, i] - mn) / (mx - mn)
        depth = normed @ view_dir

        sc = ax_3d.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                           c=depth, cmap=depth_cmap,
                           s=3, alpha=0.8)
        ax_3d.set_title(label, fontsize=14, y=1)
        ax_3d.xaxis.set_tick_params(labelbottom=False)
        ax_3d.yaxis.set_tick_params(labelleft=False)
        ax_3d.zaxis.set_tick_params(labelleft=False)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_cosine_histogram(chart_path:str, labels, embeddings_list, max_xlim=True):
    n = len(embeddings_list)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    all_max_cos = []
    for embeddings in embeddings_list:
        normed = embeddings / embeddings.norm(dim=1, keepdim=True)
        cosines = normed @ normed.T
        cosines.fill_diagonal_(-float('inf'))
        max_cos = cosines.max(dim=1).values
        all_max_cos.append(max_cos)

    cos_min = min(c.min().item() for c in all_max_cos)
    cos_max = max(c.max().item() for c in all_max_cos)
    cos_range = 1.0 - cos_min

    cos_bins = torch.linspace(cos_min, cos_max, 101, device='cpu')
    cos_counts = [torch.histogram(c, bins=cos_bins)[0] for c in all_max_cos]
    cos_ylim = max(c.max().item() for c in cos_counts) * 1.1

    cos_cmap = mplcolors.LinearSegmentedColormap.from_list('GnRd', ['green', 'red', 'red'])

    for idx, (label, max_cos) in enumerate(zip(labels, all_max_cos)):
        ax = axes[idx]
        counts, bins, patches = ax.hist(max_cos.cpu().numpy(), bins=cos_bins.tolist())

        bin_centers = (bins[:-1] + bins[1:]) / 2
        norm_positions = (bin_centers - cos_min) / (cos_max - cos_min)
        for patch, norm_pos in zip(patches, norm_positions):
            patch.set_facecolor(cos_cmap(norm_pos))
            patch.set_alpha(1)

        mutual_coherence = max_cos.max().item()
        ax.axvline(mutual_coherence, color='darkred', linestyle='--', linewidth=1)
        ax.text(mutual_coherence - 0.01*cos_range, cos_ylim * 0.9, 'Max',
                color='darkred', fontsize=12, va='top', ha='right')

        ax.set_ylabel('Vectors', fontsize=12)
        ax.set_title(f'{label} - Maximum: {max_cos.max():.6f}', fontsize=14)
        ax.set_xlim(cos_min, 1.0 if max_xlim else None)
        ax.set_ylim(0, cos_ylim)

    ax.set_xlabel('Max. Alignment', fontsize=12)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_model_pca_charts(chart_path:str, model_folder:str, weightpath:str):
    with torch.no_grad():
        untrained = torch.load(os.path.join(model_folder, 'untrained.pth'), map_location='cpu')[weightpath]
        trained = torch.load(os.path.join(model_folder, 'model.pth'), map_location='cpu')[weightpath]

        generate_pca_chart(
            chart_path + '_pca.png',
            [f'Untrained Embeddings', f'Trained Embeddings'],
            [untrained[2:], trained[2:]],
        )
        generate_cosine_histogram(
            chart_path + '_hist.png',
            [f'Untrained Embeddings', f'Trained Embeddings'],
            [untrained[2:], trained[2:]],
        )


def generate_mechinterp_scatter(data, saveto:str, rowscols:tuple[int,int], figsize:tuple[float,float],
                                plot_titles:Sequence[str], islog:Sequence, xylines: Sequence[Sequence[tuple]|None],
                                sharey:bool|str=False, sharex:bool|str=False, xlabel:str='', ylabel:str='', ylocator=''):
    marker_map = {1: 'o', 2: 'x', 3: 'D', 4:'*'}
    color_map = {1: 'blue', 2: 'red', 3: 'orange', 4: 'green'}
    size_map = {1: 4, 2: 16, 3: 8, 4: 12}
    line_map = {1: 0, 2: 1, 3: 0, 4: 0} # X is a special snowflake that is ONLY line-width unlike all the other markers

    num_plots = rowscols[0]*rowscols[1]

    assert num_plots == len(data), 'grid/data size mismatch'

    def _plot_chart(ax, chart_idx):
        chart_data = data[chart_idx]
        allxy = [(x_val, *y) for x_val, points in chart_data.items() for y in points]
        for marker in marker_map.keys():
            matching = [x for x in allxy if x[2] == marker]
            ax.scatter([x[0] for x in matching],
                       [x[1] for x in matching],
                       marker=marker_map[marker],
                       c=color_map[marker],
                       s=size_map[marker],
                       alpha=0.7, linewidths=line_map[marker])
            
        # if xylines[chart_idx] >= 2:
        #     ax.set_ylim(8/4, None)

        if islog[chart_idx]:
            lx, ly = islog[chart_idx] if isinstance(islog[chart_idx], tuple) else [islog[chart_idx], islog[chart_idx]]
            if lx:
                ax.set_xscale('log')
                ax.xaxis.set_minor_locator(ticker.NullLocator())

            if ly:
                ax.set_yscale('log')
                ax.yaxis.set_minor_locator(ticker.NullLocator())
                if ylocator == 'log':
                    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5), numticks=5))
                else:
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            else:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                ax.set_yticks(sorted(set(ax.get_yticks())))

            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))

        ax.set_xticks(list(chart_data.keys()))
        ax.set_xticklabels([str(x)  if (x & (x - 1)) == 0 else '' for x in chart_data.keys()])
        ax.grid(axis="x", linestyle="--", alpha=0.2)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if chart_idx >= (rowscols[0]-1)*rowscols[1]:
            ax.set_xlabel(xlabel)
        if (chart_idx % rowscols[1]) == 0:
            ax.set_ylabel(ylabel)
        if num_plots>1:
            ax.set_title(plot_titles[chart_idx], fontsize=14)
        ax.tick_params(labelbottom=True)

    def _fix_chart(ax, chart_idx:int):
        xrange = [*ax.get_xlim()]
        xyl = xylines[chart_idx]
        if xyl is not None:
            # ax.autoscale(enable=False)
            for fn, color in xyl:
                yrange = [fn(x) for x in xrange]
                ax.plot(xrange, yrange, ':', color=color, linewidth=1, alpha=1.0)


    fig, axes = plt.subplots(*rowscols, squeeze=False, sharey=sharey, sharex=sharex, figsize=figsize)  # type: ignore
    for i, ax in enumerate(axes.flatten()):
        _plot_chart(ax, i)
    for i, ax in enumerate(axes.flatten()):
        ax.autoscale(enable=False)
    fig.canvas.draw()
    for i, ax in enumerate(axes.flatten()):
        _fix_chart(ax, i)
        _clip_labels(fig, ax, 'y')
        _clip_labels(fig, ax, 'x')

    # fig1.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(saveto, dpi=300, bbox_inches='tight')
    plt.close(fig)


