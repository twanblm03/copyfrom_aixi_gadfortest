"""Render archived H1-Sup predictions; no model inference or evaluation."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cafe_h1_sup_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data2/mzy/GAD/CAFE_heatmap_research_20261012')
SOURCE = ROOT / 'outputs/heatmap/place/remote_h1_sup_seed1_full_run1/heatmaps_val_0.npz'
DEST = Path(__file__).resolve().parent


def main():
    with np.load(SOURCE, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout='constrained',
                             gridspec_kw={'height_ratios': [1, 1, 1.1]})
    fig.suptitle('H1-Sup: archived VAL example, clip (13, 0)\n'
                 'Heatmap supervision and embedding membership are separate outputs', fontsize=14)
    matches = sorted(zip(data['matched_groups'], data['matched_slots']))
    for col, (group, slot) in enumerate(matches):
        for row, field in enumerate(('gt_heatmaps', 'masks')):
            values = data[field][group, 2] if row == 0 else data[field][0, slot, 2]
            im = axes[row, col].imshow(values, vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            title = f'GT group {group}' if row == 0 else f'Predicted slot {slot}'
            axes[row, col].set_title(f'{title}, frame 90')
            axes[row, col].set_xlabel('Feature-grid x (40 columns)')
            axes[row, col].set_ylabel('Feature-grid y (23 rows)')
        member = data['gt_members'][0, group, :7].astype(bool)
        actors = np.arange(1, 8)
        ax = axes[2, col]
        for actor, is_member in zip(actors, member):
            if is_member:
                ax.axvspan(actor-.48, actor+.48, color='#d8eee2', alpha=.65, zorder=0)
        ax.bar(actors-.18, data['membership'][0, slot, :7], width=.35,
               color='#2763a2', label='Embedding membership')
        ax.bar(actors+.18, data['response'][0, slot, :, :7].mean(axis=0), width=.35,
               color='#dc8645', label='Heatmap ROI response (5-frame mean)')
        ax.axhline(.5, color='#b23b3b', linestyle='--', linewidth=1,
                   label='Membership threshold only')
        ax.set_xticks(actors, [str(a)+('*' if m else '') for a,m in zip(actors,member)])
        ax.set_ylim(0, 1.05)
        ax.set_title(f'Slot {slot}: first 7 actors only')
        ax.set_xlabel('Actor ID (1-based); * = GT member of this group')
        ax.set_ylabel('Response / membership')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.colorbar(im, ax=axes[:2,:].ravel().tolist(), shrink=.85,
                 label='Gaussian target / predicted sigmoid response')
    axes[2,0].legend(fontsize=7, loc='upper right')
    fig.savefig(DEST/'saved_example.png', dpi=180)
    fig.savefig(DEST/'saved_example.pdf')
    plt.close(fig)
    print(DEST/'saved_example.png')


if __name__ == '__main__':
    main()
