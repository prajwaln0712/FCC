import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

FEATURE_NAMES = {'inception': 'Inception', 'dino': 'DINO'}
PLOT_COLOR = '#2a6fbb'  # blue: points, fit line and confidence band on every FCC-vs-gain plot


def _times_new_roman():
    """Times New Roman for all text, including mathtext (the rho / r in the legends)."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + [f for f in plt.rcParams['font.serif']
                                                        if f != 'Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    plt.rcParams['pdf.fonttype'] = 42  # embed real TrueType fonts in PDFs (journals reject Type 3)
    plt.rcParams['ps.fonttype'] = 42


def _save_vector(fig, out):
    """Save the figure as both <out stem>.pdf (journal-quality vector, fonts embedded) and
    <out stem>.svg. Both are resolution-independent: every element here (markers, fit line,
    confidence band, text) is a vector, so there is no pixel resolution to raise."""
    import os
    stem = os.path.splitext(out)[0]
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.svg', bbox_inches='tight')
    print(f'Saved {stem}.pdf and {stem}.svg')


def _stat_handles(rho, rho_p, r, r_p):
    """Two text-only legend entries, rho then r, each followed by its p-value (no test names)."""
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], linestyle='', label=f'$\\rho$ = {rho:.2f}, p = {rho_p:.3f}'),
        Line2D([], [], linestyle='', label=f'$r$ = {r:.2f}, p = {r_p:.3f}'),
    ]


# Sizes copied from plot_inception_correlation / plot_dino_correlation (title, axis labels and tick
# numbers are all 32 pt there); the legend and per-point labels are scaled down from that so they
# fit alongside the data.
PAPER_FONT = 32
PAPER_LEGEND_FONT = 22
PAPER_POINT_LABEL_FONT = 16


def _paper_legend(ax, handles, loc):
    """Legend in the style of the correlation plots' stats box: white, 50% transparent, strongly
    rounded corners (boxstyle round, radius = half the font size), thin dark outline."""
    legend = ax.legend(handles=handles, loc=loc, fontsize=PAPER_LEGEND_FONT,
                       borderpad=0.5, handlelength=0, handletextpad=0)
    frame = legend.get_frame()
    frame.set_boxstyle('round,pad=0,rounding_size=0.5')
    # Fill stays 50% transparent white, but the outline is the opaque FRAME_COLOR -- the same
    # shade as the axes outline (a single framealpha would also fade the outline a second time).
    frame.set_alpha(None)
    frame.set_facecolor((1, 1, 1, 0.5))
    frame.set_edgecolor(FRAME_COLOR)
    return legend


FRAME_COLOR = '0.25'  # axes + legend outline: 0 = pure black, 1 = white (matplotlib default is black)
GRID_COLOR = '0.92'   # grid lines: 0 = black, 1 = white (matplotlib default is 0.69)


def _light_frame(ax):
    """Grid at full width but in a lighter shade, and a slightly lighter axes outline."""
    ax.grid(True, color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(FRAME_COLOR)


def plot_runtimes():
    """
    Loads runtimes from runtimes.json and generates a bar plot.
    """
    try:
        with open('runtimes.json', 'r') as f:
            runtimes = json.load(f)
    except FileNotFoundError:
        print("Error: 'runtimes.json' not found. Please run 'fid_experiment.py' first.")
        return

    # Set font to a serif font like Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # The keys are directly from the runtimes.json file.
    # Sort the methods based on their runtime for a cleaner plot.
    sorted_methods = sorted(runtimes.keys(), key=lambda k: runtimes[k])
    times = [runtimes[method] for method in sorted_methods]

    x = np.arange(len(sorted_methods))  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects = ax.bar(x, times, width)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.set_title('Runtimes for Different Dataset Distance Methods', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods, rotation=45, ha="right", fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(0, 2600)

    # Add labels to the bars
    ax.bar_label(rects, padding=3, fmt='%.0f', fontsize=12)

    fig.tight_layout()
    plt.savefig('plots/nist/runtimes_barplot.png')
    print("Bar plot saved to 'runtimes_barplot.png'")
    plt.show()

def plot_inception_correlation():
    """
    Loads distances from distances_cache.npz and generates a scatter plot
    with a linear regression line to show the correlation between
    OTDD (Exact) and FCC (Inception) distances.
    """
    try:
        with np.load('distances_cache.npz') as data:
            otdd_exact = data['OTDD (Exact) C100->C10']
            fcc_inception = data['FCC (Inception) C100->C10']
    except FileNotFoundError:
        print("Error: 'distances_cache.npz' not found. Please run 'fid_experiment.py' first.")
        return

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.regplot(x=fcc_inception, y=otdd_exact, ax=ax, scatter_kws={'alpha':0.6})

    # Calculate correlation and p-value
    rho, p_value = pearsonr(fcc_inception, otdd_exact)
    # rho,p_value = spearmanr(fcc_inception, otdd_exact)
    ax.text(0.05, 0.95, f'$\\rho$ = {rho:.2f}\np-value = {p_value:.2f}',
            transform=ax.transAxes, fontsize=32,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))

    ax.set_xlabel('FCC (Inception) Distances', fontsize=32)
    ax.set_ylabel('OTDD (Exact) Distances', fontsize=32)
    ax.set_title('Distance Correlation', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.grid(True)

    fig.tight_layout()
    plt.savefig('plots/correlation_inception_plot.png')
    print("Correlation plot saved to 'correlation_inception_plot.png'")
    plt.show()

def plot_dino_correlation():
    """
    Loads distances from distances_cache.npz and generates a scatter plot
    with a linear regression line to show the correlation between
    OTDD (Exact) and FCC (DINO) distances.
    """
    try:
        with np.load('distances_cache.npz') as data:
            otdd_exact = data['OTDD (Exact) C100->C10']
            fcc_dino = data['FCC (DINO) C100->C10']
    except FileNotFoundError:
        print("Error: 'distances_cache.npz' not found. Please run 'fid_experiment.py' first.")
        return

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.regplot(x=fcc_dino, y=otdd_exact, ax=ax, scatter_kws={'alpha':0.6})

    # Calculate correlation and p-value
    rho, p_value = pearsonr(fcc_dino, otdd_exact)
    # rho, p_value = spearmanr(fcc_dino, otdd_exact)
    ax.text(0.05, 0.95, f'$\\rho$ = {rho:.2f}\np-value = {p_value:.2f}',
            transform=ax.transAxes, fontsize=32,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))

    ax.set_xlabel('FCC (DINOv3) Distances', fontsize=32)
    ax.set_ylabel('OTDD (Exact) Distances', fontsize=32)
    ax.set_title('Distance Correlation', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.grid(True)

    fig.tight_layout()
    plt.savefig('plots/correlation_dino_plot.png')
    print("Correlation plot saved to 'correlation_dino_plot.png'")
    plt.show()

NIST_PLOT_TITLE = 'Distance vs Adaptation: NIST Datasets'
NIST_AGGREGATOR_LABELS = {
    'mean': 'FCC (pixels, mean)',
    'ratio': 'FCC (pixels, mean/max ratio)',
}
NIST_LABEL_RADII = (11, 20, 30)  # candidate label distances from the point, in points


def _pair_initials(pair):
    src, tgt = pair.split('->')
    return f'{src[0]}→{tgt[0]}'


def _place_pair_labels(fig, ax, legend, labels, xs, ys, fontsize=13):
    """Greedy label placement: for each point pick the offset whose text box overlaps the
    fewest already-placed labels, other markers, the legend and the axes edge."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    axes_box = ax.get_window_extent(renderer)
    marker_r = 9 * px / 1.6
    markers = [Bbox([[x - marker_r, y - marker_r], [x + marker_r, y + marker_r]]) for x, y in pts]
    occupied = [legend.get_window_extent(renderer).expanded(1.03, 1.03)]

    def overlap(a, b):
        w = min(a.x1, b.x1) - max(a.x0, b.x0)
        h = min(a.y1, b.y1) - max(a.y0, b.y0)
        return w * h if w > 0 and h > 0 else 0.0

    directions = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in np.argsort(xs):
        probe = Text(0, 0, labels[i], fontsize=fontsize, family=plt.rcParams['font.family'])
        probe.set_figure(fig)
        tb = probe.get_window_extent(renderer)
        w, h = tb.width, tb.height
        best = None
        for r in NIST_LABEL_RADII:
            for ux, uy in directions:
                dx, dy = ux * r / np.hypot(ux, uy), uy * r / np.hypot(ux, uy)
                ax_px, ay_px = pts[i][0] + dx * px, pts[i][1] + dy * px
                x0 = ax_px if ux > 0 else ax_px - w if ux < 0 else ax_px - w / 2
                y0 = ay_px if uy > 0 else ay_px - h if uy < 0 else ay_px - h / 2
                box = Bbox([[x0, y0], [x0 + w, y0 + h]])
                cost = sum(overlap(box, o) for o in occupied)
                cost += sum(overlap(box, m) for j, m in enumerate(markers))
                cost += 1e4 * (w * h - overlap(box, axes_box))  # penalty for leaving the axes
                cost += r  # tie-break: prefer the closest position
                if best is None or cost < best[0]:
                    best = (cost, dx, dy, ux, uy, box)
        _, dx, dy, ux, uy, box = best
        occupied.append(box)
        ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords='offset points',
                    fontsize=fontsize, ha='left' if ux > 0 else 'right' if ux < 0 else 'center',
                    va='bottom' if uy > 0 else 'top' if uy < 0 else 'center',
                    arrowprops=dict(arrowstyle='-', color='0.55', lw=0.7, shrinkA=0, shrinkB=5))


def plot_nist_pg(aggregator):
    """
    Scatter of FCC (x) vs performance gap (y) for the 20 directed *NIST pairs,
    with a seaborn regplot fit and confidence band. aggregator: 'mean' or 'ratio'.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from nist import nist_analysis

    pairs, fcc, pg = nist_analysis.load_points(aggregator)
    stats = nist_analysis.correlations(fcc, pg)

    _times_new_roman()

    color = '#2a6fbb'
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(x=fcc, y=pg, ax=ax, color=color,
                scatter_kws={'s': 70, 'alpha': 0.9, 'edgecolor': 'white', 'linewidths': 1.5},
                line_kws={'linewidth': 2})

    ax.margins(x=0.07, y=0.08)
    ax.autoscale_view()

    handles = [
        *_stat_handles(stats['spearman'], stats['spearman_p'], stats['pearson'], stats['pearson_p']),
    ]
    legend = ax.legend(handles=handles, loc='upper left', fontsize=14, framealpha=0.9)

    ax.set_xlabel(NIST_AGGREGATOR_LABELS[aggregator], fontsize=18)
    ax.set_ylabel('Performance Gap (%)', fontsize=18)
    ax.set_title(NIST_PLOT_TITLE, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _place_pair_labels(fig, ax, legend, [_pair_initials(p) for p in pairs], fcc, pg)
    out = f'nist_fcc_{aggregator}_plot.png'
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"NIST plot saved to '{out}'")


def plot_nist_embedding(features):
    """
    Same scatter as plot_nist_pg (ratio aggregator only), but with FCC computed on network embeddings
    instead of raw pixels: one figure per network ('inception' or 'dino'), FCC mean/max ratio (x) vs
    performance gap (y) for the 20 directed *NIST pairs. Reads nist/nist_distances_cache_{features}.npz
    and nist/nist_pg_vals.json (paths are relative to this file, so it works from any working
    directory); writes plots/nist/nist_fcc_{features}_plot.png.
    """
    import os
    from matplotlib.lines import Line2D
    from nist import nist_analysis

    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, 'nist', 'nist_pg_vals.json')) as f:
        pg_by_pair = json.load(f)
    with np.load(os.path.join(root, 'nist', f'nist_distances_cache_{features}.npz')) as npz:
        fcc_by_pair = {k.split('::')[0]: float(npz[k]) for k in npz.files if k.endswith('::ratio')}
    assert set(fcc_by_pair) == set(pg_by_pair), f'{features}: pair keys differ from nist_pg_vals.json'
    pairs = sorted(pg_by_pair)
    pg = np.array([pg_by_pair[p] for p in pairs])
    fcc = np.array([fcc_by_pair[p] for p in pairs])
    stats = nist_analysis.correlations(fcc, pg)
    print(f'  [{features}/ratio] Spearman rho={stats["spearman"]:.3f} (p={stats["spearman_p"]:.3f}), '
          f'Pearson r={stats["pearson"]:.3f} (p={stats["pearson_p"]:.3f})')

    _times_new_roman()

    net = FEATURE_NAMES[features]
    color = PLOT_COLOR
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(x=fcc, y=pg, ax=ax, color=color,
                scatter_kws={'s': 70, 'alpha': 0.9, 'edgecolor': 'white', 'linewidths': 1.5},
                line_kws={'linewidth': 2})
    ax.margins(x=0.07, y=0.08)
    ax.autoscale_view()

    handles = [
        *_stat_handles(stats['spearman'], stats['spearman_p'], stats['pearson'], stats['pearson_p']),
    ]
    legend = _paper_legend(ax, handles, 'upper left')

    ax.set_xlabel(f'FCC ({net})', fontsize=PAPER_FONT)
    ax.set_ylabel('Performance Gap (%)', fontsize=PAPER_FONT)
    ax.set_title(NIST_PLOT_TITLE, fontsize=PAPER_FONT)
    ax.tick_params(axis='both', which='major', labelsize=PAPER_FONT)
    _light_frame(ax)

    fig.tight_layout()
    _place_pair_labels(fig, ax, legend, [_pair_initials(p) for p in pairs], fcc, pg,
                       fontsize=PAPER_POINT_LABEL_FONT)
    out_dir = os.path.join(root, 'plots', 'nist')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'nist_fcc_{features}_plot.svg')
    _save_vector(fig, out)
    plt.close(fig)
    print(f"NIST embedding plot saved to '{out}'")


TINYSPLIT_DIR = 'tables/tinysplit'
# score -> (inception column, dino column) in that tiny_split_result_fccN_ratio.csv (ratio only)
TINYSPLIT_COLUMNS = {
    1: ('fcc_inception192_ratio', 'fcc_dinovits16_ratio'),
    2: ('fcc_inception768_ratio', 'fcc_dinovitb16_ratio'),
}


def _load_tinysplit(score):
    """Load tables/tinysplit/tiny_split_result_fcc{score}_ratio.csv, dropping the
    appended pearson_r/spearman_rho summary rows (recognizable by an empty "target" field --
    those hold one correlation number for the whole column, not a per-pair value)."""
    path = f'{TINYSPLIT_DIR}/tiny_split_result_fcc{score}_ratio.csv'
    with open(path, newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['target'] != '']
    incep_col, dino_col = TINYSPLIT_COLUMNS[score]
    return {
        'inception': np.array([float(row[incep_col]) for row in rows]),
        'dino': np.array([float(row[dino_col]) for row in rows]),
        'gain_vs_baselineA': np.array([float(row['gain_vs_baselineA']) for row in rows]),
        'gain_vs_baselineB': np.array([float(row['gain_vs_baselineB']) for row in rows]),
    }


def plot_tinysplit(score, baseline, features):
    """
    Scatter of FCC ratio (x) vs transfer gain (y) for the 90 directed tinysplit task pairs, one
    figure per network (features = 'inception' or 'dino'), style matching plot_nist_embedding
    (seaborn regplot fit + 95% confidence band) but without per-point pair labels (that greedy
    label placement was tuned for 20 points; at 90 points here it would be unreadable clutter).
    """
    from matplotlib.lines import Line2D

    data = _load_tinysplit(score)
    x = data[features]
    gain = data[f'gain_vs_baseline{baseline}']
    net = FEATURE_NAMES[features]
    color = PLOT_COLOR

    _times_new_roman()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(x=x, y=gain, ax=ax, color=color,
                scatter_kws={'s': 55, 'alpha': 0.75, 'edgecolor': 'white', 'linewidths': 1.2},
                line_kws={'linewidth': 2})
    rho, rho_p = spearmanr(x, gain)
    r, r_p = pearsonr(x, gain)

    ax.margins(x=0.07, y=0.08)
    ax.autoscale_view()

    handles = [
        *_stat_handles(rho, rho_p, r, r_p),
    ]
    _paper_legend(ax, handles, 'best')

    ax.set_xlabel(f'FCC ({net})', fontsize=PAPER_FONT)
    ax.set_ylabel('Performance Gain (%)', fontsize=PAPER_FONT)
    ax.set_title(f'Tiny-ImageNet Split: FCC{score} vs baseline{baseline}', fontsize=PAPER_FONT)
    ax.tick_params(axis='both', which='major', labelsize=PAPER_FONT)
    _light_frame(ax)

    fig.tight_layout()
    out = f'plots/FCC{score}_ratio_baseline{baseline}_{features}.svg'
    _save_vector(fig, out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_all_tinysplit():
    """2026-11-13, user request: only FCC1 vs baselineA, one plot per network (Inception, DINO).
    plot_tinysplit() itself still supports score 2 and baseline B if they are ever needed again."""
    for features in ('inception', 'dino'):
        plot_tinysplit(1, 'A', features)


IMAGENET1K_CSV = 'tables/imagenet1k/inception2048/imagenet1k_results.csv'
# features -> ratio column in imagenet1k_results.csv
IMAGENET1K_RATIO_COLUMNS = {
    'inception': 'fcc_inception2048_ratio',
    'dino': 'fcc_dinov3vith16plus_ratio',
}

# 2026-11-10: SVHN was excluded from the plotted points. 2026-11-13, user request: SVHN is
# included again (all 7 targets). Add 'svhn' back to this set to exclude it. Note that
# imagenet1k_analysis.py still excludes SVHN from its own correlation stats (EXCLUDE_FROM_CORRELATION).
IMAGENET1K_EXCLUDE_FROM_PLOT = set()


def _load_imagenet1k():
    """Load IMAGENET1K_CSV, dropping the appended pearson_r/spearman_rho summary rows
    (recognizable by an empty "n_train_native" field) and excluding
    IMAGENET1K_EXCLUDE_FROM_PLOT targets."""
    with open(IMAGENET1K_CSV, newline='') as f:
        return [row for row in csv.DictReader(f)
                if row['n_train_native'] != '' and row['target'] not in IMAGENET1K_EXCLUDE_FROM_PLOT]


def plot_imagenet1k(features):
    """
    Scatter of FCC ratio (x) vs Performance Gain (y) for the imagenet1k (ImageNet-1K source) targets,
    one figure per network (features = 'inception' or 'dino'), with direct per-point target labels
    (few enough points that no placement algorithm is needed). SVHN is excluded from the plotted
    points (IMAGENET1K_EXCLUDE_FROM_PLOT).
    """
    import os
    from matplotlib.lines import Line2D

    rows = _load_imagenet1k()
    targets = [row['target'] for row in rows]
    gain = np.array([float(row['gain']) for row in rows])
    x = np.array([float(row[IMAGENET1K_RATIO_COLUMNS[features]]) for row in rows])
    net = FEATURE_NAMES[features]
    color = PLOT_COLOR

    _times_new_roman()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(x=x, y=gain, ax=ax, color=color,
                scatter_kws={'s': 70, 'alpha': 0.8, 'edgecolor': 'white', 'linewidths': 1.3},
                line_kws={'linewidth': 2})
    rho, rho_p = spearmanr(x, gain)
    r, r_p = pearsonr(x, gain)

    ax.margins(x=0.1, y=0.1)
    ax.autoscale_view()

    handles = [
        *_stat_handles(rho, rho_p, r, r_p),
    ]
    legend = _paper_legend(ax, handles, 'upper right')

    ax.set_xlabel(f'FCC ({net})', fontsize=PAPER_FONT)
    ax.set_ylabel('Performance Gain (%)', fontsize=PAPER_FONT)
    ax.set_title('Largescale Source Transfer Imagenet', fontsize=PAPER_FONT)
    ax.tick_params(axis='both', which='major', labelsize=PAPER_FONT)
    _light_frame(ax)

    fig.tight_layout()
    _place_pair_labels(fig, ax, legend, targets, x, gain, fontsize=PAPER_POINT_LABEL_FONT)
    os.makedirs('plots/imagenet1k', exist_ok=True)
    out = f'plots/imagenet1k/imagenet1k_fcc_ratio_{features}.svg'
    _save_vector(fig, out)
    plt.close(fig)
    print(f'Saved {out}')


def plot_all_imagenet1k():
    for features in ('inception', 'dino'):
        plot_imagenet1k(features)


AUG_DIR = 'tables/augmentation'


def _aug_load(aggregator):
    """Rows of tables/augmentation/aug_{aggregator}.csv (built by dataaugmentation/build_aug_tables.py)
    and the CIFAR-10 transfer accuracy array."""
    with open(f'{AUG_DIR}/aug_{aggregator}.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows, np.array([float(row['acc_adapt']) for row in rows])


def _aug_stats(aggregator):
    """Per-network correlation stats for one aggregator table (no plotting)."""
    rows, acc = _aug_load(aggregator)
    stats = {}
    for key in ('inception', 'dino'):
        x = np.array([float(row[key]) for row in rows])
        rho, rho_p = spearmanr(x, acc)
        r, r_p = pearsonr(x, acc)
        stats[key] = {'pearson_r': r, 'pearson_p': r_p, 'spearman_rho': rho, 'spearman_p': rho_p}
    return stats


def plot_aug(features):
    """
    Scatter of FCC ratio (x) vs CIFAR-10 transfer accuracy (y) across the augmentation-strength
    seed runs, from tables/augmentation/aug_ratio.csv; one figure per network
    (features = 'inception' or 'dino').
    """
    import os
    from matplotlib.lines import Line2D

    rows, acc = _aug_load('ratio')
    x = np.array([float(row[features]) for row in rows])
    net = FEATURE_NAMES[features]
    color = PLOT_COLOR

    _times_new_roman()

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(x=x, y=acc, ax=ax, color=color,
                scatter_kws={'s': 70, 'alpha': 0.8, 'edgecolor': 'white', 'linewidths': 1.2},
                line_kws={'linewidth': 2})
    rho, rho_p = spearmanr(x, acc)
    r, r_p = pearsonr(x, acc)

    ax.margins(x=0.07, y=0.08)
    handles = [
        *_stat_handles(rho, rho_p, r, r_p),
    ]
    _paper_legend(ax, handles, 'best')
    ax.set_xlabel(f'FCC ({net})', fontsize=PAPER_FONT)
    ax.set_ylabel('Accuracy (%)', fontsize=PAPER_FONT)
    ax.set_title('Data Augmentation Tiny-ImageNet -> CIFAR10', fontsize=PAPER_FONT)
    ax.tick_params(axis='both', which='major', labelsize=PAPER_FONT)
    _light_frame(ax)

    fig.tight_layout()
    os.makedirs('plots/augment', exist_ok=True)
    out = f'plots/augment/aug_ratio_{features}_correlation.svg'
    _save_vector(fig, out)
    plt.close(fig)
    print(f'Saved {out}')


def plot_all_aug():
    """Both ratio plots (inception, dino), plus tables/augmentation/aug_correlations.csv (still
    covers both the mean and ratio tables, so that file's contents are unchanged by this restyle)."""
    for features in ('inception', 'dino'):
        plot_aug(features)
    rows = []
    for aggregator in ('mean', 'ratio'):
        for key, s in _aug_stats(aggregator).items():
            rows.append({'table': aggregator, 'feature': key, **s})
    out = f'{AUG_DIR}/aug_correlations.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['table', 'feature', 'pearson_r', 'pearson_p',
                                               'spearman_rho', 'spearman_p'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {out}')


if __name__ == '__main__':
    plot_runtimes()
    plot_inception_correlation()
    plot_dino_correlation()
    plot_nist_pg('ratio')
    plot_nist_embedding('inception')
    plot_nist_embedding('dino')
    plot_all_tinysplit()
    plot_all_imagenet1k()
    plot_all_aug()
