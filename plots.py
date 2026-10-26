import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


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
    plt.savefig('plots/runtimes_barplot.png')
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

NIST_PLOT_TITLE = 'Dataset vs Adaptation: NIST Datasets'
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

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    color = '#2a6fbb'
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.regplot(x=fcc, y=pg, ax=ax, color=color,
                scatter_kws={'s': 70, 'alpha': 0.9, 'edgecolor': 'white', 'linewidths': 1.5},
                line_kws={'linewidth': 2})

    ax.margins(x=0.07, y=0.08)
    ax.autoscale_view()

    handles = [
        Line2D([], [], marker='o', linestyle='', color=color, markersize=9, label='Directed pair (source→target)'),
        Line2D([], [], color=color, linewidth=2, label='Linear fit'),
        Patch(facecolor=color, alpha=0.25, label='95% confidence band'),
        Line2D([], [], linestyle='', label=f'Spearman $\\rho$ = {stats["spearman"]:.2f} (p = {stats["spearman_p"]:.3f})'),
        Line2D([], [], linestyle='', label=f'Pearson $r$ = {stats["pearson"]:.2f} (p = {stats["pearson_p"]:.3f})'),
    ]
    legend = ax.legend(handles=handles, loc='upper left', fontsize=14, framealpha=0.9)

    ax.set_xlabel(NIST_AGGREGATOR_LABELS[aggregator], fontsize=18)
    ax.set_ylabel('Performance gap', fontsize=18)
    ax.set_title(NIST_PLOT_TITLE, fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _place_pair_labels(fig, ax, legend, [_pair_initials(p) for p in pairs], fcc, pg)
    out = f'nist_fcc_{aggregator}_plot.png'
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"NIST plot saved to '{out}'")


if __name__ == '__main__':
    plot_runtimes()
    plot_inception_correlation()
    plot_dino_correlation()
    plot_nist_pg('mean')
    plot_nist_pg('ratio')