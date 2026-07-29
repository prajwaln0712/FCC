import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

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
    plt.savefig('runtimes_barplot.png')
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
    plt.savefig('correlation_inception_plot.png')
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
    plt.savefig('correlation_dino_plot.png')
    print("Correlation plot saved to 'correlation_dino_plot.png'")
    plt.show()

if __name__ == '__main__':
    plot_runtimes()
    plot_inception_correlation()
    plot_dino_correlation()