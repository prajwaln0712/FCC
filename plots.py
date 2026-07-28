import json
import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_ylabel('Time (seconds)', fontsize=32)
    ax.set_title('Runtimes for Different Dataset Distance Methods', fontsize=32)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods, rotation=25, ha="right", fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.set_ylim(0, 2700)

    # Add labels to the bars
    ax.bar_label(rects, padding=3, fmt='%.0f', fontsize=24)

    fig.tight_layout()
    plt.savefig('runtimes_barplot.png')
    print("Bar plot saved to 'runtimes_barplot.png'")
    plt.show()

if __name__ == '__main__':
    plot_runtimes()