import torch
import numpy as np
import data
import util
from tqdm import tqdm
import time
import json
import os

def load_distances_cache():
    if os.path.exists('distances_cache.npz'):
        return dict(np.load('distances_cache.npz'))
    return {}

def save_distances_cache(cache):
    np.savez_compressed('distances_cache.npz', **cache)

def c100_c10_inception_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes, distances_cache):
    cache_key = 'FCC (Inception) C100->C10'
    if cache_key in distances_cache:
        print(f"Loading cached distances for {cache_key}")
        return distances_cache[cache_key]

    start_time = time.time()
    fids = []
    for class_name in tqdm(trainset.classes, desc="Calculating Inception Distance (C100 -> C10)"):
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]
        fid_score_subset = util.fid(features_cifar100, selected_features_cifar10)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes[cache_key] = elapsed_time
    distances_cache[cache_key] = fids
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")
    return fids

def c100_c10_dinov3_distance(trainset, trainset100, dino_features_cifar10, dino_features_cifar100, runtimes, distances_cache):
    cache_key = 'FCC (DINO) C100->C10'
    if cache_key in distances_cache:
        print(f"Loading cached distances for {cache_key}")
        return distances_cache[cache_key]

    start_time = time.time()
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)
    for class_name in tqdm(trainset.classes, desc="Calculating DINOv3 Distance (C100 -> C10)"):
        class_index = trainset.classes.index(class_name)
        selected_dino_c10 = dino_features_cifar10[cifar10_labels == class_index]
        fid_score = util.fid(dino_features_cifar100, selected_dino_c10)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes[cache_key] = elapsed_time
    distances_cache[cache_key] = fids_dino
    print(f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")
    return fids_dino

def c100_c10_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes, distances_cache):
    cache_key = 'OTDD (Exact) C100->C10'
    if cache_key in distances_cache:
        print(f"Loading cached distances for {cache_key}")
        return distances_cache[cache_key]

    start_time = time.time()
    otdds = []
    for class_name in tqdm(trainset.classes, desc="Calculating Exact OTDD (C100 -> C10)"):
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]
        selected_labels_cifar10 = cifar_labels[cifar_labels == class_index]
        otdd_score_subset = util.exact_otdd(features_cifar100, torch.tensor(trainset100.targets), selected_features_cifar10, selected_labels_cifar10)
        otdds.append(otdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes[cache_key] = elapsed_time
    distances_cache[cache_key] = otdds
    print(f'otdds mean={np.mean(otdds)} max={np.max(otdds)} class_coverage={np.mean(otdds) / np.max(otdds)}')
    print(f"Exact OTDD calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")
    return otdds

def c100_c10_sotdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes, distances_cache):
    cache_key = 's-OTDD C100->C10'
    if cache_key in distances_cache:
        print(f"Loading cached distances for {cache_key}")
        return distances_cache[cache_key]

    start_time = time.time()
    sotdds = []
    for class_name in tqdm(trainset.classes, desc="Calculating s-OTDD (C100 -> C10)"):
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]
        selected_labels_cifar10 = cifar_labels[cifar_labels == class_index]
        sotdd_score_subset = util.sotdd(features_cifar100, torch.tensor(trainset100.targets), selected_features_cifar10, selected_labels_cifar10)
        sotdds.append(sotdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes[cache_key] = elapsed_time
    distances_cache[cache_key] = sotdds
    print(f's-otdds mean={np.mean(sotdds)} max={np.max(sotdds)} class_coverage={np.mean(sotdds) / np.max(sotdds)}')
    print(f"s-OTDD calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")
    return sotdds

def calculate_and_print_correlations(distances_cache):
    otdd_exact = distances_cache.get('OTDD (Exact) C100->C10')
    fcc_inception = distances_cache.get('FCC (Inception) C100->C10')
    fcc_dino = distances_cache.get('FCC (DINO) C100->C10')
    s_otdd = distances_cache.get('s-OTDD C100->C10')

    if otdd_exact is not None and fcc_inception is not None:
        correlation = np.corrcoef(otdd_exact, fcc_inception)[0, 1]
        print(f"Correlation between OTDD (Exact) and FCC (Inception): {correlation:.4f}")

    if otdd_exact is not None and fcc_dino is not None:
        correlation = np.corrcoef(otdd_exact, fcc_dino)[0, 1]
        print(f"Correlation between OTDD (Exact) and FCC (DINO): {correlation:.4f}")

    if otdd_exact is not None and s_otdd is not None:
        correlation = np.corrcoef(otdd_exact, s_otdd)[0, 1]
        print(f"Correlation between OTDD (Exact) and s-OTDD: {correlation:.4f}")

def main():
    runtimes = {}
    distances_cache = load_distances_cache()

    features_cifar10, features_cifar100, _ = data.load_inception()
    dino_features_cifar10, dino_features_cifar100, _ = data.load_dinov3()
    trainset, trainset100 = data.load_cifar_pytorch_datasets()

    c100_c10_inception_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes, distances_cache)
    c100_c10_dinov3_distance(trainset, trainset100, dino_features_cifar10, dino_features_cifar100, runtimes, distances_cache)
    c100_c10_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes, distances_cache)
    c100_c10_sotdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes, distances_cache)

    save_distances_cache(distances_cache)
    with open('runtimes.json', 'w') as f:
        json.dump(runtimes, f)

    calculate_and_print_correlations(distances_cache)

if __name__ == '__main__':
    main()