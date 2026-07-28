import torch
import numpy as np
import data
import util
from tqdm import tqdm
import time
import json

def c100_c10_inception_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances betweeen c10-classes and c-100-whole-dataset
    start_time = time.time()
    fids = []
    for class_name in tqdm(trainset.classes, desc="Calculating Inception Distance (C100 -> C10)"):
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]

        # calculate metric
        fid_score_subset = util.fid(features_cifar100, selected_features_cifar10)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (Inception) C100->C10'] = elapsed_time
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")


def c10_c100_inception_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances betweeen c100-classes and c-10-whole-dataset
    start_time = time.time()
    fids = []
    for class_name100 in tqdm(trainset100.classes, desc="Calculating Inception Distance (C10 -> C100)"):
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]

        # calculate metric
        fid_score_subset = util.fid(features_cifar10, selected_features_cifar100)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (Inception) C10->C100'] = elapsed_time
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (C10 -> C100) took {elapsed_time:.2f} seconds.")


def c100_c10_dinov3_distance(trainset, trainset100, dino_features_cifar10, dino_features_cifar100, runtimes):
    # Distances between CIFAR-10 classes and the whole CIFAR-100 dataset using DINO embeddings
    start_time = time.time()
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)

    for class_name in tqdm(trainset.classes, desc="Calculating DINOv3 Distance (C100 -> C10)"):
        # Get the DINO features for the specific CIFAR-10 class
        class_index = trainset.classes.index(class_name)
        selected_dino_c10 = dino_features_cifar10[cifar10_labels == class_index]

        # calculate metric (comparing whole C100 DINO vs specific C10 class DINO)
        fid_score = util.fid(dino_features_cifar100, selected_dino_c10)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (DINO) C100->C10'] = elapsed_time
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")


def c10_c100_dinov3_distances(trainset, trainset100, dino_features_cifar10, dino_features_cifar100, runtimes):
    # Distances between CIFAR-100 classes and the whole CIFAR-10 dataset using DINO embeddings
    start_time = time.time()
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)

    c10_whole_mean, c10_whole_cov = util.calculate_mean_covariance(dino_features_cifar10)
    for class_name100 in tqdm(trainset100.classes, desc="Calculating DINOv3 Distance (C10 -> C100)"):
        # Get the DINO features for the specific CIFAR-100 class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_dino_c100 = dino_features_cifar100[cifar_labels100 == class_index100]

        # Calculate metric (comparing whole CIFAR-10 DINO vs specific CIFAR-100 class DINO)
        c100_selected_mean, c100_selected_cov = util.calculate_mean_covariance(selected_dino_c100)
        fid_score = util.calculate_fid(c10_whole_mean, c10_whole_cov, c100_selected_mean, c100_selected_cov)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (DINO) C10->C100'] = elapsed_time
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (C10 -> C100) took {elapsed_time:.2f} seconds.")


def tiny_c10_inception_distance(trainset, features_cifar10, extracted_features, runtimes):
    # distances betweeen c10-classes and tiny-imagenet-whole-dataset
    start_time = time.time()
    fids = []
    for class_name in tqdm(trainset.classes, desc="Calculating Inception Distance (Tiny-ImageNet -> C10)"):
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]

        # calculate metric
        fid_score_subset = util.fid(extracted_features, selected_features_cifar10)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (Inception) Tiny->C10'] = elapsed_time
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (Tiny-ImageNet -> C10) took {elapsed_time:.2f} seconds.")


def tiny_c10_dinov3_distance(trainset, dino_features_cifar10, tiny_dino_extracted_features, runtimes):
    # distances between c10-classes and tiny-imagenet-whole-dataset using DINO embeddings
    start_time = time.time()
    fids_dino = []
    cifar10_labels = torch.tensor(trainset.targets)
    
    for class_name in tqdm(trainset.classes, desc="Calculating DINOv3 Distance (Tiny-ImageNet -> C10)"):
        # create a tensor with the image DINO features for selected class
        class_index = trainset.classes.index(class_name)
        selected_dino_c10 = dino_features_cifar10[cifar10_labels == class_index]

        # calculate metric
        fid_score = util.fid(tiny_dino_extracted_features, selected_dino_c10)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (DINO) Tiny->C10'] = elapsed_time
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (Tiny-ImageNet -> C10) took {elapsed_time:.2f} seconds.")


def c10_tiny_inception_distance(features_cifar10, extracted_features, tiny_metadata, runtimes):
    # distances betweeen tiny-imagenet-classes and c10-whole-dataset
    start_time = time.time()
    fids = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in tqdm(np.unique(tiny_imagenet_cls_index), desc="Calculating Inception Distance (C10 -> Tiny-ImageNet)"):
        # create a tensor with the image inception features for selected class
        selected_features_tiny = extracted_features[tiny_imagenet_cls_index == class_index]
        # calculate metric
        fid_score_subset = util.fid(features_cifar10, selected_features_tiny)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (Inception) C10->Tiny'] = elapsed_time
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (C10 -> Tiny-ImageNet) took {elapsed_time:.2f} seconds.")


def c10_tiny_dinov3_distance(dino_features_cifar10, tiny_dino_extracted_features, tiny_metadata, runtimes):
    # distances between tiny-imagenet-classes and c10-whole-dataset using DINO embeddings
    start_time = time.time()
    fids_dino = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in tqdm(np.unique(tiny_imagenet_cls_index), desc="Calculating DINOv3 Distance (C10 -> Tiny-ImageNet)"):
        # create a tensor with the image DINO features for selected class
        selected_dino_tiny = tiny_dino_extracted_features[tiny_imagenet_cls_index == class_index]
        # calculate metric
        fid_score = util.fid(dino_features_cifar10, selected_dino_tiny)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (DINO) C10->Tiny'] = elapsed_time
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (C10 -> Tiny-ImageNet) took {elapsed_time:.2f} seconds.")


def tiny_c100_inception_distance(trainset100, features_cifar100, extracted_features, runtimes):
    # distances betweeen c100-classes and tiny-imagenet-whole-dataset
    start_time = time.time()
    fids = []
    for class_name100 in tqdm(trainset100.classes, desc="Calculating Inception Distance (Tiny-ImageNet -> C100)"):
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]

        # calculate metric
        fid_score_subset = util.fid(extracted_features, selected_features_cifar100)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (Inception) Tiny->C100'] = elapsed_time
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (Tiny-ImageNet -> C100) took {elapsed_time:.2f} seconds.")


def tiny_c100_dinov3_distance(trainset100, dino_features_cifar100, tiny_dino_extracted_features, runtimes):
    # distances between c100-classes and tiny-imagenet-whole-dataset using DINO embeddings
    start_time = time.time()
    fids_dino = []
    cifar100_labels = torch.tensor(trainset100.targets)
    
    for class_name100 in tqdm(trainset100.classes, desc="Calculating DINOv3 Distance (Tiny-ImageNet -> C100)"):
        # create a tensor with the image DINO features for selected class
        class_index100 = trainset100.classes.index(class_name100)
        selected_dino_c100 = dino_features_cifar100[cifar100_labels == class_index100]

        # calculate metric
        fid_score = util.fid(tiny_dino_extracted_features, selected_dino_c100)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (DINO) Tiny->C100'] = elapsed_time
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (Tiny-ImageNet -> C100) took {elapsed_time:.2f} seconds.")


def c100_tiny_inception_distance(features_cifar100, extracted_features, tiny_metadata, runtimes):
    # distances betweeen tiny-imagenet-classes and c100-whole-dataset
    start_time = time.time()
    fids = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in tqdm(np.unique(tiny_imagenet_cls_index), desc="Calculating Inception Distance (C100 -> Tiny-ImageNet)"):
        # create a tensor with the image inception features for selected class
        selected_features_tiny = extracted_features[tiny_imagenet_cls_index == class_index]
        # calculate metric
        fid_score_subset = util.fid(features_cifar100, selected_features_tiny)
        fids.append(fid_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (Inception) C100->Tiny'] = elapsed_time
    print(f'fids mean={np.mean(fids)} max={np.max(fids)} class_coverage={np.mean(fids) / np.max(fids)}')
    print(f"Inception Distance calculation (C100 -> Tiny-ImageNet) took {elapsed_time:.2f} seconds.")


def c100_tiny_dinov3_distance(dino_features_cifar100, tiny_dino_extracted_features, tiny_metadata, runtimes):
    # distances between tiny-imagenet-classes and c100-whole-dataset using DINO embeddings
    start_time = time.time()
    fids_dino = []
    tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict = tiny_metadata
    for class_index in tqdm(np.unique(tiny_imagenet_cls_index), desc="Calculating DINOv3 Distance (C100 -> Tiny-ImageNet)"):
        # create a tensor with the image DINO features for selected class
        selected_dino_tiny = tiny_dino_extracted_features[tiny_imagenet_cls_index == class_index]
        # calculate metric
        fid_score = util.fid(dino_features_cifar100, selected_dino_tiny)
        fids_dino.append(fid_score)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['FCC (DINO) C100->Tiny'] = elapsed_time
    print(
        f'\nDINO FDDs: mean={np.mean(fids_dino):.4f} max={np.max(fids_dino):.4f} coverage={np.mean(fids_dino) / np.max(fids_dino):.4f}')
    print(f"DINOv3 Distance calculation (C100 -> Tiny-ImageNet) took {elapsed_time:.2f} seconds.")


def c10_c100_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances between c100-classes and c-10-whole-dataset
    start_time = time.time()
    otdds = []
    for class_name100 in tqdm(trainset100.classes, desc="Calculating Exact OTDD (C10 -> C100)"):
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]
        selected_labels_cifar100 = cifar_labels100[cifar_labels100 == class_index100]

        # calculate metric
        otdd_score_subset = util.exact_otdd(features_cifar10, torch.tensor(trainset.targets), selected_features_cifar100, selected_labels_cifar100)
        otdds.append(otdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['OTDD (Exact) C10->C100'] = elapsed_time
    print(f'otdds mean={np.mean(otdds)} max={np.max(otdds)} class_coverage={np.mean(otdds) / np.max(otdds)}')
    print(f"Exact OTDD calculation (C10 -> C100) took {elapsed_time:.2f} seconds.")


def c10_c100_gaussian_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances between c100-classes and c-10-whole-dataset
    start_time = time.time()
    otdds = []
    for class_name100 in tqdm(trainset100.classes, desc="Calculating Gaussian OTDD (C10 -> C100)"):
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]
        selected_labels_cifar100 = cifar_labels100[cifar_labels100 == class_index100]

        # calculate metric
        otdd_score_subset = util.gaussian_otdd(features_cifar10, torch.tensor(trainset.targets), selected_features_cifar100, selected_labels_cifar100)
        otdds.append(otdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['OTDD (Gaussian) C10->C100'] = elapsed_time
    print(f'otdds mean={np.mean(otdds)} max={np.max(otdds)} class_coverage={np.mean(otdds) / np.max(otdds)}')
    print(f"Gaussian OTDD calculation (C10 -> C100) took {elapsed_time:.2f} seconds.")


def c100_c10_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances between c10-classes and c-100-whole-dataset
    start_time = time.time()
    otdds = []
    for class_name in tqdm(trainset.classes, desc="Calculating Exact OTDD (C100 -> C10)"):
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]
        selected_labels_cifar10 = cifar_labels[cifar_labels == class_index]

        # calculate metric
        otdd_score_subset = util.exact_otdd(features_cifar100, torch.tensor(trainset100.targets), selected_features_cifar10, selected_labels_cifar10)
        otdds.append(otdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['OTDD (Exact) C100->C10'] = elapsed_time
    print(f'otdds mean={np.mean(otdds)} max={np.max(otdds)} class_coverage={np.mean(otdds) / np.max(otdds)}')
    print(f"Exact OTDD calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")


def c100_c10_gaussian_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances between c10-classes and c-100-whole-dataset
    start_time = time.time()
    otdds = []
    for class_name in tqdm(trainset.classes, desc="Calculating Gaussian OTDD (C100 -> C10)"):
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]
        selected_labels_cifar10 = cifar_labels[cifar_labels == class_index]

        # calculate metric
        otdd_score_subset = util.gaussian_otdd(features_cifar100, torch.tensor(trainset100.targets), selected_features_cifar10, selected_labels_cifar10)
        otdds.append(otdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['OTDD (Gaussian) C100->C10'] = elapsed_time
    print(f'otdds mean={np.mean(otdds)} max={np.max(otdds)} class_coverage={np.mean(otdds) / np.max(otdds)}')
    print(f"Gaussian OTDD calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")


def c10_c100_sotdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances between c100-classes and c-10-whole-dataset
    start_time = time.time()
    sotdds = []
    for class_name100 in tqdm(trainset100.classes, desc="Calculating s-OTDD (C10 -> C100)"):
        # create a tensor with the image inception features for selected class
        cifar_labels100 = torch.tensor(trainset100.targets)
        class_index100 = trainset100.classes.index(class_name100)
        selected_features_cifar100 = features_cifar100[cifar_labels100 == class_index100]
        selected_labels_cifar100 = cifar_labels100[cifar_labels100 == class_index100]

        # calculate metric
        sotdd_score_subset = util.sotdd(features_cifar10, torch.tensor(trainset.targets), selected_features_cifar100, selected_labels_cifar100)
        sotdds.append(sotdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['s-OTDD C10->C100'] = elapsed_time
    print(f's-otdds mean={np.mean(sotdds)} max={np.max(sotdds)} class_coverage={np.mean(sotdds) / np.max(sotdds)}')
    print(f"s-OTDD calculation (C10 -> C100) took {elapsed_time:.2f} seconds.")


def c100_c10_sotdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes):
    # distances between c10-classes and c-100-whole-dataset
    start_time = time.time()
    sotdds = []
    for class_name in tqdm(trainset.classes, desc="Calculating s-OTDD (C100 -> C10)"):
        # create a tensor with the image inception features for selected class
        cifar_labels = torch.tensor(trainset.targets)
        class_index = trainset.classes.index(class_name)
        selected_features_cifar10 = features_cifar10[cifar_labels == class_index]
        selected_labels_cifar10 = cifar_labels[cifar_labels == class_index]

        # calculate metric
        sotdd_score_subset = util.sotdd(features_cifar100, torch.tensor(trainset100.targets), selected_features_cifar10, selected_labels_cifar10)
        sotdds.append(sotdd_score_subset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    runtimes['s-OTDD C100->C10'] = elapsed_time
    print(f's-otdds mean={np.mean(sotdds)} max={np.max(sotdds)} class_coverage={np.mean(sotdds) / np.max(sotdds)}')
    print(f"s-OTDD calculation (C100 -> C10) took {elapsed_time:.2f} seconds.")


def main():
    runtimes = {}
    features_cifar10, features_cifar100, extracted_features = data.load_inception()
    dino_features_cifar10, dino_features_cifar100, tiny_dino_extracted_features = data.load_dinov3()
    trainset, trainset100 = data.load_cifar_pytorch_datasets()
    tiny_metadata = data.load_tinyimagenet_metadata()

    c100_c10_inception_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c10_c100_inception_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c100_c10_dinov3_distance(trainset, trainset100, dino_features_cifar10, dino_features_cifar100, runtimes)
    # c10_c100_dinov3_distances(trainset, trainset100, dino_features_cifar10, dino_features_cifar100, runtimes)
    # tiny_c10_inception_distance(trainset, features_cifar10, extracted_features, runtimes)
    # c10_tiny_inception_distance(features_cifar10, extracted_features, tiny_metadata, runtimes)
    # tiny_c10_dinov3_distance(trainset, dino_features_cifar10, tiny_dino_extracted_features, runtimes)
    # c10_tiny_dinov3_distance(dino_features_cifar10, tiny_dino_extracted_features, tiny_metadata, runtimes)
    # tiny_c100_inception_distance(trainset100, features_cifar100, extracted_features, runtimes)
    # c100_tiny_inception_distance(features_cifar100, extracted_features, tiny_metadata, runtimes)
    # tiny_c100_dinov3_distance(trainset100, dino_features_cifar100, tiny_dino_extracted_features, runtimes)
    # c100_tiny_dinov3_distance(dino_features_cifar100, tiny_dino_extracted_features, tiny_metadata, runtimes)
    # c10_c100_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c10_c100_gaussian_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c100_c10_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c100_c10_gaussian_otdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c10_c100_sotdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)
    # c100_c10_sotdd_distance(trainset, trainset100, features_cifar10, features_cifar100, runtimes)

    with open('runtimes.json', 'w') as f:
        json.dump(runtimes, f)

main()