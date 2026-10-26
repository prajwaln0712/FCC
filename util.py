# TODO(license): exact_otdd / gaussian_otdd / sotdd below are said to be inherited from
# MIT-licensed OTDD code (possibly github.com/microsoft/otdd — source URL unconfirmed).
# Before this is shared or published: confirm the exact upstream repo/commit and copyright
# holder, decide scope (just these functions vs. whole file vs. whole repo), add the
# upstream MIT license text (e.g. a THIRD_PARTY_NOTICES.md), and add a proper attribution
# header above the relevant code. See conversation on 2026-10-12.

import numpy as np
import torch
from scipy.linalg import sqrtm
import ot


# Function to calculate mean and covariance of features
def calculate_mean_covariance(features):
    # Calculate mean
    mean = torch.mean(features, dim=0).numpy()
    # Calculate covariance
    # Note: torch.cov requires a tensor of shape (n_features, n_samples)
    # We need to transpose the features tensor for this.
    # Also, it's recommended to move the tensor to CPU if it's on GPU
    # for covariance calculation if memory is a concern.
    cov = torch.cov(features.T).numpy()
    return mean, cov


# Function to calculate the Fréchet Inception Distance (FID)
def calculate_fid(mu1, sigma1, mu2, sigma2, ridge=0.0, max_imag=None):
    # Calculate the squared difference between the means
    diff = mu1 - mu2
    sum_sq_diff = np.sum(diff**2)

    # Ridge regularisation for rank-deficient covariances (e.g. raw pixels)
    if ridge > 0:
        eye = np.eye(sigma1.shape[0], dtype=sigma1.dtype)
        sigma1 = sigma1 + ridge * eye
        sigma2 = sigma2 + ridge * eye

    # Calculate the square root of the product of covariances
    # This requires handling potential numerical instability
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check for numerical issues and correct if necessary
    if np.iscomplexobj(covmean):
        imag_max = np.abs(covmean.imag).max()
        if max_imag is not None and imag_max > max_imag:
            raise ValueError(f'sqrtm produced imaginary component {imag_max:.3e} > {max_imag:.1e}')
        covmean = covmean.real

    # Calculate the trace of the sum of covariances minus twice the square root of their product
    fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


# Wrapper function for FID calculation
def fid(images1, images2):
    # Calculate mean and covariance for 'cat' features
    mean_cat, cov_cat = calculate_mean_covariance(images1)
    # print("Mean and covariance calculated for 'cat' features.")
    # print(f"Mean shape: {mean_cat.shape}")
    # print(f"Covariance shape: {cov_cat.shape}")
    # Calculate mean and covariance for 'fox' features
    mean_fox, cov_fox = calculate_mean_covariance(images2)
    # print("Mean and covariance calculated for 'fox' features.")
    # print(f"Mean shape: {mean_fox.shape}")
    # print(f"Covariance shape: {cov_fox.shape}")
    return calculate_fid(mean_cat, cov_cat, mean_fox, cov_fox)


def _label_distance_matrix(labels1, labels2):
    """
    Computes a matrix of pairwise label distances.
    Distance is 0 if labels are the same, 1 otherwise.
    """
    return (labels1.unsqueeze(1) != labels2.unsqueeze(0)).float()


def exact_otdd(X1, y1, X2, y2, class_dist_scale=1.0, reg=0.001):
    """
    Computes the exact Optimal Transport Dataset Distance (OTDD).

    Args:
        X1, X2: Feature tensors for the two datasets.
        y1, y2: Label tensors for the two datasets.
        class_dist_scale: Weight for the label distance.

    Returns:
        The OTDD value.
    """
    n1, n2 = len(X1), len(X2)
    a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2

    # Intra-dataset cost matrices
    dist_X1 = torch.cdist(X1, X1)
    dist_y1 = _label_distance_matrix(y1, y1)
    cost_1 = (class_dist_scale * dist_y1 + dist_X1).numpy()

    dist_X2 = torch.cdist(X2, X2)
    dist_y2 = _label_distance_matrix(y2, y2)
    cost_2 = (class_dist_scale * dist_y2 + dist_X2).numpy()

    # Compute potentials using Sinkhorn algorithm
    # We are transporting the dataset to itself, so a=b
    uniform_dist1 = np.ones(n1) / n1
    _, log1 = ot.sinkhorn(uniform_dist1, uniform_dist1, cost_1, reg=reg, log=True)
    potentials1 = log1['u']

    uniform_dist2 = np.ones(n2) / n2
    _, log2 = ot.sinkhorn(uniform_dist2, uniform_dist2, cost_2, reg=reg, log=True)
    potentials2 = log2['u']

    # Inter-dataset cost matrix based on potentials
    # The cost is the squared L2 distance between potentials.
    # Here potentials are scalars, so it's just squared difference.
    cost_inter = (potentials1[:, np.newaxis] - potentials2[np.newaxis, :])**2

    # Final OTDD calculation using EMD
    otdd_dist = ot.emd2(a, b, cost_inter)

    return otdd_dist


def gaussian_otdd(X1, y1, X2, y2, class_dist_scale=1.0, reg=0.1):
    """
    Computes the Gaussian-approximated Optimal Transport Dataset Distance (OTDD).
    """
    n1, n2 = len(X1), len(X2)

    # Intra-dataset cost matrices
    dist_X1 = torch.cdist(X1, X1)
    dist_y1 = _label_distance_matrix(y1, y1)
    cost_1 = (class_dist_scale * dist_y1 + dist_X1).numpy()

    dist_X2 = torch.cdist(X2, X2)
    dist_y2 = _label_distance_matrix(y2, y2)
    cost_2 = (class_dist_scale * dist_y2 + dist_X2).numpy()

    # Compute potentials using Sinkhorn algorithm
    uniform_dist1 = np.ones(n1) / n1
    _, log1 = ot.sinkhorn(uniform_dist1, uniform_dist1, cost_1, reg=reg, log=True)
    potentials1 = log1['u']

    uniform_dist2 = np.ones(n2) / n2
    _, log2 = ot.sinkhorn(uniform_dist2, uniform_dist2, cost_2, reg=reg, log=True)
    potentials2 = log2['u']

    # Approximate distributions of potentials with Gaussians
    mean1, std1 = np.mean(potentials1), np.std(potentials1)
    mean2, std2 = np.mean(potentials2), np.std(potentials2)

    # Compute 2-Wasserstein distance between the two 1D Gaussians
    otdd_dist = (mean1 - mean2)**2 + (std1 - std2)**2

    return otdd_dist


def sotdd(X1, y1, X2, y2, n_projections=100, class_dist_scale=1.0):
    """
    Computes the Sliced-Wasserstein Optimal Transport Dataset Distance (s-OTDD).
    """
    n1, d = X1.shape
    n2 = X2.shape[0]

    # Generate random projections
    projections = torch.randn(d, n_projections, device=X1.device)
    projections /= torch.linalg.norm(projections, dim=0, keepdim=True)

    # Project features
    X1_proj = X1 @ projections
    X2_proj = X2 @ projections

    # Compute potentials for each dataset
    potentials1 = torch.zeros(n1, device=X1.device)
    for i in range(n_projections):
        # Sort projected data and labels
        sorted_indices1 = torch.argsort(X1_proj[:, i])
        y1_sorted = y1[sorted_indices1]

        # Compute label distance matrix for the sorted labels
        dist_y1 = _label_distance_matrix(y1_sorted, y1_sorted)

        # Compute potentials for this projection
        # This is a simplified version of the potential calculation in the paper
        # where we consider the cost as the label dissimilarity.
        # The potential of a point is the average cost to all other points.
        potentials1 += dist_y1.mean(dim=1) * class_dist_scale

    potentials2 = torch.zeros(n2, device=X2.device)
    for i in range(n_projections):
        sorted_indices2 = torch.argsort(X2_proj[:, i])
        y2_sorted = y2[sorted_indices2]
        dist_y2 = _label_distance_matrix(y2_sorted, y2_sorted)
        potentials2 += dist_y2.mean(dim=1) * class_dist_scale

    # Normalize potentials
    potentials1 /= n_projections
    potentials2 /= n_projections

    # Compute 1D Wasserstein distance between the potentials
    # This is equivalent to sorting both and computing the L1 distance
    potentials1_sorted = torch.sort(potentials1).values
    potentials2_sorted = torch.sort(potentials2).values

    # To compare distributions of different sizes, we need to resample
    # the smaller one to match the size of the larger one.
    if n1 > n2:
        potentials2_sorted = torch.from_numpy(np.interp(np.linspace(0, n2-1, n1), np.arange(n2), potentials2_sorted.cpu().numpy())).to(X1.device)
    elif n2 > n1:
        potentials1_sorted = torch.from_numpy(np.interp(np.linspace(0, n1-1, n2), np.arange(n1), potentials1_sorted.cpu().numpy())).to(X1.device)

    sotdd_dist = torch.abs(potentials1_sorted - potentials2_sorted).mean()

    return sotdd_dist.item()


def main():
    print('FID and OTDD routines loaded')


# main()