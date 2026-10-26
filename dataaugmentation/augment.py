import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import data
import util
from data import load_imagenet
from models.resnet import ResNet50
from models.trainer import train, test_func, frozen_module
from nist.nist_extract_embeddings import build_network, resize
import os
import random
import time
from torch.utils.data import DataLoader
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use CUDA or not: {DEVICE}")

FCC_RIDGE = 1e-6
FCC_MAX_IMAG = 1e-3


def main():
    parser = argparse.ArgumentParser(description='Arguments for Augmentation')
    parser.add_argument('--parent_dir', type=str, default="dataaugmentation", help='Parent directory')
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    parser.add_argument('--epochs-pretrain', type=int, default=300, help='ImageNet backbone pretraining epochs')
    parser.add_argument('--epochs-adapt', type=int, default=30, help='CIFAR10 head adaptation epochs')
    parser.add_argument('--inception-weights', choices=['fid', 'torchvision'], default='fid',
                        help='Inception-v3 weights used for FCC feature extraction')
    parser.add_argument('--dino-model', default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                        help='DINOv3 model id used for FCC feature extraction')

    args = parser.parse_args()

    # Seed every RNG this run touches so a given --seed is fully reproducible end to end.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parent_dir = f"{args.parent_dir}/aug_{str(args.seed)}"
    os.makedirs(parent_dir, exist_ok=True)
    result_file = f"{parent_dir}/result.txt"

    _t_run_start = time.time()

    def create_data():
        brightness = random.uniform(0.1, 0.9)
        contrast = random.uniform(0.1, 0.9)
        saturation = random.uniform(0.1, 0.9)
        hue = random.uniform(0, 0.5)
        print(f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue}")
        with open(result_file, 'a') as file:
            file.write(
                f"Random: brightness: {brightness}, contrast: {contrast}, saturation: {saturation}, hue: {hue} \n")
        datadir_tiny_imagenet = "data/tiny-ImageNet/tiny-imagenet-200"
        imagenet = load_imagenet(datadir=datadir_tiny_imagenet,
                                 resize=32,
                                 tiny=True,
                                 augmentations=True,
                                 brightness=brightness,
                                 contrast=contrast,
                                 saturation=saturation,
                                 maxsize=None,
                                 hue=hue,
                                 seed=args.seed)
        imagenet_trainset, imagenet_testset = imagenet
        imagenet_trainloader = DataLoader(imagenet_trainset, batch_size=64, shuffle=True)
        imagenet_testloader = DataLoader(imagenet_testset, batch_size=64, shuffle=False)

        datadir_cifar10 = "data/CIFAR10"
        cifar10_trainset, cifar10_testset = data.load_cifar_pytorch_datasets(
            datadir=datadir_cifar10, download=False, include_cifar100=False)
        cifar10_trainloader = DataLoader(cifar10_trainset, batch_size=64, shuffle=True)
        cifar10_testloader = DataLoader(cifar10_testset, batch_size=64, shuffle=False)

        def save_data(data_set, saved_tensor_path):
            list_images = list()
            list_labels = list()
            for img, label in data_set:
                list_images.append(img)
                list_labels.append(label)
            tensor_images = torch.stack(list_images)
            tensor_labels = torch.tensor(list_labels)
            torch.save((tensor_images, tensor_labels), saved_tensor_path)
            print(f"Number of data: {len(list_images)}, Save data into {saved_tensor_path}")

        save_data(data_set=imagenet_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_imagenet.pt")
        save_data(data_set=imagenet_testset, saved_tensor_path=f"{parent_dir}/transformed_test_imagenet.pt")
        save_data(data_set=cifar10_trainset, saved_tensor_path=f"{parent_dir}/transformed_train_cifar10.pt")
        save_data(data_set=cifar10_testset, saved_tensor_path=f"{parent_dir}/transformed_test_cifar10.pt")

        return {
            "augmentation": {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "hue": hue,
            },
            "cifar10": {
                "trainset": cifar10_trainset,
                "testset": cifar10_testset,
                "trainloader": cifar10_trainloader,
                "testloader": cifar10_testloader
            },
            "imagenet": {
                "trainset": imagenet_trainset,
                "testset": imagenet_testset,
                "trainloader": imagenet_trainloader,
                "testloader": imagenet_testloader
            }
        }

    # Train model, retrieve accuracy
    def transfer_learning(train_imagenet_loader, test_imagenet_loader, train_cifar10_loader, test_cifar10_loader,
                          num_epochs_pretrain=300, num_epochs_adapt=30, device=DEVICE):
        print("Training backbone on Tiny-ImageNet...")
        # Pretrain backbone on Tiny-ImageNet
        imagenet_feature_extractor = ResNet50().to(device)
        imagenet_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 200).to(device)
        feature_extractor_optimizer = optim.SGD(imagenet_feature_extractor.parameters(), lr=0.1, momentum=0.9,
                                                weight_decay=1e-4)
        classifier_optimizer = optim.SGD(imagenet_classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        _t_pretrain_start = time.time()
        for epoch in range(1, num_epochs_pretrain + 1):
            train_loss = train(feature_extractor=imagenet_feature_extractor,
                               classifier=imagenet_classifier,
                               device=device,
                               train_loader=train_imagenet_loader,
                               epoch=epoch,
                               criterion=nn.CrossEntropyLoss(),
                               ft_extractor_optimizer=feature_extractor_optimizer,
                               classifier_optimizer=classifier_optimizer)
            print(f"In training backbone on Tiny-ImageNet, epoch {epoch}, training loss: {train_loss}")
        _t_pretrain_elapsed = time.time() - _t_pretrain_start
        print(f"Tiny-ImageNet pretraining ({num_epochs_pretrain} epochs) took {_t_pretrain_elapsed:.1f}s "
              f"({_t_pretrain_elapsed / num_epochs_pretrain:.2f}s/epoch)")
        frozen_module(imagenet_feature_extractor)

        ft_extractor_path = f'{parent_dir}/imagenet_ft_extractor.pth'
        torch.save(imagenet_feature_extractor.state_dict(), ft_extractor_path)

        classifier_path = f'{parent_dir}/imagenet_classifier.pth'
        torch.save(imagenet_classifier.state_dict(), classifier_path)

        imagenet_acc_no_adapt = test_func(feature_extractor=imagenet_feature_extractor, classifier=imagenet_classifier,
                                          device=device, test_loader=test_imagenet_loader)

        print("Training transfer learning in CIFAR10...")
        # Transfer learning on CIFAR10
        cifar10_classifier = nn.Linear(imagenet_feature_extractor.latent_dims, 10).to(device)
        cifar10_classifier_optimizer = optim.SGD(cifar10_classifier.parameters(), lr=0.01, momentum=0.9,
                                                 weight_decay=1e-4)
        _t_adapt_start = time.time()
        for epoch in range(1, num_epochs_adapt + 1):
            train_loss = train(feature_extractor=imagenet_feature_extractor,
                               classifier=cifar10_classifier,
                               device=device,
                               train_loader=train_cifar10_loader,
                               epoch=epoch,
                               criterion=nn.CrossEntropyLoss(),
                               ft_extractor_optimizer=None,
                               classifier_optimizer=cifar10_classifier_optimizer)
            print(f"In training transfer learning in CIFAR10, epoch {epoch}, training loss: {train_loss}")
        _t_adapt_elapsed = time.time() - _t_adapt_start
        print(f"CIFAR10 adaptation ({num_epochs_adapt} epochs) took {_t_adapt_elapsed:.1f}s "
              f"({_t_adapt_elapsed / num_epochs_adapt:.2f}s/epoch)")

        classifier_path = f'{parent_dir}/cifar10_classifier.pth'
        torch.save(cifar10_classifier.state_dict(), classifier_path)

        cifar10_acc_adapt = test_func(feature_extractor=imagenet_feature_extractor, classifier=cifar10_classifier,
                                      device=device, test_loader=test_cifar10_loader)
        print(f"Accuracy of CIFAR10: {cifar10_acc_adapt}")
        return cifar10_acc_adapt

    print("Start creating data")
    _t_data_start = time.time()
    DATA_DICT = create_data()
    _t_data_elapsed = time.time() - _t_data_start
    print(f"Finish creating data (took {_t_data_elapsed:.1f}s)")

    imagenet_trainloader = DATA_DICT["imagenet"]["trainloader"]
    cifar10_trainloader = DATA_DICT["cifar10"]["trainloader"]
    imagenet_testloader = DATA_DICT["imagenet"]["testloader"]
    cifar10_testloader = DATA_DICT["cifar10"]["testloader"]

    cifar10_acc_adapt = transfer_learning(train_imagenet_loader=imagenet_trainloader,
                                          test_imagenet_loader=imagenet_testloader,
                                          train_cifar10_loader=cifar10_trainloader,
                                          test_cifar10_loader=cifar10_testloader,
                                          num_epochs_pretrain=args.epochs_pretrain,
                                          num_epochs_adapt=args.epochs_adapt,
                                          device=DEVICE)

    # --- FCC via Inception + DINOv3 embeddings: TinyImageNet (source) -> CIFAR10 (target) ---
    print("Computing FCC (Inception + DINOv3 embeddings): TinyImageNet (source) -> CIFAR10 (target)...")
    _t_fcc_start = time.time()

    imagenet_trainset = DATA_DICT["imagenet"]["trainset"]
    cifar10_trainset = DATA_DICT["cifar10"]["trainset"]

    imagenet_mean, imagenet_std = data.DATASET_NORMALIZATION['ImageNet']
    imagenet_mean_t = torch.tensor(imagenet_mean, device=DEVICE).view(1, 3, 1, 1)
    imagenet_std_t = torch.tensor(imagenet_std, device=DEVICE).view(1, 3, 1, 1)

    def embed_dataset(dataset, net, denormalize, batch_size=64):
        # load_imagenet bakes in ImageNet Normalize; load_cifar_pytorch_datasets leaves images in [0, 1].
        # Undo the former so both datasets reach the network's own preprocessing as plain [0, 1] images.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        feats, labels = [], []
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(DEVICE)
                if denormalize:
                    x = (x * imagenet_std_t + imagenet_mean_t).clamp(0, 1)
                x = resize(x, net['size'])
                mean = torch.tensor(net['mean'], device=DEVICE).view(1, 3, 1, 1)
                std = torch.tensor(net['std'], device=DEVICE).view(1, 3, 1, 1)
                x = (x - mean) / std
                feats.append(net['forward'](x).float().cpu())
                labels.append(y)
        return torch.cat(feats).double(), torch.cat(labels)

    # Fixed, seed-derived offsets (arbitrary, just distinct) -- not args.seed itself, so FCC's
    # own random draws (TinyImageNet augmentation during embed_dataset, DataLoader worker
    # seeding) depend only on args.seed and which network is being processed, never on how
    # much upstream training happened (epoch count, batch size, AMP on/off) or on whether
    # inception or dino ran first. Reproducible standalone from the same data, independent of
    # the rest of the run.
    FCC_SEED_OFFSET = {'inception': 10_000_001, 'dino': 10_000_002}

    fcc_results = {}
    for network in ['inception', 'dino']:
        fcc_seed = args.seed + FCC_SEED_OFFSET[network]
        random.seed(fcc_seed)
        np.random.seed(fcc_seed)
        torch.manual_seed(fcc_seed)
        torch.cuda.manual_seed_all(fcc_seed)

        print(f"  extracting {network} embeddings...")
        net = build_network(network, args, DEVICE)
        src_X, _ = embed_dataset(imagenet_trainset, net, denormalize=True)
        tgt_X, tgt_y = embed_dataset(cifar10_trainset, net, denormalize=False)

        src_pooled = util.calculate_mean_covariance(src_X)
        n_classes = int(tgt_y.max().item()) + 1
        tgt_per_class = [util.calculate_mean_covariance(tgt_X[tgt_y == c]) for c in range(n_classes)]
        per_class_fcc = np.array([
            util.calculate_fid(src_pooled[0], src_pooled[1], mu_c, sigma_c, ridge=FCC_RIDGE, max_imag=FCC_MAX_IMAG)
            for mu_c, sigma_c in tgt_per_class
        ])
        fcc_results[network] = dict(mean=float(per_class_fcc.mean()), ratio=float(per_class_fcc.mean() / per_class_fcc.max()))
        print(f"  FCC[{network}](TinyImageNet -> CIFAR10): mean={fcc_results[network]['mean']:.4f} "
              f"ratio={fcc_results[network]['ratio']:.4f}")
        del net
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    _t_fcc_elapsed = time.time() - _t_fcc_start
    print(f"FCC computation (both networks) took {_t_fcc_elapsed:.1f}s")

    results = {
        "seed": args.seed,
        "epochs_pretrain": args.epochs_pretrain,
        "epochs_adapt": args.epochs_adapt,
        "augmentation": DATA_DICT["augmentation"],
        "cifar10_acc_adapt": cifar10_acc_adapt,
        "fcc": fcc_results,
    }
    with open(f'{parent_dir}/results.json', 'w') as file:
        json.dump(results, file, indent=2)
    print(f"Saved {parent_dir}/results.json")
    print(f"Total run time: {time.time() - _t_run_start:.1f}s")


if __name__ == "__main__":
    main()
