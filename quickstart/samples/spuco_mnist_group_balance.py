import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.optim import SGD
from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
import torchvision.transforms as T
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed, get_group_ratios
from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.invariant_train import GroupBalanceBatchERM

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/spucoanimals_jtt.csv")
parser.add_argument("--difficulty", type=str, default="magnitude_easy")
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--feature_noise", type=int, default=0.1)
parser.add_argument("--label_noise", type=int, default=0.1)
parser.add_argument("--infer_num_epochs", type=int, default=7)

parser.add_argument("--upsample_factor", type=int, default=100)

args = parser.parse_args()


device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

if args.difficulty == "magnitude_easy":
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_EASY
elif args.difficulty == "magnitude_medium":
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_MEDIUM
elif args.difficulty == "magnitude_hard":
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_HARD
elif args.difficulty == "variance_easy":
    difficulty = SpuriousFeatureDifficulty.VARIANCE_EASY
elif args.difficulty == "variance_medium":
    difficulty = SpuriousFeatureDifficulty.VARIANCE_MEDIUM
elif args.difficulty == "variance_hard":
    difficulty = SpuriousFeatureDifficulty.VARIANCE_HARD


trainset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train"
)
trainset.initialize()

testset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test"
)
testset.initialize()
group_trainset = GroupLabeledDatasetWrapper(trainset, trainset.group_partition)
model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)

val_evaluator = Evaluator(
    testset=testset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
group_balance = GroupBalanceBatchERM(
    model=model,
    num_epochs=num_epochs,
    trainset=group_trainset,
    batch_size=batch_size,
    optimizer=SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay),
    device=device,
    verbose=True
)
group_balance.train()
model = group_dro.best_model

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()