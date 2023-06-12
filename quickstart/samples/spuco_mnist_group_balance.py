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
from spuco.evaluate import Evaluator
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--diff", type=str, default="magnitude_easy")
parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--feature_noise", type=float, default=0)

args = parser.parse_args()


device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

if args.diff == "easy":
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_EASY
elif args.diff == "medium":
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_MEDIUM
elif args.diff == "hard":
    difficulty = SpuriousFeatureDifficulty.MAGNITUDE_HARD
elif args.diff == "easyv":
    difficulty = SpuriousFeatureDifficulty.VARIANCE_EASY
elif args.diff == "mediumv":
    difficulty = SpuriousFeatureDifficulty.VARIANCE_MEDIUM
elif args.diff == "hardv":
    difficulty = SpuriousFeatureDifficulty.VARIANCE_HARD


trainset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    spurious_correlation_strength=0.995,
    core_feature_noise=args.feature_noise,
    label_noise=args.label_noise,
    classes=classes,
    split="train"
)
trainset.initialize()
valset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="val"
)
valset.initialize()
testset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="test"
)
testset.initialize()
model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)

val_evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
group_balance = GroupBalanceBatchERM(
    model=model,
    val_evaluator=val_evaluator, 
    num_epochs=args.num_epochs,
    group_partition = trainset.group_partition,
    trainset=trainset,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay),
    device=device,
    verbose=True
)
group_balance.train()
model = group_balance.best_model

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
np.savez("group_balance/group_balance_diff_{}_seed_{}_feat_{}_label_{}".format(args.diff, args.seed, args.feature_noise, args.label_noise), worst_acc=evaluator.worst_group_accuracy[1], ave_acc=evaluator.average_accuracy)