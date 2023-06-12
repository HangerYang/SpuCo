import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch.optim import SGD
from wilds import get_dataset
from copy import deepcopy
from sklearn.metrics import precision_score,recall_score
from spuco.datasets import GroupLabeledDatasetWrapper, SpuCoAnimals
from spuco.evaluate import Evaluator
from spuco.group_inference import JTTInference
from spuco.invariant_train import CustomSampleERM
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed, get_group_ratios

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data")
parser.add_argument("--label_noise", type=float, default=0.0)
parser.add_argument("--results_csv", type=str, default="results/spucoanimals_jtt.csv")

parser.add_argument("--arch", type=str, default="resnet18")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_num_epochs", type=int, default=7)

parser.add_argument("--upsample_factor", type=int, default=100)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
trainset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="train",
    transform=transform,
)
trainset.initialize()
valset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="val",
    transform=transform,
)
valset.initialize()
model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
for i in range(args.infer_num_epochs):
    trainer.train(1)

    predictions = torch.argmax(trainer.get_trainset_outputs(), dim=-1).detach().cpu().tolist()
    jtt = JTTInference(
        predictions=predictions,
        class_labels=valset.labels
    )

    group_partition = jtt.infer_groups()
    
    upsampled_indices = deepcopy(group_partition[(0,1)])
    minority_indices = deepcopy(trainset.group_partition[(0,1)])
    minority_indices.extend(trainset.group_partition[(1,0)] + trainset.group_partition[(2,3)] + trainset.group_partition[(3,2)])
    # compute precision score on the validation set
    upsampled = np.zeros(len(predictions))
    upsampled[np.array(upsampled_indices)] = 1
    minority = np.zeros(len(predictions))
    minority[np.array(minority_indices)] = 1
    precision = precision_score(minority, upsampled)
    recall = recall_score(minority, upsampled)
    print(len(group_partition[(0,1)]), len(minority_indices))
    print(precision)
    print(recall)
    print(i)
    # if precision > max_precision:
    #     max_precision = precision
    #     args.infer_num_epochs = i
    #     group_partition = group_partition
    #     print("New best precision score:", precision, "at epoch", i)
    np.savez("/home/hyang/SpuCo/quickstart/samples/jtt_analyze/jtt_{}".format(i), logits=trainer.get_trainset_outputs().cpu().detach().numpy(), precision=precision, recall=recall)

    # for key in sorted(group_partition.keys()):
    #     print(key, len(group_partition[key]))
    # np.savez("jtt_analyze/epoch_{}_lr_{}_wd_{}_seed_{}".format(i, args.lr, args.weight_decay, args.seed), gp=group_partition[(0,1)], gd = get_group_ratios(group_partition[(0,1)], valset.group_partition))



