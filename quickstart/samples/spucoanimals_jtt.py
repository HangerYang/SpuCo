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

from spuco.datasets import GroupLabeledDatasetWrapper, SpuCoAnimals
from spuco.evaluate import Evaluator
from spuco.group_inference import JTTInference
from spuco.invariant_train import CustomSampleERM
from spuco.models import model_factory
from spuco.utils import Trainer, set_seed

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

testset = SpuCoAnimals(
    root=args.root_dir,
    label_noise=args.label_noise,
    split="test",
    transform=transform,
)
testset.initialize()

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
os.makedirs("models/spucoanimals", exist_ok=True)
torch.save(model.state_dict(), f"models/spucoanimals/jtt_lr={args.lr}_wd={args.weight_decay}_seed={args.seed}_upsample_factor={args.upsample_factor}.pt")


if not args.pretrained and args.infer_num_epochs < 0:
    max_f1 = 0
    logits_files = glob(f"logits/spucoanimals/lr=0.001_wd=0.0001_seed={args.seed}/valset*.pt")
    for logits_file in logits_files:
        epoch = int(logits_file.split("/")[-1].split(".")[0].split("_")[-1])
        if epoch >= args.num_epochs:
            continue
        logits = torch.load(logits_file)
        predictions = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        jtt = JTTInference(
            predictions=predictions,
            class_labels=valset.labels
        )
        epoch_group_partition = jtt.infer_groups()

        upsampled_indices = epoch_group_partition[(0,1)]
        minority_indices = valset.group_partition[(0,1)]
        minority_indices.extend(valset.group_partition[(1,0)])
        # compute F1 score on the validation set
        upsampled = np.zeros(len(predictions))
        upsampled[np.array(upsampled_indices)] = 1
        minority = np.zeros(len(predictions))
        minority[np.array(minority_indices)] = 1
        f1 = f1_score(minority, upsampled)
        if f1 > max_f1:
            max_f1 = f1
            args.infer_num_epochs = epoch
            group_partition = epoch_group_partition
            print("New best F1 score:", f1, "at epoch", epoch)


trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=args.batch_size,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
trainer.train(num_epochs=args.infer_num_epochs)

predictions = torch.argmax(trainer.get_trainset_outputs(), dim=-1).detach().cpu().tolist()
jtt = JTTInference(
    predictions=predictions,
    class_labels=trainset.labels
)

group_partition = jtt.infer_groups()

for key in sorted(group_partition.keys()):
    print(key, len(group_partition[key]))
evaluator = Evaluator(
    testset=trainset,
    group_partition=group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()

invariant_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

val_evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)

indices = []
indices.extend(group_partition[(0,0)])
indices.extend(group_partition[(0,1)] * args.upsample_factor)

print("Training on", len(indices), "samples")

model = model_factory(args.arch, trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)
jtt_train = CustomSampleERM(
    model=model,
    num_epochs=args.num_epochs,
    trainset=trainset,
    batch_size=args.batch_size,
    indices=indices,
    val_evaluator=val_evaluator,
    optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
    device=device,
    verbose=True
)
jtt_train.train()

# save the model
os.makedirs("models/spucoanimals", exist_ok=True)
torch.save(model.state_dict(), f"models/spucoanimals/jtt_lr={args.lr}_wd={args.weight_decay}_seed={args.seed}_upsample_factor={args.upsample_factor}.pt")

# save the best model
torch.save(jtt_train.best_model.state_dict(), f"models/spucoanimals/jtt_lr={args.lr}_wd={args.weight_decay}_seed={args.seed}_upsample_factor={args.upsample_factor}_best.pt")

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
results = pd.DataFrame(index=[0])
results["timestamp"] = pd.Timestamp.now()
results["seed"] = args.seed
results["pretrained"] = args.pretrained
results["lr"] = args.lr
results["weight_decay"] = args.weight_decay
results["momentum"] = args.momentum
results["num_epochs"] = args.num_epochs
results["batch_size"] = args.batch_size
results["upsample_factor"] = args.upsample_factor
results["infer_num_epochs"] = args.infer_num_epochs

results["worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["average_accuracy"] = evaluator.average_accuracy

evaluator = Evaluator(
    testset=testset,
    group_partition=testset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=args.batch_size,
    model=jtt_train.best_model,
    device=device,
    verbose=True
)
evaluator.evaluate()


results["early_stopping_worst_group_accuracy"] = evaluator.worst_group_accuracy[1]
results["early_stopping_average_accuracy"] = evaluator.average_accuracy

if os.path.exists(args.results_csv):
    results_df = pd.read_csv(args.results_csv)
else:
    results_df = pd.DataFrame()

results_df = pd.concat([results_df, results], ignore_index=True)
results_df.to_csv(args.results_csv, index=False)

print('Done!')
print('Results saved to', args.results_csv)


