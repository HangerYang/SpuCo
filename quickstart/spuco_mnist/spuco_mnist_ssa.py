from datetime import datetime
import argparse
import os
import sys
import wandb

import pandas as pd
import torch
from torch.optim import SGD

from spuco.datasets import (GroupLabeledDatasetWrapper, SpuCoMNIST,
                            SpuriousFeatureDifficulty,
                            SpuriousTargetDatasetWrapper)
from spuco.evaluate import Evaluator
from spuco.group_inference import SSA
from spuco.robust_train import GroupDRO
from spuco.models import model_factory
from spuco.utils import set_seed

class EnumAction(argparse.Action):
    def __init__(self, **kwargs):
        # We expect the enum type to be passed with the 'choices' keyword
        enum_type = kwargs.pop("type", None)
        kwargs["choices"] = [e.name for e in enum_type]
        super().__init__(**kwargs)
        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert the input string to the corresponding enum member
        setattr(namespace, self.dest, self._enum[values])

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--root_dir", type=str, default="/data/mnist/")
parser.add_argument("--results_csv", type=str, default="spuco_mnist_ssa.csv")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--pretrained", action="store_true")

parser.add_argument("--infer_lr", type=float, default=1e-3)
parser.add_argument("--infer_weight_decay", type=float, default=1e-5)
parser.add_argument("--infer_momentum", type=float, default=0.9)
parser.add_argument("--infer_num_iters", type=int, default=500)
parser.add_argument("--infer_val_frac", type=float, default=0.5)

parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="spuco")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default="spuco_sun_ssa")
parser.add_argument('--difficulty', action=EnumAction, type=SpuriousFeatureDifficulty, help='Choose a difficulty')
args = parser.parse_args()

if args.wandb:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=args)
    # remove the stdout_file argument
    del args.stdout_file
    del args.results_csv
else:
    # check if the stdout file already exists, and if want to overwrite it
    DT_STRING = "".join(str(datetime.now()).split())
    args.stdout_file = f"{DT_STRING}-{args.stdout_file}"
    if os.path.exists(args.stdout_file):
        print(f"stdout file {args.stdout_file} already exists, overwrite? (y/n)")
        response = input()
        if response != "y":
            sys.exit()
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    # redirect stdout to a file
    sys.stdout = open(args.stdout_file, "w")

print(args)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Load the full dataset, and download it if necessary
classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

trainset = SpuCoMNIST(
    root=args.root_dir,
    spurious_feature_difficulty=args.difficulty,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train"
)
trainset.initialize()

testset = SpuCoMNIST(
    root=args.root_dir,
    spurious_feature_difficulty=args.difficulty,
    classes=classes,
    split="test"
)
testset.initialize()

valset = SpuCoMNIST(
    root=args.root_dir,
    spurious_feature_difficulty=args.difficulty,
    classes=classes,
    split="val"
)
valset.initialize()

model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes, pretrained=args.pretrained).to(device)

ssa = SSA(
    spurious_unlabeled_dataset=trainset,
    spurious_labeled_dataset=SpuriousTargetDatasetWrapper(valset, valset.spurious),
    model=model,
    labeled_valset_size=args.infer_val_frac,
    lr=args.infer_lr,
    weight_decay=args.infer_weight_decay,
    num_iters=args.infer_num_iters,
    tau_g_min=0.95,
    num_splits=3,
    device=device,
    verbose=True
)

group_partition = ssa.infer_groups()
for key in sorted(group_partition.keys()):
    print(key, len(group_partition[key]))
# evaluator = Evaluator(
#     testset=trainset,
#     group_partition=group_partition,
#     group_weights=trainset.group_weights,
#     batch_size=args.batch_size,
#     model=model,
#     device=device,
#     verbose=True
# )
# evaluator.evaluate()

# robust_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)

# valid_evaluator = Evaluator(
#     testset=valset,
#     group_partition=valset.group_partition,
#     group_weights=trainset.group_weights,
#     batch_size=64,
#     model=model,
#     device=device,
#     verbose=True
# )
# group_dro = GroupDRO(
#     model=model,
#     val_evaluator=valid_evaluator,
#     num_epochs=args.num_epochs,
#     trainset=robust_trainset,
#     batch_size=args.batch_size,
#     optimizer=SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum),
#     device=device,
#     verbose=True
# )
# group_dro.train()

results = pd.DataFrame(index=[0])
evaluator = Evaluator(
    testset=valset,
    group_partition=valset.group_partition,
    group_weights=trainset.group_weights,
    batch_size=64,
    model=model,
    device=device,
    verbose=True
)
evaluator.evaluate()
results["val_spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()
results[f"val_wg_acc"] = evaluator.worst_group_accuracy[1]
results[f"val_avg_acc"] = evaluator.average_accuracy

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
results["test_spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()
results[f"test_wg_acc"] = evaluator.worst_group_accuracy[1]
results[f"test_avg_acc"] = evaluator.average_accuracy


# evaluator = Evaluator(
#     testset=valset,
#     group_partition=valset.group_partition,
#     group_weights=trainset.group_weights,
#     batch_size=args.batch_size,
#     model=group_dro.best_model,
#     device=device,
#     verbose=True
# )
# evaluator.evaluate()
# results["val_early_stopping_spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()
# results[f"val_early_stopping_wg_acc"] = evaluator.worst_group_accuracy[1]
# results[f"val_early_stopping_avg_acc"] = evaluator.average_accuracy

# evaluator = Evaluator(
#     testset=testset,
#     group_partition=testset.group_partition,
#     group_weights=trainset.group_weights,
#     batch_size=args.batch_size,
#     model=group_dro.best_model,
#     device=device,
#     verbose=True
# )
# evaluator.evaluate()
# results["test_early_stopping_spurious_attribute_prediction"] = evaluator.evaluate_spurious_attribute_prediction()
# results[f"test_early_stopping_wg_acc"] = evaluator.worst_group_accuracy[1]
# results[f"test_early_stopping_avg_acc"] = evaluator.average_accuracy


print(results)

if args.wandb:
    # convert the results to a dictionary
    results = results.to_dict(orient="records")[0]
    wandb.log(results)
else:
    results["alg"] = "ssa"
    results["timestamp"] = pd.Timestamp.now()
    args_dict = vars(args)
    for key in args_dict.keys():
        results[key] = args_dict[key]


    if os.path.exists(args.results_csv):
        results_df = pd.read_csv(args.results_csv)
    else:
        results_df = pd.DataFrame()

    results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_csv(args.results_csv, index=False)

    print('Results saved to', args.results_csv)

    # close the stdout file
    sys.stdout.close()

    # restore stdout
    sys.stdout = sys.__stdout__

print('Done!')