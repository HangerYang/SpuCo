import subprocess
import sys
import numpy as np
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("../..")

import torch 
from spuco.datasets import SpuCoMNIST, SpuriousFeatureDifficulty
import torchvision.transforms as T
from spuco.datasets import GroupLabeledDatasetWrapper
from torch.optim import SGD
from spuco.evaluate import Evaluator
from spuco.group_inference import JTTInference
from spuco.models import model_factory 
from spuco.invariant_train.jtt_erm import JTTERM
from spuco.utils import Trainer
from spuco.utils import set_seed, get_group_ratios

device = torch.device("cuda:0")
infer_lr = 0.001
infer_momentum = 0.9
infer_batch_size = 32
infer_epoch= 10
infer_weight_decay = 0.01
infer_upsample = 800
label_noise = 0
feature_noise= 0


dro_lr = infer_lr
dro_momentum = infer_momentum
dro_epoch = 20
dro_batch_size = 32
dro_weight_decay = infer_weight_decay
classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
difficulty = SpuriousFeatureDifficulty.VARIANCE_HARD
seed = 1
set_seed(seed)
trainset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    core_feature_noise=feature_noise,
    label_noise = label_noise,
    spurious_correlation_strength=0.995,
    classes=classes,
    split="train"
)
trainset.initialize()
# majority_group = []
# minority_group = []
# for color in range(5):
#     for number in range(5):
#         if color == number:
#             majority_group = majority_group + trainset.true_group_partition[(color, number)]
#         else:
#             minority_group =  minority_group + trainset.true_group_partition[(color, number)]

testset = SpuCoMNIST(
                        root="/data/mnist/",
                        spurious_feature_difficulty=difficulty,
                        classes=classes,
                        split="test"
                    )
testset.initialize()

valset = SpuCoMNIST(
    root="/data/mnist/",
    spurious_feature_difficulty=difficulty,
    classes=classes,
    split="val"
)
valset.initialize()
# ratios = {"0":[], "1":[], "2":[]}

model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)
trainer = Trainer(
    trainset=trainset,
    model=model,

    batch_size=infer_batch_size,
    optimizer=SGD(model.parameters(), lr=infer_lr, momentum=infer_momentum, weight_decay=infer_weight_decay),
    device=device,
    verbose=True
)
trainer.train(infer_epoch)
predictions = torch.argmax(trainer.get_trainset_outputs(), dim=-1).detach().cpu().tolist()
jtt = JTTInference(
    predictions=predictions,
    class_labels=trainset.labels)
group_partition = jtt.infer_groups()
group_labeled_trainset = GroupLabeledDatasetWrapper(trainset, group_partition)
model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)


upsample_count = infer_upsample
indices = []
min_leng = min([len(group_partition[idx]) for idx in group_partition])
worst_group = [key for key, val in group_partition.items() if len(val) == min_leng][0]
for key in group_partition.keys():
    group_reordered_indices = [group_partition[key][i] for i in torch.randperm(len(group_partition[key])).tolist()]
    group_indices = []
    if key == worst_group:
        while len(group_indices) < upsample_count * len(group_partition[worst_group]):
            group_indices.extend(group_reordered_indices)
        indices.extend(group_indices)
    else:
        group_indices.extend(group_reordered_indices)
        indices.extend(group_indices)

jtt_train = JTTERM(
    model=model,
    num_epochs=dro_epoch,
    trainset=trainset,
    valset=valset,
    batch_size=dro_batch_size,
    indices=indices,
    optimizer=SGD(model.parameters(), lr=dro_lr, momentum=dro_momentum, weight_decay = dro_weight_decay),
    device=device,
    verbose=True
)
jtt_train.train()
model = jtt_train.best_model

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

np.savez("jtt/jtt_cmnist_final_result_{}_{}2".format(difficulty, seed), worst_acc=evaluator.worst_group_accuracy[1], ave_acc=evaluator.average_accuracy)
#         wrong_mj = 0
#         wrong_mi = 0
#         # super_wrong_mj = 0
#         # super_wrong_mi = 0
#         for i in group_partition[(0,1)]:
#             if i in majority_group:
#                 wrong_mj = wrong_mj + 1
#                 # if i in noisy_group:
#                 #     super_wrong_mj = super_wrong_mj + 1

#             else:
#                 wrong_mi = wrong_mi + 1
#                 # if i in noisy_group:
#                 #     super_wrong_mi = super_wrong_mi + 1
#         precision = wrong_mi / (wrong_mj + wrong_mi + 0.000001)
#         recall = wrong_mi / (len(minority_group )+ 0.0000001)
#         ratios[str(seed)].append((precision, recall))
# np.savez("jtt/res_easyv", ratio=ratios)
