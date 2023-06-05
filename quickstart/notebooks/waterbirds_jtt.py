import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("../..")
import torch 
from spuco.evaluate import Evaluator
from wilds import get_dataset
import torchvision.transforms as transforms

from spuco.utils import get_class_labels 
from spuco.utils import GroupLabeledDataset
from spuco.group_inference import JTTInference
from spuco.utils import Trainer
from torch.optim import SGD
from spuco.utils import WILDSDatasetWrapper
from spuco.invariant_train import UpSampleERM, DownSampleERM, CustomSampleERM
from spuco.models import model_factory 
import numpy as np

device = torch.device("cuda:6")

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True)

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
infer_lr = 1e-5
infer_momentum = 0.9
infer_batch_size = 64
infer_epoch = 60
infer_weight_decay = 1.0
infer_upsample = 100

dro_lr = 1e-5
dro_momentum = 0.9
dro_epoch = 300
dro_batch_size = 64
dro_weight_decay = 1.0

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transform
)

# Get the training set
test_data = dataset.get_subset(
    "test",
    transform=transform
)

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)


model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes).to(device)
trainer = Trainer(
    trainset=trainset,
    model=model,
    batch_size=infer_batch_size,
    optimizer=SGD(model.parameters(), lr=infer_lr, momentum=infer_momentum),
    device=device,
    verbose=True
)
trainer.train(infer_epoch)



predictions = torch.argmax(trainer.get_trainset_outputs(), dim=-1).detach().cpu().tolist()
jtt = JTTInference(
    predictions=predictions,
    class_labels=get_class_labels(trainset)
)

group_partition = jtt.infer_groups()



group_labeled_trainset = GroupLabeledDataset(trainset, group_partition)


model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes).to(device)

upsample_count = infer_upsample
indices = []
for key in group_partition.keys():
    group_reordered_indices = [group_partition[key][i] for i in torch.randperm(len(group_partition[key])).tolist()]
    group_indices = []
    while len(group_indices) < upsample_count:
        group_indices.extend(group_reordered_indices)
        indices.extend(group_indices[:upsample_count])

jtt_train = CustomSampleERM(
    model=model,
    num_epochs=dro_epoch,
    trainset=trainset,
    batch_size=dro_batch_size,
    indices=indices,
    optimizer=SGD(model.parameters(), lr=dro_lr, momentum=dro_momentum, weight_decay = dro_weight_decay),
    device=device,
    verbose=True
)
jtt_train.train()



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

np.savez("res/jtt_result_{}_{}".format(str(infer_lr), str(infer_epoch)), worst_acc=evaluator.worst_group_accuracy[1], ave_acc=evaluator.average_accuracy)