import torch 
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("../..")
device = torch.device("cuda:6")

from spuco.utils import set_seed

set_seed(0)

from wilds import get_dataset
import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True)

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

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

from spuco.utils import WILDSDatasetWrapper

trainset = WILDSDatasetWrapper(dataset=train_data, metadata_spurious_label="background", verbose=True)
testset = WILDSDatasetWrapper(dataset=test_data, metadata_spurious_label="background", verbose=True)

from spuco.utils import GroupLabeledDataset

group_labeled_trainset = GroupLabeledDataset(trainset, trainset.group_partition)

trainset.group_weights

from spuco.models import model_factory 

model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes).to(device)

from torch.optim import SGD
from spuco.invariant_train import GroupDRO 

group_dro = GroupDRO(
    model=model,
    num_epochs=150,
    trainset=group_labeled_trainset,
    batch_size=128,
    optimizer=SGD(model.parameters(), lr=0.00001, momentum=0.9,weight_decay = 1),
    device=device,
    verbose=True
)
group_dro.train()
# erm = ERM(
#     model=model,
#     num_epochs=1,
#     trainset=trainset,
#     batch_size=128,
#     optimizer=SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True),
#     device=device,
#     verbose=True
# )
# erm.train()

from spuco.evaluate import Evaluator

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
print(evaluator.average_accuracy)