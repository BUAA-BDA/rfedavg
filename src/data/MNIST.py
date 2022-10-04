import torchvision.datasets
import torchvision.transforms as transforms
from src.data.utils import *

root_dir = './data/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root_dir, 
    train=True, 
    transform=transform, 
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    root_dir, 
    train=False, 
    transform=transform, 
    download=True,
)

def niid(params):
    num_user = params['Trainer']['n_clients']
    s = params['Dataset']['s']
    dataset_split = split_dataset_by_percent(train_dataset, test_dataset, s, num_user)
    testset_dict = {
        'train': None,
        'test': test_dataset,
    }
    return dataset_split, testset_dict