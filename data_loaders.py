import torch
import torchvision
from torchvision import transforms


def setup_data_loaders(logger):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])   #load the CIFAR10 dataset and normalize
    
    batch_size = 4
    logger.debug("about to setup datasets and data loaders")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader,testloader


def get_images_labels(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    return images,labels