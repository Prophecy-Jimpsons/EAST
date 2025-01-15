# src/main.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.east import EAST
from training.trainer import train_model
from torchvision.models import resnet18
from torch.utils.data import Subset, DataLoader
import numpy as np

def create_small_resnet():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)
    return model

def create_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    return trainset

def create_small_dataloader(dataset, samples=100):
    indices = np.random.choice(len(dataset), samples, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

def main():
    # Create datasets and dataloaders
    trainset = create_datasets()
    small_dataloader = create_small_dataloader(trainset)
    
    # Create model and optimizer
    model = create_small_resnet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create EAST instance
    east = EAST(
        model,
        max_sparsity=0.9999,
        min_sparsity=0.99,
        cycle_length=50,
        num_cycles=2,
        update_freq=50
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    num_epochs = 250
    train_model(model, optimizer, criterion, small_dataloader, num_epochs, east)

if __name__ == "__main__":
    main()
