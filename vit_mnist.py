from __future__ import print_function
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
import optuna

wandb.init(project='vit_mnist',name='mnist_optuna')

#wandb.save(__file__)
torch.manual_seed(413)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 128
train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': batch_size}

cuda_kwargs = {'num_workers': 8,
               'pin_memory': True,
               'shuffle': True}
    
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)

train_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.2, 2), 
                               contrast=(0.3, 2), 
                               ),
        transforms.RandomAffine(30),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)



dataset1 = datasets.MNIST('../data', train=True, download=True,
                   transform=train_transforms)
dataset2 = datasets.MNIST('../data', train=False,
                   transform=val_transforms)

train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train(model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    print('Train Epoch: {} Loss: {:.6f}'.format(
        epoch, loss.item()))
    



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        val_acc))
    
    return val_acc

from vit_pytorch import ViT

epochs = 20
gamma = 0.7


def Objective(trial):
    
    dim = trial.suggest_categorical('dim',[32, 64, 128])
    #patch_size = trial.suggest_int('patch_size',7, 14, 7)
    patch_size = 7
    depth = trial.suggest_categorical('depth',[8, 16, 32])
    heads = trial.suggest_categorical('heads',[8, 16, 32])
    mlp_dim = trial.suggest_categorical('mlp_dim',[128, 512, 1024])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    print('dim:', dim, 'mlp_dim:',mlp_dim, 'depth:',depth, 'heads:',heads)
    model = ViT(
        dim=dim,
        image_size=28,
        patch_size=patch_size,
        num_classes=10,
        depth=depth, # number of transformer blocks
        heads=heads, # number of multi-channel attention
        mlp_dim=mlp_dim,
        channels=1,
        #dropout=0.2,
    )


    # vanila cnn : 0.96
    # model = Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, criterion, device, train_loader, optimizer, epoch)
        val_acc = test(model, device, test_loader)
        scheduler.step()
        
        if 0:
            torch.save(model.state_dict(), "mnist_cnn.pt")
    
        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    wandb.log({'val_acc': val_acc})        
    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(Objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
