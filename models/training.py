import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_func(output, target):
    if output.ndim > 2:
        output = torch.reshape(output, (-1, output.shape[1]))
    return nn.CrossEntropyLoss()(output, target)


def get_optim(config, model):
    # set optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])

    # set scheduler
    if config['step_lr'][0]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_lr'][1], 
            gamma=config['step_lr'][2]
        )
    return optimizer, scheduler


def get_loaders(config, trainset, testset):
    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        testset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def train(model, config, project_name, trainset, testset):
    out_dict = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }

    # Initialise wandb
    wandb.init(project=project_name, config=config)
    
    # Set optimizer and acheduler
    optimizer, scheduler = get_optim(config, model)

    # Get data loaders
    train_loader, test_loader = get_loaders(config, trainset, testset)

    # do training
    print("Start of training\n\n")
    for _ in tqdm(range(config["epochs"]), desc='epoch'):
        model.train()
        
        # For each epoch
        train_correct = []
        train_loss = []
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            
            # Forward pass your image through the network
            output = model(data)
            
            # Compute the loss
            loss = loss_func(output, target)
            train_loss.append(loss.item())
            
            # Backward pass through the network
            loss.backward()
            
            # Update the weights
            optimizer.step()

            # Compute how many were correctly classified
            predicted = torch.reshape(output, (-1, 11)).argmax(1)
            train_correct += (target==predicted).sum().cpu().item()

        if config['step_lr']:
            scheduler.step()

        # Compute the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_func(output, target).cpu().item())
            predicted = torch.reshape(output, (-1, 11)).argmax(1)
            test_correct += (target==predicted).sum().cpu().item()

        # save losses and accuracy
        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        
        # save as dict
        out_dict['train_acc'].append(train_acc)
        out_dict['test_acc'].append(test_acc)
        out_dict['train_loss'].append(train_loss)
        out_dict['test_loss'].append(test_loss)

        # log to weight & bias
        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })
    return out_dict
