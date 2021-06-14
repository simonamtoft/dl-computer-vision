from tqdm import tqdm
import wandb
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt

from .losses import loss_func
from helpers import compute_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_medical(model, config, train_loader, val_loader, project_name="tmp", plotting=True):
    # Initialise wandb
    wandb.init(project=project_name, config=config)

    # Set optimizer
    if config["optimizer"] == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    
    # set learning rate scheduler
    if config['step_lr'][0]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_lr'][1], 
            gamma=config['step_lr'][2]
        )
    
    # set loss function
    if config['loss_func'] == 'ce':
        raise Exception('Cannot use normal Cross Entropy loss for this training.\nUse BCE instead.')
    else:
        loss_fn = loss_func(config['loss_func'])

    # perform training
    for epoch in range(config['epochs']):
        print(f"* Epoch {epoch+1}/{config['epochs']}")

        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            if Y_batch.ndim > 4:
                Y_batch = Y_batch[:, 0, :, :]

            # set parameter gradients to zero
            optimizer.zero_grad()

            # model pass
            Y_pred = model(X_batch)

            # pad output with zeros such that it fits original shape
            pad_size = (Y_batch.shape[2] - Y_pred.shape[2])//2
            Y_pred = F.pad(Y_pred, (pad_size, pad_size, pad_size, pad_size))

            # update
            loss = loss_fn(Y_pred, Y_batch) # forward-pass
            loss.backward()                 # backward-pass
            optimizer.step()                # update weights

            # calculate metrics to show the user
            avg_loss += loss / len(train_loader)
    
        print(' - loss: %f' % avg_loss)

        if config['step_lr'][0]:
            scheduler.step()

        # show intermediate results
        model.eval()  # testing mode
        X_val, Y_val = next(iter(val_loader))
        with torch.no_grad():
            Y_hat = F.sigmoid(model(X_val.to(device))).detach().cpu()
        clear_output(wait=True)
        
        if plotting:
            f, ax = plt.subplots(3, 6, figsize=(14, 6))
            for k in range(6):
                ax[0,k].imshow(X_val[k, 0].numpy(), cmap='gray')
                ax[0,k].set_title('Real data')
                ax[0,k].axis('off')

                y_hat = Y_hat[k, 0]
                ax[1,k].imshow(Y_hat[k, 0], cmap='gray')
                ax[1,k].set_title('Model Output')
                ax[1,k].axis('off')

                ax[2,k].imshow(Y_val[k, 0], cmap='gray')
                ax[2,k].set_title('Real Segmentation')
                ax[2,k].axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, config['epochs'], avg_loss))
            plt.show()

        # print performance metrics
        dice, iou, acc, sens, spec = compute_metrics(Y_hat, Y_batch)
        print(f'Dice: {dice}\nIoU: {iou}\nAccuracy: {acc}\nSensitivity: {sens}\nSpecificity: {spec}')

        # log to weight & bias
        wandb.log({
            "train_loss": avg_loss,
        })
    
    # finish run
    wandb.finish()


def train(model, config, project_name, train_loader, test_loader, n_train, n_test):
    # define output dict for return of function
    out_dict = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': []
    }

    # Initialise wandb
    wandb.init(project=project_name, config=config)
    
    # Set optimizer
    if config["optimizer"] == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    
    # set learning rate scheduler
    if config['step_lr'][0]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_lr'][1], 
            gamma=config['step_lr'][2]
        )

    # do training
    for _ in tqdm(range(config["epochs"]), desc='epoch'):
        model.train()
        
        # For each epoch
        train_correct = 0
        train_loss = []
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            
            # Forward pass your image through the network
            output = model(data)
            
            # Compute the loss
            loss = loss_func()(output, target)
            
            # Backward pass through the network
            loss.backward()
            
            # Update the weights
            optimizer.step()
            train_loss.append(loss.item())

            # Compute how many were correctly classified
            predicted = torch.reshape(output, (-1, 11)).argmax(1)
            train_correct += (target==predicted).sum().cpu().item()

        if config['step_lr'][0]:
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
            predicted = torch.reshape(output,(-1, 11)).argmax(1)
            test_correct += (target==predicted).sum().cpu().item()

        # compute losses and accuracy
        train_loss = np.mean(train_loss)
        test_loss = np.mean(test_loss)
        train_acc = train_correct/n_train
        test_acc = test_correct/n_test

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
    
    # finish run
    wandb.finish()
    return out_dict
