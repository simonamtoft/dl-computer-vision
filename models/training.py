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
from models import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_ensemble(config, train_loader, val_loader, project_name, plotting=True, save_fig=False):
    wandb.init(project=project_name, config=config)
    models = [UNet(config).to(device) for i in range(4)]
    # Set optimizer
    if config["optimizer"] == "adam":
        optimizers = [torch.optim.Adam(models[i].parameters(), lr=config["learning_rate"]) for i in range(4)]
    elif config["optimizer"] == "sgd":
        optimizers = [torch.optim.SGD(models[i].parameters(), lr=config["learning_rate"]) for i in range(4)]
    else: 
        raise Exception('Optimizer not implemented. Chose "adam" or "sgd".')
    
    # set learning rate scheduler
    if config['step_lr'][0]:
        schedulers = [torch.optim.lr_scheduler.StepLR(
            optimizers[i], 
            step_size=config['step_lr'][1], 
            gamma=config['step_lr'][2]
        ) for i in range(4)]
    
    # set loss function
    loss_fn = loss_func(config)

    # perform training
    clear_output(wait=True)
    for epoch in range(config['epochs']):
        print(f"* Epoch {epoch+1}/{config['epochs']}")

        avg_losses = [0 for _ in range(4)]
        [models[i].train() for i in range(4)]
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            for i in range(4):
                # set parameter gradients to zero
                optimizers[i].zero_grad()

                # model pass
                Y_pred = models[i](X_batch)

                # update
                loss = loss_fn(Y_pred, Y_batch[:,i,:,:]) # forward-pass
                loss.backward()                 # backward-pass
                optimizers[i].step()            # update weights

                # calculate metrics to show the user
                avg_losses[i] += loss.item() / len(train_loader)
        
        # print some metrics
        [print(' - loss: %f' % avg_losses[i]) for i in range(4)]

        if config['step_lr'][0]:
            [schedulers[i].step() for i in range(4)]

        # show intermediate results
        [models[i].eval() for i in range(4)]
        X_val, Y_val = next(iter(val_loader))
        with torch.no_grad():
            Y_hats = torch.stack([torch.sigmoid(models[i](X_val.to(device))).detach() for i in range(4)], dim=1).cpu()

        if plotting:
            clear_output(wait=True)
            f, ax = plt.subplots(4, 6, figsize=(14, 6))
            for k in range(6):
                ax[0,k].imshow(X_val[k, 0].numpy(), cmap='gray')
                ax[0,k].set_title('Real data')
                ax[0,k].axis('off')

                ax[1,k].imshow(torch.std(Y_hats, axis=1)[k, 0], cmap='hot')
                ax[1,k].set_title('Ensemble Std')
                ax[1,k].axis('off')

                ax[2,k].imshow(torch.std(Y_val, axis=1)[k, 0], cmap='hot')
                ax[2,k].set_title('Segmentation Std')
                ax[2,k].axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, config['epochs'], np.mean(avg_losses)))
            if not save_fig:
                plt.show()
            else: 
                plt.savefig(f"fig{epoch+1}.png", transparent=True)
                plt.close()

        # log to weight & bias
        wandb.log({
            "train_loss0": avg_losses[0],
            "train_loss1": avg_losses[1],
            "train_loss2": avg_losses[2],
            "train_loss3": avg_losses[3],
        })
    
    # finish run
    wandb.finish()
    return models

def train_medical(model, config, train_loader, val_loader, project_name="tmp", plotting=True, save_fig=False):
    # Initialise wandb
    wandb.init(project=project_name, config=config)

    # Set optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else: 
        raise Exception('Optimizer not implemented. Chose "adam" or "sgd".')
    
    # set learning rate scheduler
    if config['step_lr'][0]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_lr'][1], 
            gamma=config['step_lr'][2]
        )
    
    # set loss function
    loss_fn = loss_func(config)

    # perform training
    clear_output(wait=True)
    for epoch in range(config['epochs']):
        print(f"* Epoch {epoch+1}/{config['epochs']}")

        avg_loss = 0
        model.train()
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # If we have multiple annotations loaded
            if Y_batch.ndim > 4:
                Y_batch = Y_batch[:, 0, :, :]

            # set parameter gradients to zero
            optimizer.zero_grad()

            # model pass
            Y_pred = model(X_batch)

            # update
            loss = loss_fn(Y_pred, Y_batch) # forward-pass
            loss.backward()                 # backward-pass
            optimizer.step()                # update weights

            # calculate metrics to show the user
            avg_loss += loss.item() / len(train_loader)
        
        # print some metrics
        print(' - loss: %f' % avg_loss)

        if config['step_lr'][0]:
            scheduler.step()

        # show intermediate results
        model.eval()
        X_val, Y_val = next(iter(val_loader))
        with torch.no_grad():
            Y_hat = torch.sigmoid(model(X_val.to(device))).detach().cpu()

        # If we have multiple annotations loaded
        if Y_val.ndim > 4:
            Y_val = Y_val[:, 0, :, :]

        if plotting:
            clear_output(wait=True)
            f, ax = plt.subplots(3, 6, figsize=(14, 6))
            for k in range(6):
                ax[0,k].imshow(X_val[k, 0].numpy(), cmap='gray')
                ax[0,k].set_title('Real data')
                ax[0,k].axis('off')

                ax[1,k].imshow(Y_hat[k, 0], cmap='gray')
                ax[1,k].set_title('Model Output')
                ax[1,k].axis('off')

                ax[2,k].imshow(Y_val[k, 0], cmap='gray')
                ax[2,k].set_title('Real Segmentation')
                ax[2,k].axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, config['epochs'], avg_loss))
            if not save_fig:
                plt.show()
            else: 
                plt.savefig(f"fig{epoch+1}.png", transparent=True)
                plt.close()

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
