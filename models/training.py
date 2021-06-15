from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output
import matplotlib.pyplot as plt

from .losses import loss_func
from helpers import compute_metrics
from models import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_ensemble(N, config, train_loader, val_loader, project_name, plotting=True, save_fig=False):
    models = []
    for i in range(N):
        print(f"\nTrain model {i+1}:\n")
        model = UNet(config).to(device)
        train_medical(model, config, train_loader, val_loader, project_name, plotting, save_fig)
        models.append(model)
    return models


def train_anno_ensemble(config, train_loader, val_loader, project_name, plotting=True, save_fig=False):
    wandb.init(project=project_name, config=config)
    
    # Define four models
    models = [UNet(config).to(device) for _ in range(4)]
    
    # Set optimizers for each model
    if config["optimizer"] == "adam":
        optimizers = [torch.optim.Adam(models[i].parameters(), lr=config["learning_rate"]) for i in range(4)]
    elif config["optimizer"] == "sgd":
        optimizers = [torch.optim.SGD(models[i].parameters(), lr=config["learning_rate"]) for i in range(4)]
    else: 
        raise Exception('Optimizer not implemented. Chose "adam" or "sgd".')
    
    # set learning rate scheduler for each model
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
                loss = loss_fn(Y_pred, Y_batch[:, i, :, :]) # forward-pass
                loss.backward()                             # backward-pass
                optimizers[i].step()                        # update weights

                # calculate metrics to show the user
                avg_losses[i] += loss.item() / len(train_loader)

        # Step the learning rate
        if config['step_lr'][0]:
            [schedulers[i].step() for i in range(4)]

        [models[i].eval() for i in range(4)]

        val_losses = [0 for _ in range(4)]
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            for i in range(4):
                with torch.no_grad():
                    output = models[i](X_val)

                output = pad_output(output, Y_val[:, i, :, :])
                val_losses[i] += loss_fn(output, Y_val[:, i, :, :]).cpu().item() / len(val_loader)

        # Plot annotations against model predictions on validation data
        if plotting:
            X_val, Y_val = next(iter(val_loader))
            with torch.no_grad():
                Y_hats = torch.stack([torch.sigmoid(models[i](X_val.to(device))).detach() for i in range(4)], dim=1).cpu()
            
            # 
            Y_hats_std = torch.std(Y_hats, axis=1)
            Y_val_std = torch.std(Y_val, axis=1)
            Y_hats_mean = torch.mean(Y_hats, axis=1)
            Y_val_mean = torch.mean(Y_val, axis=1)

            # Show plots
            clear_output(wait=True)
            f, ax = plt.subplots(3, 6, figsize=(14, 6))
            for k in range(6):
                ax[0,k].imshow(X_val[k, 0].numpy(), cmap='gray')
                ax[0,k].set_title('Real data')
                ax[0,k].axis('off')

                ax[1,k].imshow(Y_hats_std[k, 0], cmap='hot')
                ax[1,k].set_title('Ensemble Std')
                ax[1,k].axis('off')

                ax[2,k].imshow(Y_val_std[k, 0], cmap='hot')
                ax[2,k].set_title('Segmentation Std')
                ax[2,k].axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, config['epochs'], np.mean(avg_losses)))
            if not save_fig:
                plt.show()
            else: 
                plt.savefig(f"fig{epoch+1}_std.png", transparent=True)
                plt.close()

            f, ax = plt.subplots(3, 6, figsize=(14, 6))
            for k in range(6):
                ax[0,k].imshow(X_val[k, 0].numpy(), cmap='gray')
                ax[0,k].set_title('Real data')
                ax[0,k].axis('off')

                ax[1,k].imshow(Y_hats_mean[k, 0], cmap='hot')
                ax[1,k].set_title('Ensemble Mean')
                ax[1,k].axis('off')

                ax[2,k].imshow(Y_val_mean[k, 0], cmap='hot')
                ax[2,k].set_title('Segmentation Mean')
                ax[2,k].axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, config['epochs'], np.mean(avg_losses)))
            if not save_fig:
                plt.show()
            else: 
                plt.savefig(f"fig{epoch+1}_mean.png", transparent=True)
                plt.close()

        # log to weight & bias
        print("Train losses:")
        print(avg_losses)
        print("Validation losses:")
        print(val_losses)
        wandb.log({
            "train_loss0": avg_losses[0],
            "train_loss1": avg_losses[1],
            "train_loss2": avg_losses[2],
            "train_loss3": avg_losses[3],
            "val_loss0": val_losses[0],
            "val_loss1": val_losses[1],
            "val_loss2": val_losses[2],
            "val_loss3": val_losses[3],
        })
    
    # finish run
    wandb.finish()
    return models


def train_medical(model, config, train_loader, val_loader, project_name="tmp", plotting=True, save_fig=False):
    train_dict = {'loss': []}
    val_dict = {'loss': []}

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

    clear_output(wait=True)
    for epoch in range(config['epochs']):
        print(f"* Epoch {epoch+1}/{config['epochs']}")

        model.train()

        # Training pass
        avg_loss = 0
        metrics_train = torch.tensor([0, 0, 0, 0, 0])
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
            Y_pred = pad_output(Y_pred, Y_batch)
            loss = loss_fn(Y_pred, Y_batch) # forward-pass
            loss.backward()                 # backward-pass
            optimizer.step()                # update weights

            # calculate metrics to show the user
            avg_loss += loss.item() / len(train_loader)

            if epoch == config['epochs']-1:
                metrics_train = update_metrics(metrics_train, Y_pred, Y_batch, len(train_loader))
        
        # Step the learning rate
        if config['step_lr'][0]:
            scheduler.step()

        model.eval()

        # Compute validation loss
        val_loss = 0
        metrics_val = torch.tensor([0, 0, 0, 0, 0])
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            with torch.no_grad():
                output = model(X_val)

            # If we have multiple annotations loaded
            if Y_val.ndim > 4:
                Y_val = Y_val[:, 0, :, :]
            
            output = pad_output(output, Y_val)
            val_loss += loss_fn(output, Y_val).cpu().item() / len(val_loader)

        # Plot annotations against model predictions on validation data
        if plotting:
            # Get some validation data
            X_val, Y_val = next(iter(val_loader))
            with torch.no_grad():
                Y_hat = torch.sigmoid(model(X_val.to(device))).detach().cpu()
            
            # If we have multiple annotations loaded
            if Y_val.ndim > 4:
                Y_val = Y_val[:, 0, :, :]
            
            Y_hat = pad_output(Y_hat, Y_val)

            if epoch == config['epochs']-1:
                metrics_val = update_metrics(metrics_val, Y_hat, Y_val, len(val_loader))
            
            # Plot
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
        
        # save loss in dicts
        train_dict['loss'].append(avg_loss)
        val_dict['loss'].append(val_loss)

        # log to weight & bias
        wandb.log({
            "train_loss": avg_loss,
            "valid_loss": val_loss,
        })
    
    # Add metrics to dicts
    train_dict['metrics'] = metrics_train
    val_dict['metrics'] = metrics_val


    # finish run
    wandb.finish()
    return train_dict, val_dict


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


def pad_output(Y_pred, Y_batch):
    pad_size = (Y_batch.shape[2] - Y_pred.shape[2])//2
    return F.pad(Y_pred, (pad_size, pad_size, pad_size, pad_size))


def update_metrics(metrics, y_pred, y_real, n):
    # Get predictions
    y_pred = torch.sigmoid(y_pred).detach().cpu()
    y_pred = y_pred > 0.5
    
    # Update metrics
    metrics += compute_metrics(y_pred, y_real.cpu())
    metrics /= n
    return metrics
    

    

