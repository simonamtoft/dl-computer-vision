import torch
from training import update_metrics, remove_anno_dim
from helpers import loss_func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_pass_medical(config, model, test_loader):

    # Initialize loss function
    loss_fn = loss_func(config)

    # Initialize metrics
    test_dict = {'loss': []}
    test_loss = 0
    metrics_test = torch.tensor([0, 0, 0, 0, 0])

    # Test loop on entire test dataset
    model.eval()
    for X_test, Y_test in test_loader:
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        with torch.no_grad():
            Y_pred = model(X_test)
        
        # fix some dimensionality
        Y_test = remove_anno_dim(Y_test)

        # Compute loss and metrics
        n_test = len(test_loader)
        test_loss += loss_fn(Y_pred, Y_test).cpu().item() / n_test
        metrics_test = update_metrics(metrics_test, Y_pred, Y_test, n_test)
        test_dict['loss'].append(test_loss)

    test_dict['metrics'] = metrics_test
    return test_dict