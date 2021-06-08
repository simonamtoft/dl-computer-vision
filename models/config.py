model_name = "model_1"
project_name = "project-1-1"
config_1_1 = {
    'epochs': 20,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'conv_dim': [16, 16, 16, 32, 32, 32, 48, 48, 64],
    'fc_dim': [1024],
    'batch_norm': True,
    'step_lr': [True, 1, 0.8],
}
