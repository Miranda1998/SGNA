""" 
    This file constrains best paramaters found by random search over 100 configurations
    for each instance and NN-{E,P} model.  These can be imported directly to train each model 
    and reproduce the results.  
"""

# Optimal parameters for NN-P
nn_p_params = {
    'dblrp_10_10': {
        'hidden_dims': [64],
        'lr': 0.00768,
        'dropout': 0.07346,
        'optimizer_type': 'RMSprop',
        'batch_size': 32,
        'loss_fn': 'MSELoss',
        'wt_lasso': 0.0,
        'wt_ridge': 0.0,
        'log_freq': 10,
        'n_epochs': 1000,
        'use_wandb': 0}}

# Optimal parameters for NN-E
nn_e_params = {
    'dblrp_10_10': {
        'embed_hidden_dim': 512,
        'embed_dim1': 64,
        'embed_dim2': 16,
        'relu_hidden_dim': 512,
        'agg_type': 'mean',
        'lr': 0.00436,
        'dropout': 0.04226,
        'optimizer_type': 'Adam',
        'batch_size': 128,
        'loss_fn': 'MSELoss',
        'wt_lasso': 0.09067,
        'wt_ridge': 0.02603,
        'log_freq': 10,
        'n_epochs': 2000,
        'use_wandb': 0}
   }