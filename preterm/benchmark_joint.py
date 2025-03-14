import itertools
import yaml
import os

nhead = [2]
epochs = [200, 400]
batch_size = [32, 64, 128]

combinations = list(itertools.product(
        epochs, batch_size
    ))

output_dir = './yaml_LS_pips'
os.makedirs(output_dir, exist_ok=True)

base_config = {
    'data_loading': {
        'test_ratio': 0.2,
        'val_ratio': 0.2,
        'norm_std': 'standard',
        'norm_mode': 'local_before',
        'seq_min': 15,
        'pad_min': 3, 
        'step_min': 1,
    },
    'init_mode': 'pips',
    'model_mode': 'LS_Transformer',
    'init_config': {
        'ws_rate': 0.1,
        'num_pip': 0.1, 
        'num_shapelets_make': 100, 
        'num_shapelets': 10,
    },
    'model_config': {
        'epochs': 200, 
        'batch_size': 32, 
        'nhead': 2,
        'num_layers': 4,
        'model_path': './model/best_model',
        'step': 1,
        'lr': 1e-3, 
        'wd': 1e-4, 
        'epsilon': 1e-7
    },
}

for i, combo in enumerate(combinations):
    config = base_config.copy()
    config['model_config']['model_path'] = f'./model/preterm_LT_PIPs_{i}.pth'
    config['model_config']['epochs'] = combo[0]
    config['model_config']['batch_size'] = combo[1]
    
    with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

