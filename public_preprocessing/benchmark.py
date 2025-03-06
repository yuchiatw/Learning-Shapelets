import itertools
import yaml
import os

window_size = [20, 30, 40]
n_bins = [2, 5, 8]
window_step = [1]

combinations = list(itertools.product(
        window_size, n_bins, window_step
    ))

output_dir = './yaml_BOSS'
os.makedirs(output_dir, exist_ok=True)

base_config = {
    'data_loading': {
        'test_ratio': 0.2,
        'val_ratio': 0.2,
        'norm_std': 'standard',
    },
    'model_mode': 'BOSS',
    'model_config':{}
}

for i, combo in enumerate(combinations):
    config = base_config.copy()
    config['model_config']['window_size'] = combo[0]
    config['model_config']['n_bins'] = combo[1]
    config['model_config']['window_step'] = combo[2]
    
    with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)