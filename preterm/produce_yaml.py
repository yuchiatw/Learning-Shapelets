import itertools
import yaml
import os

folder = ['results']
batch_size = [256]
num_epochs = [200]
nhead = [2, 4]
num_layers = [2]
size_ratio = [0.05, 0.1, 0.15, 0.2]
num_shapelets_ratio = [0.2]
step = [25]
combinations = list(itertools.product(folder, batch_size, num_epochs, nhead, num_layers, step, size_ratio, num_shapelets_ratio))

output_dir = "/home/yuchia/Learning-Shapelets/preterm/yaml_configs_new"
os.makedirs(output_dir, exist_ok=True)

for i, combo in enumerate(combinations):
    config = {
        'folder': combo[0], 
        'batch_size': combo[1],
        'num_epochs': combo[2],
        'nhead': combo[3],
        'num_layers': combo[4],
        'step': combo[5], 
        'size_ratio': combo[6], 
        'num_shapelets_ratio': combo[7]
    }
    with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
        yaml.dump(config, file)