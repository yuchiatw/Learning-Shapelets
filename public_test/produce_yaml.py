import itertools
import yaml
import os

# batch_size = [8]
dataset = [ "robot", "ECG200"]
num_epochs = [10, 100, 200]
nhead = [2, 4]
num_layers = [2, 4]
# num_shapelets_ratio = [0.1, 0.2, 0.3]
# size_ratio = [0.1, 0.15, 0.2, 0.25]
step = [1, 5, 10]
combinations = list(itertools.product(dataset, num_epochs, nhead, num_layers, step))

output_dir = "/home/yuchia/Learning-Shapelets/public_test/yaml_configs"
os.makedirs(output_dir, exist_ok=True)

for i, combo in enumerate(combinations):
    config = {
        'dataset': combo[0],
        'num_epochs': combo[1],
        'nhead': combo[2],
        'num_layers': combo[3],
        'step': combo[4]
    }
    with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
        yaml.dump(config, file)