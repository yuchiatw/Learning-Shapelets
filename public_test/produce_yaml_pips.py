import itertools
import yaml
import os

batch_size = [8]
dataset = ["ECG200"]
num_epochs = [200]
window_size_ratio = [0.1]
num_pips = [0.1, 0.2]
num_shapelets = [5, 10, 20]
combinations = list(itertools.product(
    dataset, 
    num_epochs,
    batch_size,
    window_size_ratio,
    num_pips, 
    num_shapelets
))

output_dir = "/home/yuchia/Learning-Shapelets/public_test/yaml_configs_pips"
os.makedirs(output_dir, exist_ok=True)

for i, combo in enumerate(combinations):
    config = {
        'dataset': combo[0],
        'num_epochs': combo[1],
        'batch_size': combo[2],
        'window_size_ratio': combo[3],
        'num_pips': combo[4],
        'num_shapelets': combo[5]
        
    }
    with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
        yaml.dump(config, file)