import itertools
import yaml
import os

def create_yaml():
    dataset = ["ECG200"]
    num_epochs = [2000]
    lr = [0.001]
    wd = [0.01, 0.001]
    combinations = list(itertools.product(dataset, num_epochs, lr, num_shapelets_ratio,wd, size_ratio))

    output_dir = "/home/yuchia/Learning-Shapelets/public_test/yaml_configs_normal"
    os.makedirs(output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        config = {
            'dataset': combo[0],
            'num_epochs': combo[1],
            'lr': combo[2],
            'num_shapelets_ratio': combo[3],
            'size_ratio': combo[5],
            'wd':combo[4]
        }
        with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
            yaml.dump(config, file)