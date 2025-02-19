import itertools
import yaml
import os

def create_yaml():
    dataset = ["ECG200", "robot"]
    num_epochs = [2000]
    lr = [0.001]
    wd = [0.01, 0.001]
    size_ratio = [0.1, 0.125, 0.15, 0.175, 0.2]
    combinations = list(itertools.product(dataset, num_epochs, lr, wd))

    output_dir = "./yaml_configs_normal"
    os.makedirs(output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        config = {
            'dataset': combo[0],
            'num_epochs': combo[1],
            'lr': combo[2],
            'wd':combo[3], 
            'size_ratio':size_ratio
        }
        with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
            yaml.dump(config, file)

if __name__ == '__main__':
    create_yaml()