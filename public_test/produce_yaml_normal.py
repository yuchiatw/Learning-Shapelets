import itertools
import yaml
import os

def create_yaml():
    dataset = ["ECG200", "robot"]
    batch_size = [8, 16, 32, 64, 128]
    lr = [0.001]
    wd = [0.01, 0.001]
    size_ratio = [[0.125], [0.125, 0.15, 0.2]]
    combinations = list(itertools.product(dataset,batch_size, lr, wd, size_ratio))

    output_dir = "./yaml_configs_normal"
    os.makedirs(output_dir, exist_ok=True)

    for i, combo in enumerate(combinations):
        config = {
            'dataset': combo[0],
            'batch_size': combo[1],
            'lr': combo[2],
            'wd':combo[3], 
            'size_ratio':combo[4]
        }
        with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
            yaml.dump(config, file)

if __name__ == '__main__':
    create_yaml()