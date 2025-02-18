import itertools
import yaml
import os

dataset = [ "robot", "ECG200"]
num_epochs = [5000]
nhead = [2, 4]
num_layers = [2, 4]
batch_size = [64, 128, 256]
combinations = list(itertools.product(dataset, num_epochs, nhead, num_layers, batch_size))

output_dir = "/home/yuchia/Learning-Shapelets/public_test/yaml_configs_vanilla"
os.makedirs(output_dir, exist_ok=True)

for i, combo in enumerate(combinations):
    config = {
        'dataset': combo[0],
        'num_epochs': combo[1],
        'nhead': combo[2],
        'num_layers': combo[3],
        'batch_size': combo[4], 
        'model_path': os.path.join('./', os.path.join('model', 'best_model_'+combo[0]+'_'+str(i)+'.pth'))
    }
    with open(os.path.join(output_dir, f'config_{i}.yaml'), 'w') as file:
        yaml.dump(config, file)