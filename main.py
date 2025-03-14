import os
import glob
from exp_functions import exp
from utils.evaluation_and_save import save_results_to_csv
import argparse
parser = argparse.ArgumentParser()
# ------------------------------ Input and Output -------------------------------------
parser.add_argument('--datatype', type=str, default='private', choices={'public', 'private'})
parser.add_argument('--dataset', type=str, default='preterm')
parser.add_argument('--version', type=str, default='3')
parser.add_argument('--yaml_path')
# -------------------------------------------------------------------------------------
# configuration loading
config = { # default
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
    'model_mode': 'LS_Transformer', # 'JOINT' / 'LS_FCN' / 'LS_Transformer' / 'BOSS'
    'init_config': {
        'ws_rate': 0.1,
        'num_pip': 0.1, 
        'num_shapelets_make': 100, 
        'num_shapelets': 10,
    },
    # 'init_config': {
    #     'size_ratio': [0.1, 0.2], 
    #     'num_shapelets': 10
    # },
    'model_config': {
        'epochs': 300, 
        'batch_size': 64, 
        'model_path': './model/best_model.pth',
        'step': 1,
        'lr': 1e-3, 
        'wd': 1e-4, 
        'epsilon': 1e-7,
        'k': 6,
        'l1': 1e-5,
        'l2': 0
    },
}

yaml_config = config
args = parser.parse_args()

report = []
yaml_files = glob.glob("./public_preprocessing/yaml_BOSS/*.yaml")
print(yaml_files)
# for config_path in yaml_files:
    # try:
    #     with open(config_path, "r") as file:
    #         yaml_config = yaml.safe_load(file)  # Read YAML file
    #         # if yaml_config:  # Ensure it's not empty
    #         #     update_config(config, yaml_config)  # Update the config
    # except FileNotFoundError:
    #     print(f"Warning: {config_path} not found. Using default config.")

for k in range(1):
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    val_loss_list = []
    elapsed_list = []
    for j in range(10):
        elapsed, results, val_loss = exp(
            yaml_config, datatype=args.datatype, 
            dataset=args.dataset, version=args.version
        )
        acc_list.append(results['accuracy'])
        precision_list.append(results['precision'])
        f1_list.append(results['f1_score'])
        recall_list.append(results['recall'])
        val_loss_list.append(val_loss)
        elapsed_list.append(elapsed)
    
    avg_acc = sum(acc_list) / len(acc_list)
    avg_prec = sum(precision_list) / len(precision_list)
    avg_f1 = sum(f1_list) / len(f1_list)
    avg_recall = sum(recall_list) / len(recall_list)
    avg_loss = sum(val_loss_list) / len(val_loss_list)
    avg_elapsed = sum(elapsed_list) / len(elapsed_list)

    print(f"Average accuracy: {avg_acc}")
    print(f"Average precision: {avg_prec}")
    print(f"Average f1-score: {avg_f1}")
    print(f"Average recall score: {avg_recall}")
    print(f"Average validation loss: {avg_loss}")

    result = {
        'avg_accuracy': avg_acc,
        'avg_f1': avg_f1,
        'avg_recall': avg_recall,
        'avg_precision': avg_prec,
        'avg_val_loss': avg_loss,
        'elapsed_time': avg_elapsed
    }
    # result['init_mode'] = config['init_mode']
    result['model_mode'] = yaml_config['model_mode']
    for key, value in yaml_config['data_loading'].items():
        result[f'data_{key}'] = value
    for key, value in yaml_config['init_config'].items():
        result[f'init_{key}'] = value
    for key, value in yaml_config['model_config'].items():
        result[f'model_{key}'] = value
    
    print(result)
    report.append(result)
    print("-----------------")
output_dir = f"./log/{args.dataset}/"
os.makedirs(output_dir, exist_ok=True)
save_results_to_csv(report, filename=os.path.join(output_dir, 'LS_Trans_regularization.csv'))