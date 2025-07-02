import argparse
import os
from exp_pipeline import exp
from utils.evaluation_and_save import save_results_to_csv
import torch
torch.cuda.set_device(0)

def ablation_experiment(config):
    datatype = 'public'
    dataset = 'Strawberry'

    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    val_loss_list = []
    elapsed_list = []
    for j in range(10):
        elapsed, results, val_loss, _ = \
            exp(config, datatype=datatype, dataset=dataset, version='', store_results=False)
        acc_list.append(results['accuracy'])
        precision_list.append(results['precision'])
        f1_list.append(results['f1_score'])
        recall_list.append(results['recall'])
        val_loss_list.append(val_loss)
        elapsed_list.append(elapsed)
        print(results)
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
        'dataset': dataset,
        'avg_accuracy': avg_acc,
        'avg_f1': avg_f1,
        'avg_recall': avg_recall,
        'avg_precision': avg_prec,
        'avg_val_loss': avg_loss,
        'elapsed_time': avg_elapsed
    }
    # result['init_mode'] = config['init_mode']
    result['model_mode'] = config['model_mode']
    for key, value in config['data_loading'].items():
        result[f'data_{key}'] = value
    for key, value in config['init_config'].items():
        result[f'init_{key}'] = value
    for key, value in config['model_config'].items():
        result[f'model_{key}'] = value
    
    return result

def generate_config(base_config, param_changes):
        """
        Generate a new configuration by applying parameter changes to the base configuration.
        :param base_config: The base configuration dictionary.
        :param param_changes: A dictionary of parameter changes to apply.
        :return: A new configuration dictionary with the changes applied.
        """
        new_config = base_config.copy()
        for key, changes in param_changes.items():
            if key in new_config:
                new_config[key].update(changes)
            else:
                new_config[key] = changes
        return new_config
if __name__ == "__main__":
   
    config = { # default
        'data_loading': {
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'minmax',
        },
        'init_mode': 'pips',
        'model_mode': 'JOINT',
        'init_config': {
            'ws_rate': 0.1,
            'num_pip': 0.2, 
            'num_shapelets_make': 500, 
            'num_shapelets': 30,
        },
        'model_config': {
            'window_size': 10,
            'n_bins': 4,
            'epochs': 1000, 
            'batch_size': 128,
            'model_path': f'./model/best_model_ablation.pth',
            'joint_mode': 'concat', # concat / fusion
            'step': 10,
            'lr': 5e-4, 
            'wd': 1e-4, 
            'd_model': 8,
            'nhead':2,
            'num_layers': 2, 
            'epsilon': 1e-7,
            'shuffle': True,
            'k': 0,
            'l1': 0,
            'l2': 0
        },
    }
    report = []
    # Define the parameters to adjust for the ablation study
    param_combinations = [
        {'model_config': {'batch_size': 256}},
        {'init_config': {'num_shapelets': 50}},  # Increase number of shapelets to make
    ]

    # Perform the ablation study
    for param_change in param_combinations:
        adjusted_config = generate_config(config, param_change)
        result = ablation_experiment(adjusted_config)
        report.append(result)

    # result = ablation_experiment(config)
    # report.append(result)
    print("-----------------")
    output_dir = f"./log/ablation_study/"
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_csv(report, filename=os.path.join(output_dir, 'performance_d3.csv'))