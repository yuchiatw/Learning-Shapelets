import argparse
import os
from exp_functions_synthetic import exp
from utils.evaluation_and_save import save_results_to_csv
def synthetic_experiment(num_samples, time_step, window_size=None, window_step=None):
    datatype = 'synthetic'
    dataset = f'synthetic_{num_samples}_{time_step}'
    config = { # default
        'data_loading': {
            'time_step': time_step,
            'num_samples': num_samples,
            'test_ratio': 0.2,
            'val_ratio': 0.2,
            'norm_std': 'minmax',
        },
        'init_mode': 'pips',
        'model_mode': 'JOINT',
        'init_config': {
            'ws_rate': 0.1,
            'num_pip': 0.1, 
            'num_shapelets_make': 500, 
            'num_shapelets': 50,
        },
        'model_config': {
            'window_size': window_size,
            'window_step': window_step,  # 10% of time_step
            'n_bins': 4,
            'epochs': 500, 
            'batch_size': 64,
            'model_path': f'./model/best_model_{num_samples}_{time_step}.pth',
            'joint_mode': 'concat', # concat / fusion
            'step': window_step, 
            'lr': 1e-3, 
            'wd': 1e-4, 
            'nhead': 2, 
            'd_model': 8,
            'num_layers': 2, 
            'epsilon': 1e-7,
            'shuffle': True,
            'k': 0,
            'l1': 0,
            'l2': 0
        },
    }

    max_shape_time = 0
    elapsed_list = []
    
    elapsed, shapetime= \
        exp(config, datatype=datatype, dataset=dataset, version='4', store_results=False)

    return elapsed, shapetime
if __name__ == "__main__":
    report = []
    wsz = 0.2
    wst = 0.01
    
    for num_samples in [10000]:
        for time_step in [300, 500, 1000]:
            print(f"Running experiment for num_samples={num_samples}, time_step={time_step}")
            window_size = int(wsz * time_step)
            window_step = int(wst * time_step)
            print(window_size, window_step)
            avg_elapsed, max_shape_time = synthetic_experiment(num_samples=num_samples, time_step=time_step, 
                                                               window_size=window_size, window_step=window_step)
            print(f"Average elapsed time: {avg_elapsed:.2f} seconds")
            result = {
                'num_samples': num_samples,
                'time_step': time_step,
                'window_size': window_size,
                'window_step': window_step,
                'elapsed_time': avg_elapsed, 
                'shape_time': max_shape_time
            }
            report.append(result)
            output_dir = f"./log/synthetic"
            os.makedirs(output_dir, exist_ok=True)
            save_results_to_csv(report, filename=os.path.join(output_dir, f'time_temp_4.csv'))

    print("-----------------")
    output_dir = f"./log/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_csv(report, filename=os.path.join(output_dir, f'time.csv'))