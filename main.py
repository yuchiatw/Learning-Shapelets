import os
from exp_pipeline import exp
from utils.evaluation_and_save import save_results_to_csv
import argparse
parser = argparse.ArgumentParser()
# ------------------------------ Input and Output -------------------------------------
parser.add_argument('--datatype', type=str, default='private', choices={'public', 'private'})
parser.add_argument('--dataset', type=str, default='preterm')
parser.add_argument('--version', type=str, default='3')
parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio for data splitting')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio for data splitting')
parser.add_argument('--norm_std', type=str, default='standard', choices={'standard', 'minmax'}, help='Normalization standard')
parser.add_argument('--norm_mode', type=str, default='local_before', choices={'local_before', 'global_after'}, help='Normalization mode')
parser.add_argument('--seq_min', type=int, default=15, help='Minimum sequence length')
parser.add_argument('--pad_min', type=int, default=3, help='Minimum padding length')
parser.add_argument('--step_min', type=int, default=1, help='Minimum step size')
parser.add_argument('--init_mode', type=str, default='pips', choices={'pips', 'random'}, help='Initialization mode')
parser.add_argument('--model_mode', type=str, default='JOINT', choices={'JOINT', 'LS_FCN', 'LS_Transformer', 'BOSS'}, help='Model mode')
parser.add_argument('--ws_rate', type=float, default=0.1, help='Window size rate for initialization')
parser.add_argument('--num_pip', type=float, default=0.1, help='Number of PIPs for initialization')
parser.add_argument('--num_shapelets_make', type=int, default=100, help='Number of shapelets to make during initialization')
parser.add_argument('--num_shapelets', type=int, default=10, help='Number of shapelets to use in the model')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--model_path', type=str, default='./model/best_model.pth', help='Path to save the best model')
parser.add_argument('--step', type=int, default=1, help='Step size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay for the optimizer')
parser.add_argument('--epsilon', type=float, default=1e-7, help='Epsilon for numerical stability in the optimizer')
parser.add_argument('--k', type=int, default=6, help='Parameter k for the model')
parser.add_argument('--l1', type=float, default=1e-5, help='L1 regularization coefficient')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization coefficient')
parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file")
parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads for the model')
parser.add_argument('--d_model', type=int, default=8, help='Dimension of the model for the transformer')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the transformer model')
# --------------------------------------------------------------------------------------
# configuration loading
args = parser.parse_args()
config = { # default
    'data_loading': {
        'test_ratio': args.test_ratio,
        'val_ratio': args.val_ratio,
        'norm_std': args.norm_std,
        'norm_mode': args.norm_mode,
        'seq_min': args.seq_min,
        'pad_min': args.pad_min,
        'step_min': args.step_min,
    },
    'init_mode': args.init_mode, # 'pips' / 'random'
    'model_mode': args.model_mode, # 'JOINT' / 'LS_FCN' / 'LS_Transformer' / 'BOSS'
    'init_config': {
        'size_ratio': [0.1, 0.2], 
        'ws_rate': args.ws_rate,
        'num_pip': args.num_pip, 
        'num_shapelets_make': args.num_shapelets_make, 
        'num_shapelets': args.num_shapelets,
    },

    'model_config': {
        'joint_mode': 'concat',
        'nhead':args.nhead,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'shuffle': True,
        'epochs': args.epochs, 
        'batch_size': args.batch_size, 
        'model_path': './model/best_model.pth',
        'step': args.step,
        'lr': args.lr, 
        'wd': args.wd, 
        'epsilon': args.epsilon,
        'k': args.k,
        'l1': args.l1,
        'l2': args.l2
    },
}



report = []

acc_list = []
f1_list = []
recall_list = []
precision_list = []
val_loss_list = []
elapsed_list = []
for j in range(10):
    elapsed, results, val_loss = exp(
        config, datatype=args.datatype, 
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
result['model_mode'] = config['model_mode']
for key, value in config['data_loading'].items():
    result[f'data_{key}'] = value
for key, value in config['init_config'].items():
    result[f'init_{key}'] = value
for key, value in config['model_config'].items():
    result[f'model_{key}'] = value

print(result)
report.append(result)
print("-----------------")
output_dir = f"./log/{args.dataset}/"
os.makedirs(output_dir, exist_ok=True)
save_results_to_csv(report, filename=os.path.join(output_dir, 'LS_Trans_regularization.csv'))