import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())
import time

import tsfel
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # classifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import log_loss

from preterm_preprocessing.preterm_preprocessing import preterm_pipeline
from public_preprocessing.public_preprocessing import public_pipeline

from utils.evaluation_and_save import eval_results, save_results_to_csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import argparse

names = [
    # "Nearest Neighbors",
    # "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    # "Random Forest",
    # "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

def feature_engineering(config, dataset, clf = DecisionTreeClassifier(), datatype='public', version: str = '', store_results = False):
    
    root = './data'
    data_path = os.path.join(root, f'{dataset}.npz')
    meta_path = os.path.join(root,'filtered_clinical_data.csv')
    strip_path = os.path.join(root, 'filtered_strips_data.json')
    if len(version) > 0 and datatype == 'private':
        data_path = os.path.join(root, f'{dataset}_v{version}.npz')
        meta_path = os.path.join(root, f'filtered_clinical_data_v{version}.csv')
        strip_path = os.path.join(root, f'filtered_strips_data_v{version}.json')
    if os.path.exists(data_path):
        data = np.load(data_path)
    elif datatype == 'private':
        data = preterm_pipeline(
            config=config['data_loading'], 
            meta_path=meta_path, 
            strip_path=strip_path,
            data_path=data_path
        )
    else:
        data = public_pipeline(
            dataset=dataset, 
            output=store_results, 
            root=root,
            config=config['data_loading'], 
        )
    
    
    x_train = data['X_train'].transpose(0, 2, 1)
    x_val = data['X_val'].transpose(0, 2, 1)
    x_train = np.concatenate((x_train, x_val), axis=0)
    x_test = data['X_test'].transpose(0, 2, 1)
    y_train = data['y_train']
    y_val = data['y_val']
    y_train = np.concatenate((y_train, y_val), axis=0)
    print(y_train)
    y_test = data['y_test']
    
    model_config = config['model_config']
    
    t1 = time.time()
    cfg_file = tsfel.get_features_by_domain()
    X_train = tsfel.time_series_features_extractor(cfg_file, x_train)
    X_val = tsfel.time_series_features_extractor(cfg_file, x_val)
    X_test = tsfel.time_series_features_extractor(cfg_file, x_test)
    
    corr_features, X_train = tsfel.correlated_features(X_train, drop_correlated=True)
    X_val.drop(corr_features, axis=1, inplace=True)
    X_test.drop(corr_features, axis=1, inplace=True)
    selector = VarianceThreshold(threshold=model_config['threshold'])
    X_train = selector.fit_transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    scaler = preprocessing.StandardScaler()
    nX_train = scaler.fit_transform(X_train)
    nX_val = scaler.transform(X_val)
    nX_test = scaler.transform(X_test)
    classifier = clf
    elapsed = time.time() - t1
    # Train the classifier
    classifier.fit(nX_train, y_train)

    # Predict on test data
    y_val_pred = classifier.predict(nX_val)
    y_predict = classifier.predict(nX_test)

    # Get the classification report
    # val_loss = log_loss(y_val, y_val_pred)
    results = eval_results(y_test, y_predict)
    
    
    return elapsed, results
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with specified datatype and dataset.")
    parser.add_argument('--datatype', type=str, default='public', choices=['public', 'private'], help='Type of data to use')
    parser.add_argument('--dataset', type=str, default='ECG5000', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default="LS_FCN", help='Batch size for training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    datatype = args.datatype
    dataset = args.dataset
    batch_size= args.batch_size
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
        'model_config': {
            'threshold': 0,
        },
    }
    report = []
    acc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    val_loss_list = []
    elapsed_list = []
    for name, clf in zip(names, classifiers):
        for i in range(10):
            elapsed, results = feature_engineering(config, dataset, clf=clf, datatype=datatype, version='3')
            acc_list.append(results['accuracy'])
            precision_list.append(results['precision'])
            f1_list.append(results['f1_score'])
            recall_list.append(results['recall'])
            # val_loss_list.append(val_loss)
            elapsed_list.append(elapsed)
        avg_acc = sum(acc_list) / len(acc_list)
        avg_prec = sum(precision_list) / len(precision_list)
        avg_f1 = sum(f1_list) / len(f1_list)
        avg_recall = sum(recall_list) / len(recall_list)
        # avg_loss = sum(val_loss_list) / len(val_loss_list)
        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        result = {
            'avg_accuracy': avg_acc,
            'avg_f1': avg_f1,
            'avg_recall': avg_recall,
            'avg_precision': avg_prec,
            # 'avg_val_loss': avg_loss,
            'elapsed_time': avg_elapsed,
            'model': name,
        }
        print(result)
        for key, value in config['data_loading'].items():
            result[f'data_{key}'] = value
        for key, value in config['model_config'].items():
                result[f'model_{key}'] = value
        report.append(result)
    output_dir = f"./log/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    save_results_to_csv(report, filename=os.path.join(output_dir, 'FE.csv'))