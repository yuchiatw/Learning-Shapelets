from sklearn.metrics import accuracy_score, f1_score, \
    roc_auc_score, precision_score, recall_score
import csv

def eval_results(y, yhat, average='macro'):
    if len(yhat.shape) == 2:
        yhat = yhat.argmax(axis=1)
    
    accuracy = accuracy_score(y, yhat)
    
    # Handle multi-class with 'average' argument
    precision = precision_score(y, yhat, average=average)
    f1 = f1_score(y, yhat, average=average)
    recall = recall_score(y, yhat, average=average)
    
    return {
        'accuracy': accuracy,
        'precision': precision, 
        'f1_score':f1, 
        'recall': recall,
    }
def save_results_to_csv(results, filename="results.csv"):
        keys = results[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
            
# def save_results_to_csv(results, filename="results.csv"):
#     with open(filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Parameter", "Value"])  # Header
#         for key, value in results.items():
#             writer.writerow([key, value])
