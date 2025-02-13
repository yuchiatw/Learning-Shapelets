import csv

# Define the results
results = {
    'num_epochs': args.num_epochs,
    'batch_size': args.batch_size, 
    'nhead':args.nhead, 
    'num_layers': args.num_layers,
    'lr': args.lr, 
    'wd': args.wd, 
    'epsilon': args.epsilon, 
    'dist_measure': args.dist_measure,
    'num_shapelets_ratio': args.num_shapelets_ratio,
    'size_ratio': args.size_ratio, 
    'step': args.step, 
    'kmeans_init': args.kmeans_init,
    'accuracy': accuracy, 
    'elapsed_time': elapsed_time,
}

# Specify the CSV file path
csv_file_path = '/home/yuchia/Learning-Shapelets/preterm/results.csv'

# Write the results to the CSV file
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=results.keys())
    
    # Write the header only if the file is empty
    if file.tell() == 0:
        writer.writeheader()
    
    writer.writerow(results)