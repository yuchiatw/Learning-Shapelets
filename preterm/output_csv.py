import csv
import re

log_file_path = 'log/preterm_2.log'
csv_file_path = 'output.csv'

# Regular expression to match the log entries
log_entry_pattern = re.compile(r'Namespace\((.*?)\)')
accuracy_pattern = re.compile(r'Accuracy:\s*(\d+\.\d+)')
time_pattern = re.compile(r'Total elapsed time:\s*(\d+\.\d+) seconds')
separator_pattern = re.compile(r'------------------------------------------')

# Read the log file and parse the entries
log_entries = []
with open(log_file_path, 'r') as log_file:
    current_entry = {}
    for line in log_file:
        if separator_pattern.match(line):
            if current_entry:
                log_entries.append(current_entry)
                current_entry = {}
            continue
        namespace_match = log_entry_pattern.search(line)
        if namespace_match:
            namespace_content = namespace_match.group(1)
            entries = dict(item.split('=') for item in namespace_content.split(', '))
            current_entry.update(entries)
        accuracy_match = accuracy_pattern.search(line)
        if accuracy_match:
            current_entry['Accuracy'] = accuracy_match.group(1)
        time_match = time_pattern.search(line)
        if time_match:
            current_entry['Total elapsed time'] = time_match.group(1)
    if current_entry:
        log_entries.append(current_entry)

# Write the parsed entries to a CSV file
print(log_entries)
if log_entries:
    fieldnames = log_entries[0].keys()
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(log_entries)

print(f'Log entries have been written to {csv_file_path}')
