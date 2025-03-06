import json
import yaml

# Path to input JSON and output YAML file
json_file_path = "test.json"
yaml_file_path = "test.yaml"

# Read JSON file
with open(json_file_path, "r") as json_file:
    config_data = json.load(json_file)

# Convert to YAML and save
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(config_data, yaml_file, default_flow_style=False, sort_keys=False)

print(f"YAML file saved: {yaml_file_path}")