import json
import os

# Load the JSON file
with open('anno_bvh.json', 'r') as f:
    data = json.load(f)

# Create a directory to store the JSON files
output_directory = 'GT_bvh'
os.makedirs(output_directory, exist_ok=True)

# Iterate over each key in the JSON data
for key in data.keys():
    # Construct the filename from the key
    filename = os.path.join(output_directory, f"{key}.json")

    # Write the corresponding value to a separate JSON file
    with open(filename, 'w') as f:
        json.dump({key: data[key]}, f, indent=4)
