import pandas as pd
import json

# Read the CSV file into a DataFrame
df = pd.read_csv("InHARD.csv")

# Initialize an empty dictionary to store the desired information
data_dict = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    filename = row['File']
    start_frame = row['Action_start_bvh_frame']
    end_frame = row['Action_end_bvh_frame']
    label = row['Meta_action_label']
    idx = row['Meta_action_class_number']
    
    
    # Construct a dictionary with filename as key and start/end frames as values
    if filename not in data_dict:
        data_dict[filename] = []
    data_dict[filename].append({'Action_start_bvh_frame': start_frame, 'Action_end_bvh_frame': end_frame, 'label': label, 'class_idx': idx})

# Save the dictionary as a JSON file
with open('anno_bvh.json', 'w') as f:
    json.dump(data_dict, f, indent=4)
