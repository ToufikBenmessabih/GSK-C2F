import json
import os
import csv

def get_frame_count(file_name):
    with open('bvh_info.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(row['Video'])
            if row['Video'] == file_name:
                nbr_frames = int(row['Frame_Count'])
                return nbr_frames
    
    # Return a default value or handle the case when the file name is not found
    return None

def generate_annotations(file_path, file_name):

      # Read predictions from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        key = list(data.keys())[0]

    # Create a dictionary to store annotations by seconds
    annotations_by_frames = {}

    nbr_frames = int(get_frame_count(file_name))
    for frame in range(nbr_frames):
        annotations_by_frames[frame] = {'label': 'No action'} #init with BG

    # Iterate through predictions
    for prediction in data[key]:
        start_time = prediction['Action_start_bvh_frame']
        end_time = prediction['Action_end_bvh_frame']
        label = prediction['label']
        
        
        for frame in range(start_time, end_time + 1):
            annotations_by_frames[frame] = {'label': label}

    # Convert the dictionary to the desired result format
    result = []
    for frame, annotation in sorted(annotations_by_frames.items()):
        result.append(f"{annotation['label']}")

    return result

def write_2_file(result, output_file):
    with open(output_file, 'w') as file:
        for annotation in result:
            file.write(annotation + '\n')



# Directory containing the text files
directory_path = './GT_bvh'
out_directory_path = './bvh_30fps'

# Iterate over all files in the directory
if os.path.exists(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            file_name = os.path.splitext(filename)[0]

            result = generate_annotations(file_path, file_name)

            output_path = os.path.join(out_directory_path, file_name + '.txt')
            write_2_file(result, output_path)

