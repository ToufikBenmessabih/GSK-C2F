import pandas as pd
import os

# Replace 'your_directory' with the actual directory containing your CSV files
directory_path = 'csv_dataset/60fps'

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Video', 'Frame_Count', 'FPS', 'Seconds', 'split'])

# Get a list of all CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

# Iterate over each CSV file in the directory
for csv_file in csv_files:
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(directory_path, csv_file)

    # Read the CSV file into a DataFrame using the specified encoding
    df = pd.read_csv(csv_file_path, delimiter=';', encoding='latin1')

    frames = len(df)
    fps = 60
    sec = round(frames / fps,2)
    

    # Get the file name without '.csv'
    file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    print(file_name)
    split = 'train'


    # Append the values to the result DataFrame
    result_df = result_df.append({'Video': file_name, 'Frame_Count': frames, 'FPS': fps, 'Seconds': sec, 'split': split},
                                 ignore_index=True)

# Save the result DataFrame to a new CSV file
result_csv_path = os.path.join(directory_path, 'inhard_3_info_60fps.csv')
result_df.to_csv(result_csv_path, index=False)

# Print the result CSV path
print(f"Result summary saved to: {result_csv_path}")
