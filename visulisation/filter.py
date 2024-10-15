import pandas as pd

# Load the CSV
file_path = 'InHARD-13/P09_R02.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# List of joints you're interested in
selected_joints = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 
                   'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'Neck', 'Head',
                   'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot']

# Build the column names for the selected joints (Xposition, Yposition, Zposition for each joint)
selected_columns = []
for joint in selected_joints:
    selected_columns.extend([f"{joint}_Xposition", f"{joint}_Yposition", f"{joint}_Zposition"])

# Filter the DataFrame
df_selected = df[selected_columns]

# Now you can work with the reduced DataFrame
print(df_selected.head())  # Just to check the first few rows

# Save the filtered data
df_selected.to_csv('P09_R02_21.csv', index=False)
