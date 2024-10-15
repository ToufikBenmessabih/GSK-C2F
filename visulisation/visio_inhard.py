import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
csv_path = 'InHARD-13/P09_R02_21.csv'
data = pd.read_csv(csv_path)

# Extract the joint positions (assuming 21 joints, each with X, Y, Z coordinates)
n_joints = 21

# Assuming the first 3 columns for each joint are X, Y, Z, the joint names would be in the format "JointX", "JointY", "JointZ"
# Modify this based on how the column names appear in your file
joint_names = [col.split('_')[0] for col in data.columns[:n_joints*3:3]]  # Assuming "JointX" format, only take one per joint

# Extract joint coordinates for the first frame (row)
joints = data.iloc[0, :n_joints*3].values.reshape(n_joints, 3)  # First row for example

# The neighbor links (provided)
neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), 
                 (4, 5), (5, 6), (6, 7), (7, 8), 
                 (4, 9), (9, 10), (10, 11), (11, 12),
                 (4, 13), (13, 14), 
                 (0, 15), (15, 16), (16, 17),
                 (0, 18), (18, 19), (19, 20)]

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the joints as scatter points
ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o', s=50, label='Joints')

# Label each joint with its name from the CSV
for i in range(n_joints):
    ax.text(joints[i, 0], joints[i, 1], joints[i, 2], joint_names[i], color='black')

# Plot the connections using arrows
for (start, end) in neighbor_link:
    ax.quiver(joints[start, 0], joints[start, 1], joints[start, 2], 
              joints[end, 0] - joints[start, 0], 
              joints[end, 1] - joints[start, 1], 
              joints[end, 2] - joints[start, 2], 
              color='b', arrow_length_ratio=0.1)

# Labels and view adjustments
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Skeleton Visualization with Joint Labels')
plt.show()