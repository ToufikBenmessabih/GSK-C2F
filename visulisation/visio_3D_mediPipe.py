import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video
video_path = 'IKEA_ASM/video.avi'
cap = cv2.VideoCapture(video_path)

# Get the video frame dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize drawing utility
mp_drawing = mp.solutions.drawing_utils

# Create a matplotlib figure for 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def draw_3d_skeleton(landmarks):
    # Clear the previous plot
    ax.cla()

    # Get x, y, z coordinates for each landmark
    x_vals = [landmark.x for landmark in landmarks.landmark]
    y_vals = [landmark.y for landmark in landmarks.landmark]
    z_vals = [landmark.z for landmark in landmarks.landmark]

    # Convert from normalized to pixel coordinates (optional, for better visualization)
    x_vals = np.array(x_vals) * frame_width
    y_vals = np.array(y_vals) * frame_height
    z_vals = np.array(z_vals) * frame_width  # Adjust based on video scale


    # Plot the 3D points (joints)
    ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')

    # Set plot limits so that x and y both start from the same origin (0, 0)
    ax.set_xlim([0, frame_width])  # X-axis starts from 0
    ax.set_ylim([0, frame_height])  # Y-axis starts from 0
    ax.set_zlim([-frame_width / 2, frame_width / 2])  # Adjust Z-axis range based on video scale
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Force the aspect ratio of x and y to be equal
    ax.set_box_aspect([frame_width, frame_height, frame_width / 2])

    # Draw arrows (connections) between the joints based on POSE_CONNECTIONS
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        # Get the start and end points for each connection
        start_point = (x_vals[start_idx], y_vals[start_idx], z_vals[start_idx])
        end_point = (x_vals[end_idx], y_vals[end_idx], z_vals[end_idx])

        # Plot an arrow from start to end
        ax.quiver(start_point[0], start_point[1], start_point[2], 
                  end_point[0] - start_point[0], 
                  end_point[1] - start_point[1], 
                  end_point[2] - start_point[2],
                  color='blue', arrow_length_ratio=0.1)

    # Update the plot
    plt.draw()
    plt.pause(0.001)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(frame_rgb)
    
    # Extract and visualize the 2D skeleton
    if results.pose_landmarks:
        # Draw the 2D skeleton on the video frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Visualize the 3D skeleton with arrows (connections)
        draw_3d_skeleton(results.pose_landmarks)
    
    # Display the frame with the 2D skeleton
    cv2.imshow('Video with 2D Skeleton', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
