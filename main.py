import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import plotly.graph_objects as go


st.set_page_config(page_title = 'Pose', layout = 'wide')
st.markdown("<h4 style='text-align: center;'>MeasureUp</h4>", unsafe_allow_html=True)
st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 1px !important;
    padding-bottom: 1px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

def create_full_keypoints(df_pose):
    imag = go.Figure()
    for slide in range(0, len(df_pose)):
        try:
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_11'].iloc[slide][0], x1 = df_pose['landmark_12'].iloc[slide][0], 
                        y0 = df_pose['landmark_11'].iloc[slide][1], y1 = df_pose['landmark_12'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_12'].iloc[slide][0], x1 = df_pose['landmark_14'].iloc[slide][0], 
                        y0 = df_pose['landmark_12'].iloc[slide][1], y1 = df_pose['landmark_14'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_14'].iloc[slide][0], x1 = df_pose['landmark_16'].iloc[slide][0], 
                        y0 = df_pose['landmark_14'].iloc[slide][1], y1 = df_pose['landmark_16'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_12'].iloc[slide][0], x1 = df_pose['landmark_11'].iloc[slide][0], 
                        y0 = df_pose['landmark_12'].iloc[slide][1], y1 = df_pose['landmark_11'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_11'].iloc[slide][0], x1 = df_pose['landmark_13'].iloc[slide][0], 
                        y0 = df_pose['landmark_11'].iloc[slide][1], y1 = df_pose['landmark_13'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_13'].iloc[slide][0], x1 = df_pose['landmark_15'].iloc[slide][0], 
                        y0 = df_pose['landmark_13'].iloc[slide][1], y1 = df_pose['landmark_15'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_23'].iloc[slide][0], x1 = df_pose['landmark_24'].iloc[slide][0], 
                        y0 = df_pose['landmark_23'].iloc[slide][1], y1 = df_pose['landmark_24'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_23'].iloc[slide][0], x1 = df_pose['landmark_25'].iloc[slide][0], 
                        y0 = df_pose['landmark_23'].iloc[slide][1], y1 = df_pose['landmark_25'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_25'].iloc[slide][0], x1 = df_pose['landmark_27'].iloc[slide][0], 
                        y0 = df_pose['landmark_25'].iloc[slide][1], y1 = df_pose['landmark_27'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_24'].iloc[slide][0], x1 = df_pose['landmark_26'].iloc[slide][0], 
                        y0 = df_pose['landmark_24'].iloc[slide][1], y1 = df_pose['landmark_26'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_26'].iloc[slide][0], x1 = df_pose['landmark_28'].iloc[slide][0], 
                        y0 = df_pose['landmark_26'].iloc[slide][1], y1 = df_pose['landmark_28'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_11'].iloc[slide][0], x1 = df_pose['landmark_23'].iloc[slide][0], 
                        y0 = df_pose['landmark_11'].iloc[slide][1], y1 = df_pose['landmark_23'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_12'].iloc[slide][0], x1 = df_pose['landmark_24'].iloc[slide][0], 
                        y0 = df_pose['landmark_12'].iloc[slide][1], y1 = df_pose['landmark_24'].iloc[slide][1],
                        line_color = 'grey', opacity = 0.3)
            for marker in df_pose.columns:
                if marker.startswith('landmark'):
                    imag.add_trace(go.Scatter(mode = 'markers', x = [df_pose[marker].iloc[slide][0]], y = [df_pose[marker].iloc[slide][1]], 
                                            marker = dict(color = color, size = 5, opacity = 0.5),showlegend = False))
        except:
            continue
    imag.update_layout(height = 900, width = 500)
    return imag
def plot_keypoints_3d(df_pose, time):
    # Define the body part names and indices
    BODY_PARTS = {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7,
        'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
        'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15,
        'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
        'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
        'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27,
        'right_ankle': 28, 'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
        'right_foot_index': 32
    }

    # Define the connections between body parts
    BODY_PART_CONNECTIONS = [
        (BODY_PARTS['left_shoulder'], BODY_PARTS['right_shoulder']),
        (BODY_PARTS['left_shoulder'], BODY_PARTS['left_elbow']),
        (BODY_PARTS['left_elbow'], BODY_PARTS['left_wrist']),
        (BODY_PARTS['left_shoulder'], BODY_PARTS['left_hip']),
        (BODY_PARTS['left_hip'], BODY_PARTS['left_knee']),
        (BODY_PARTS['left_knee'], BODY_PARTS['left_ankle']),
        (BODY_PARTS['right_shoulder'], BODY_PARTS['right_elbow']),
        (BODY_PARTS['right_elbow'], BODY_PARTS['right_wrist']),
        (BODY_PARTS['right_shoulder'], BODY_PARTS['right_hip']),
        (BODY_PARTS['right_hip'], BODY_PARTS['right_knee']),
        (BODY_PARTS['right_knee'], BODY_PARTS['right_ankle']),
        (BODY_PARTS['left_shoulder'], BODY_PARTS['right_hip']),
        (BODY_PARTS['right_shoulder'], BODY_PARTS['left_hip']),
        (BODY_PARTS['left_ankle'], BODY_PARTS['left_heel']),
        (BODY_PARTS['right_ankle'], BODY_PARTS['right_heel']),
        (BODY_PARTS['left_heel'], BODY_PARTS['left_foot_index']),
        (BODY_PARTS['right_heel'], BODY_PARTS['right_foot_index']),
        (BODY_PARTS['left_wrist'], BODY_PARTS['left_thumb']),
        (BODY_PARTS['left_wrist'], BODY_PARTS['left_pinky']),
        (BODY_PARTS['right_wrist'], BODY_PARTS['right_thumb']),
        (BODY_PARTS['right_wrist'], BODY_PARTS['right_pinky'])
    ]

    # Create a Plotly figure
    fig = go.Figure()

    for i, landmark in enumerate(df_pose.columns):
        if landmark.startswith('landmark_'):
            x = [df_pose[landmark].iloc[0][0]]
            y = [df_pose[landmark].iloc[0][1]]
            z = [df_pose[landmark].iloc[0][2]]
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                name=landmark.split('_')[-1],
                marker=dict(
                    size=4,
                    color='red',
                    opacity=0.8
                )
            ))
    # Add lines to connect the points
    for connection in BODY_PART_CONNECTIONS:
        try:
            x = [df_pose.iloc[0][connection[0] * 3], df_pose.iloc[0][connection[1] * 3]]
            y = [df_pose.iloc[0][connection[0] * 3 + 1], df_pose.iloc[0][connection[1] * 3 + 1]]
            z = [df_pose.iloc[0][connection[0] * 3 + 2], df_pose.iloc[0][connection[1] * 3 + 2]]
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                name='Connection',
                line=dict(
                    color='black',
                    width=2
                )
            ))
        except: 
            continue

    # Show the plot
    return fig


@st.cache(allow_output_mutation=True)
def extract_pose_keypoints(video_path):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_path.read())
    cap = cv2.VideoCapture(tfile.name)

    # Define mediapipe pose detection module
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the pose detection module
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Create a dataframe to store the pose keypoints
        df_pose = pd.DataFrame()

        # Create a list to store the images
        image_list = []

        # Define the drawing specifications for the landmarks
        landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=0)
        connection_drawing_spec = mp_drawing.DrawingSpec(color=(128, 128, 128), thickness=5)

        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
        capture_interval = int(frame_rate / 10)  # Capture a frame every second
        frame_count = 0

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if we have reached the end of the video
            if not ret:
                break

            frame_count += 1

            # Check if the frame count matches the capture interval
            if frame_count % capture_interval != 0:
                continue

            # Convert the frame to RGB and resize if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fx = 640
            fy = 480
            frame = cv2.resize(frame, (fx, fy))  # Adjust the size as needed

            # Process the frame to extract the pose keypoints
            results = pose.process(frame)

            # Extract the pose landmarks from the results
            landmarks = results.pose_landmarks

            # If landmarks are detected, draw them on the frame
            #if landmarks is not None:
                #annotated_frame = frame.copy()
                # Draw the landmarks on the frame
                #mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                        #                  landmark_drawing_spec=landmark_drawing_spec,
                        #                  connection_drawing_spec=connection_drawing_spec)
                # Add joint angles on the image
                # Define the joint indices
                #joint_indices = {'Left Shoulder': 12, 'Left Elbow': 14, 'Left Wrist': 16,
                #                'Right Shoulder': 11, 'Right Elbow': 13, 'Right Wrist': 15,
                #                'Left Hip': 23, 'Left Knee': 25, 'Left Ankle': 27,
                #                'Right Hip': 24, 'Right Knee': 26, 'Right Ankle': 28}
                ##joint_angles = {}
                #for joint, idx in joint_indices.items():
                #    if joint in joint_angles:
                 #       x, y = int(landmarks.landmark[idx].x * frame.shape[1]), int(landmarks.landmark[idx].y * frame.shape[0])
                 #       cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                 #       cv2.putText(frame, f'{joint}', (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #frame = cv2.addWeighted(frame, 0.6, annotated_frame, 0.4, 0)
            # Add the frame to the image list
            image_list.append(frame)

            # Create a dictionary to store the pose landmarks
            landmarks_dict = {}

            # If landmarks are detected, store them in the dictionary
            if landmarks is not None:
                for idx, landmark in enumerate(landmarks.landmark):
                    landmarks_dict[f'landmark_{idx}'] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

            # Add the landmarks to the dataframe
            df_pose = df_pose.append(landmarks_dict, ignore_index=True)

        # Convert the dataframe to seconds
        df_pose['Frame'] = df_pose.index / 10
        data_points = len(df_pose)
        time_interval = pd.Timedelta(seconds=0.1)

        df_pose['time'] = pd.date_range(start='00:00:00', periods=data_points, freq=time_interval)
        df_pose = df_pose.set_index('time')

    return df_pose, image_list




@st.cache(allow_output_mutation=True)
def calculate_joint_angles(df_pose):
    # Define the joint angle calculation function
    def get_joint_angle(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cosine_angle) 
        return np.degrees(angle)
    


    # Define the joint indices
    joint_indices = {'Left Shoulder': 11, 'Left Elbow': 13, 'Left Wrist': 15,
                     'Right Shoulder': 12, 'Right Elbow': 14, 'Right Wrist': 16,
                     'Left Hip': 24, 'Left Knee': 26, 'Left Ankle': 28,
                     'Right Hip': 23, 'Right Knee': 25, 'Right Ankle': 27}

    # Create a dataframe to store the joint angles
    df_joint_angles = pd.DataFrame(columns=list(joint_indices.keys()))

    # Loop through each second of the video
    for i in range(len(df_pose)):
        # Get the pose landmarks for the current second
        pose_landmarks = df_pose.iloc[i, :].values

        # Calculate the joint angles
        joint_angles = {}
        for joint, idx in joint_indices.items():
            try:
                if joint == 'Right Shoulder':
                    p1lbl = 24
                    p2lbl = 12
                    p3lbl = 14
                if joint == 'Right Elbow':
                    p1lbl = 12
                    p2lbl = 14
                    p3lbl = 16
                if joint == 'Right Wrist':
                    p1lbl = 14
                    p2lbl = 16
                    p3lbl = 20
                    
                if joint == 'Left Shoulder':
                    p1lbl = 23
                    p2lbl = 11
                    p3lbl = 13
                if joint == 'Left Elbow':
                    p1lbl = 11
                    p2lbl = 13
                    p3lbl = 15
                if joint == 'Left Wrist':
                    p1lbl = 13
                    p2lbl = 15
                    p3lbl = 19

                if joint == 'Right Hip':
                    p1lbl = 24
                    p2lbl = 23
                    p3lbl = 25
                if joint == 'Right Knee':
                    p1lbl = 23
                    p2lbl = 25
                    p3lbl = 27
                if joint == 'Right Ankle':
                    p1lbl = 25
                    p2lbl = 27
                    p3lbl = 32

                if joint == 'Left Hip':
                    p1lbl = 23
                    p2lbl = 24
                    p3lbl = 26
                if joint == 'Left Knee':
                    p1lbl = 24
                    p2lbl = 26
                    p3lbl = 28
                if joint == 'Left Ankle':
                    p1lbl = 26
                    p2lbl = 28
                    p3lbl = 31
                p1 = pose_landmarks[p1lbl][:3]
                p2 = pose_landmarks[p2lbl][:3]
                p3 = pose_landmarks[p3lbl][:3]
                angle = get_joint_angle(p1, p2, p3)

                joint_angles[joint] = angle
            except:
                joint_angles[joint] = np.nan

        # Add the joint angles to the dataframe
        df_joint_angles.loc[df_pose.index[i]] = joint_angles

    return df_joint_angles
@st.cache(allow_output_mutation=True)
def plot_joint_angles(df_joint_angles):
    # Plot joint angles over time using Plotly
    fig = px.line(df_joint_angles, x=df_joint_angles.index, y=df_joint_angles.columns,
                  labels={'x': 'Time', 'y': 'Joint Angle'}, title='Joint Angles over Time')
    return fig
@st.cache(allow_output_mutation=True)
def calculate_joint_angle_velocities(df_joint_angles):
    # Calculate the joint angle velocities
    df_joint_angle_velocities = df_joint_angles.diff().dropna()
    df_joint_angle_velocities.index = pd.to_datetime(df_joint_angle_velocities.index).time

    return df_joint_angle_velocities
@st.cache(allow_output_mutation=True)
def plot_joint_angles(df_joint_angles, df_joint_angle_velocities):
    # Plot joint angles over time using Plotly
    fig = px.line(df_joint_angles, x=df_joint_angles.index, y=df_joint_angles.columns,
                  labels={'x': 'Time', 'y': 'Joint Angle'}, title='Joint Angles over Time')

    for joint in df_joint_angle_velocities.columns:
        fig.add_trace(px.line(df_joint_angle_velocities, x=df_joint_angle_velocities.index, y=joint,
                               labels={'x': 'Time', 'y': 'Joint Angle Velocity'}, title='Joint Angle Velocities over Time').data[0])

    return fig

def display_joints(df_pose, df_joint_angles, key_arr, slide, line_color = 'grey'):
    if slide == None:
        imag = px.imshow(key_arr)
    else:
        imag = px.imshow(key_arr[slide])
    imag.update_yaxes(visible=False, showticklabels=False)
    imag.update_xaxes(visible=False, showticklabels=False)
    imag.update_layout(height = 550)
    try:
        if df_pose['landmark_11'].iloc[slide][3] > 0.5 and df_pose['landmark_12'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                        x0 = df_pose['landmark_11'].iloc[slide][0]*640, x1 = df_pose['landmark_12'].iloc[slide][0]*640, 
                        y0 = df_pose['landmark_11'].iloc[slide][1]*480, y1 = df_pose['landmark_12'].iloc[slide][1]*480,
                        line_color = line_color, opacity = 0.3)
        if df_pose['landmark_12'].iloc[slide][3] > 0.5 and df_pose['landmark_14'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_12'].iloc[slide][0]*640, x1 = df_pose['landmark_14'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_12'].iloc[slide][1]*480, y1 = df_pose['landmark_14'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_14'].iloc[slide][3] > 0.5 and df_pose['landmark_16'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_14'].iloc[slide][0]*640, x1 = df_pose['landmark_16'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_14'].iloc[slide][1]*480, y1 = df_pose['landmark_16'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_12'].iloc[slide][3] > 0.5 and df_pose['landmark_11'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_12'].iloc[slide][0]*640, x1 = df_pose['landmark_11'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_12'].iloc[slide][1]*480, y1 = df_pose['landmark_11'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_11'].iloc[slide][3] > 0.5 and df_pose['landmark_13'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_11'].iloc[slide][0]*640, x1 = df_pose['landmark_13'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_11'].iloc[slide][1]*480, y1 = df_pose['landmark_13'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_13'].iloc[slide][3] > 0.5 and df_pose['landmark_15'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_13'].iloc[slide][0]*640, x1 = df_pose['landmark_15'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_13'].iloc[slide][1]*480, y1 = df_pose['landmark_15'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_23'].iloc[slide][3] > 0.5 and df_pose['landmark_24'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_23'].iloc[slide][0]*640, x1 = df_pose['landmark_24'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_23'].iloc[slide][1]*480, y1 = df_pose['landmark_24'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_23'].iloc[slide][3] > 0.5 and df_pose['landmark_25'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_23'].iloc[slide][0]*640, x1 = df_pose['landmark_25'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_23'].iloc[slide][1]*480, y1 = df_pose['landmark_25'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_25'].iloc[slide][3] > 0.5 and df_pose['landmark_27'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_25'].iloc[slide][0]*640, x1 = df_pose['landmark_27'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_25'].iloc[slide][1]*480, y1 = df_pose['landmark_27'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_24'].iloc[slide][3] > 0.5 and df_pose['landmark_26'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_24'].iloc[slide][0]*640, x1 = df_pose['landmark_26'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_24'].iloc[slide][1]*480, y1 = df_pose['landmark_26'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_26'].iloc[slide][3] > 0.5 and df_pose['landmark_28'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_26'].iloc[slide][0]*640, x1 = df_pose['landmark_28'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_26'].iloc[slide][1]*480, y1 = df_pose['landmark_28'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_11'].iloc[slide][3] > 0.5 and df_pose['landmark_23'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_11'].iloc[slide][0]*640, x1 = df_pose['landmark_23'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_11'].iloc[slide][1]*480, y1 = df_pose['landmark_23'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_12'].iloc[slide][3] > 0.5 and df_pose['landmark_24'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_12'].iloc[slide][0]*640, x1 = df_pose['landmark_24'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_12'].iloc[slide][1]*480, y1 = df_pose['landmark_24'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_28'].iloc[slide][3] > 0.5 and df_pose['landmark_32'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_28'].iloc[slide][0]*640, x1 = df_pose['landmark_32'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_28'].iloc[slide][1]*480, y1 = df_pose['landmark_232'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_27'].iloc[slide][3] > 0.5 and df_pose['landmark_31'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_27'].iloc[slide][0]*640, x1 = df_pose['landmark_31'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_27'].iloc[slide][1]*480, y1 = df_pose['landmark_31'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_16'].iloc[slide][3] > 0.5 and df_pose['landmark_20'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_16'].iloc[slide][0]*640, x1 = df_pose['landmark_20'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_16'].iloc[slide][1]*480, y1 = df_pose['landmark_20'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
        if df_pose['landmark_15'].iloc[slide][3] > 0.5 and df_pose['landmark_19'].iloc[slide][3] > 0.5:
            imag.add_shape(type = 'line', 
                    x0 = df_pose['landmark_15'].iloc[slide][0]*640, x1 = df_pose['landmark_19'].iloc[slide][0]*640, 
                    y0 = df_pose['landmark_15'].iloc[slide][1]*480, y1 = df_pose['landmark_19'].iloc[slide][1]*480,
                    line_color = line_color, opacity = 0.3)
    except:
        pass
    #for marker in df_pose.columns:
        #try:
            #if marker.startswith('landmark'):
                #if df_pose[marker].iloc[slide][3] > 0.5:
                    #imag.add_trace(go.Scatter(mode = 'markers', hovertext = marker + " " + str(round(df_pose[marker].iloc[slide][3], 2)) + " confidence", x = [df_pose[marker].iloc[slide][0]*640], y = [df_pose[marker].iloc[slide][1]*480], 
                             #           marker = dict(color = 'lightgrey', size = 1), showlegend = False))
       # except:
           # continue
    keypoints = {
        'landmark_12': 'Right Shoulder',
        'landmark_11': 'Left Shoulder',
        'landmark_23': 'Left Hip',
        'landmark_24': 'Right Hip',
        'landmark_14': 'Right Elbow',
        'landmark_13': 'Left Elbow',
        'landmark_15': 'Left Wrist',
        'landmark_16': 'Right Wrist',
        'landmark_26': 'Right Knee',
        'landmark_25': 'Left Knee',
        'landmark_28': 'Right Ankle',
        'landmark_27': 'Left Ankle'
    }

    for col in df_pose.columns:
        try:
            if col.startswith('landmark') and col.endswith(('11', '12', '24', '23', '14', '13', '16', '15', '26', '25', '28', '27')):
                if df_pose[col].iloc[slide][3] > 0.5:
                    text = str(round(df_joint_angles[keypoints[col]].iloc[slide], 2))
                    imag.add_annotation(text=text, 
                                        x=df_pose[col].iloc[slide][0] * 640 + 10, y=df_pose[col].iloc[slide][1] * 480 + 10, 
                                        showarrow=False, font=dict(size=8, color='white'),
                                        hovertext = keypoints[col] + " : " + text + " Degrees")
        except:
            continue
    return imag

def create_joint_line_plot(df_joint_angles, slide):
    joint_line_plot = px.line(df_joint_angles, y = jnt) # color_discrete_map={'Right Shoulder': '#ff0000', 'Right Elbow': '#b30000', 'Right Wrist': '#ff6600' }
    joint_line_plot.update_layout(height = 450, hovermode="x")
    joint_line_plot.update_xaxes(tickformat="%H:%M:%S", title = 'Seconds')
    joint_line_plot.update_yaxes(range=[0,190], title = 'Angle')
    joint_line_plot.add_vline(x = df_joint_angles['time'].iloc[slide])
    return joint_line_plot

def create_joint_velocity_plot(df_joint_angles, slide):
    df_joint_angles['time'] = df_joint_angles.index
    joint_velocity_plot = px.area(df_joint_angles.diff(10).abs(), y = jnt)# color_discrete_map={'Right Shoulder': '#ff0000', 'Right Elbow': '#b30000', 'Right Wrist': '#ff6600' }
    joint_velocity_plot.update_layout(height = 450, hovermode="x")
    joint_velocity_plot.update_xaxes(tickformat="%H:%M:%S", title = '')
    joint_velocity_plot.update_yaxes(range=[0,300], title = 'Velocity (r/s)')
    joint_velocity_plot.add_vline(x = df_joint_angles['time'].iloc[slide])
    return joint_velocity_plot



# Upload a video
video_file = st.sidebar.file_uploader("Upload a video")
with st.sidebar.expander("Settings"):
    line_color = st.color_picker("Lines")


if video_file is not None:
    # Process the video to extract pose keypoints
    df_pose, key_arr = extract_pose_keypoints(video_file)
    # Calculate joint angles
    df_joint_angles = calculate_joint_angles(df_pose)
    # Perform exponential weighted mean on joint angles to smooth data
    df_joint_angles = df_joint_angles.ewm(com=1.5, adjust = False).mean()
    # Slider to display specific time of values
    if 'slide_value' not in st.session_state:
        st.session_state['slide_value'] = 0.0
    # Retrieve image of data in time and display joints, lines, and agnles
    imag = display_joints(df_pose, df_joint_angles, key_arr, slide = int(st.session_state['slide_value'] * 10), line_color = line_color)
    # Display joints, lines, and angles with streamlit
    st.plotly_chart(imag, use_container_width=True)
    slide = st.slider("Time", min_value = 0.0, max_value = df_pose['Frame'].max(), step = 0.1,  key = 'side1', value = float(st.session_state['slide_value']))
    st.session_state['slide_value'] = slide
    c1, c2, c3 = st.columns(3)
    prev = c1.button("⏮️ Previous", use_container_width=True)
    if prev:
        st.session_state['slide_value']  = st.session_state['slide_value'] - 0.1
    next = c2.button("Next ⏭️", use_container_width=True)
    if next:
        st.session_state['slide_value']  = st.session_state['slide_value'] + 0.1
    play = st.button("Play")
    if play:
        for i in range(len(df_pose)):
            st.session_state['slide_value'] = st.session_state['slide_value']

    # Select joint to plot
    jnt = st.multiselect('Joint', options = df_joint_angles.columns)
    # Initate columns
    column1, column2 = st.columns(2)
    
    # Create joint line plot
    df_joint_angles['time'] = df_joint_angles.index
    joint_line_plot = create_joint_line_plot(df_joint_angles, slide = int(st.session_state['slide_value'] * 10))
    # Create joint velocity plot
    joint_velocity_plot = create_joint_velocity_plot(df_joint_angles, slide = int(st.session_state['slide_value'] * 10))
    # Show joint line plot
    column1.plotly_chart(joint_line_plot, use_container_width=True)
    # Show joint velocity plot
    column2.plotly_chart(joint_velocity_plot, use_container_width=True)
    #ploter = create_full_keypoints(df_pose)
    #st.plotly_chart(ploter, use_container_width=True)
    st.dataframe(df_joint_angles.describe(), use_container_width=True)
