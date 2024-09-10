import cv2
import numpy as np
import os
from config import optical_flow_input_folder, optical_flow_output_file

def calculate_optical_flow(input_folder, output_file):
    # retrieves list of jpg's in input folder
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    prev_frame = None
    flow_data = [] # empty list to store the optical flow between data

    # calculates number of frames by length of frame_files
    total_frames = len(frame_files)
    print(f"Total frames: {total_frames}")

    # loops through in each frame w/ enumerate to track i and frame_file
    for i, frame_file in enumerate(frame_files):
        # constructs path to current frame file & reads as greyscale
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        # checks for first frame and skips calculation
        if prev_frame is None:
            prev_frame = frame
            continue

        # calculates the dense optical flow using the Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_data.append(flow) 
        prev_frame = frame

        # prints progress 10 frames at a time
        if i % 10 == 0:
            print(f"Processed {i}/{total_frames} frames")

    # saves as numpy array to output
    np.save(output_file, flow_data)
    print(f"Calculated optical flow for {len(flow_data)} frame pairs.")

if __name__ == "__main__":
    calculate_optical_flow(optical_flow_input_folder, optical_flow_output_file)
