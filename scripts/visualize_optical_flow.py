import cv2
import numpy as np
import os
from config import visualization_input_folder, visualization_flow_data_file, visualization_output_folder

def visualize_optical_flow(input_folder, flow_data_file, output_folder):
    # checks if output directory exists if not creates it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # load the optical flow data
    flow_data = np.load(flow_data_file, allow_pickle=True)
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

    # makes sure number of frames matches the number of flow data entries
    assert len(flow_data) == len(frame_files) - 1, "Number of flow data entries does not match number of frames."

    print(f"Number of frames: {len(frame_files)}")
    print(f"Number of flow data entries: {len(flow_data)}")

    # loops through frames and flow data
    for i, (flow, frame_file) in enumerate(zip(flow_data, frame_files[:-1])):
        # reads frame and gets dimensions
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        print(f"Processing frame {i}: {frame_file}")

        # loops through frame with stpe size 10 pixels in x and y to reduce flow vectors
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                fx, fy = flow[y, x]
                cv2.line(frame, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # saves output
        output_path = os.path.join(output_folder, f"flow_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved visualization to {output_path}")

    print(f"Visualized optical flow for {len(flow_data)} frames.")

if __name__ == "__main__":
    visualize_optical_flow(visualization_input_folder, visualization_flow_data_file, visualization_output_folder)




