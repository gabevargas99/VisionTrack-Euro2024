import cv2
import os
from config import video_paths, output_dir 

def extract_frames(video_path, output_dir):
    # extract base name of video file & open video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    # retrieves total numbers of frames & converts to int
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # creates output directory if it doesnt exist
    os.makedirs(output_dir, exist_ok=True)
    
    # initialize frame counter and loop while video is successfully opened
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: # checks if video ends
            break
        # saves frame as image in directory and increases counter 1
        cv2.imwrite(os.path.join(output_dir, f"{video_name}_frame_{count}.jpg"), frame)
        count += 1
    # closes video file
    cap.release()

if __name__ == "__main__":
    for video_path in video_paths:
        extract_frames(video_path, output_dir)
