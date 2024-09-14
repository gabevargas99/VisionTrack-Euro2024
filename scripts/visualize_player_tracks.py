import cv2
import numpy as np
import os
from config import tracks_input_folder, tracks_mask_folder, tracks_output_folder

def draw_bounding_boxes(frame, mask):
    # finds contours in binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # loops through them and calculates the bounding box and draws
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def visualize_player_tracks(input_folder, mask_folder, output_folder):
    # retrieves and sorts list og jpg in input folder
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    # retrieves and sorts list og jpg in mask folder
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.jpg')])
    
    # loops through frame and masks
    for i, (frame_file, mask_file) in enumerate(zip(frame_files[:-1], mask_files)):
        frame_path = os.path.join(input_folder, frame_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # draws bounding boxes on frame using mask
        frame_with_boxes = draw_bounding_boxes(frame, mask)
        
        # saves output frame
        output_path = os.path.join(output_folder, f"tracked_{i:04d}.jpg")
        cv2.imwrite(output_path, frame_with_boxes)
        print(f"Saved tracked frame to {output_path}")

if __name__ == "__main__":
    # creates output directory if it doesn't exist
    if not os.path.exists(tracks_output_folder):
        os.makedirs(tracks_output_folder)
    visualize_player_tracks(tracks_input_folder, tracks_mask_folder, tracks_output_folder)
