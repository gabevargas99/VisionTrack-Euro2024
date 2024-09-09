import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from config import input_folder, flow_data_file, output_folder

def segment_players(flow):
    # gets height and width of optical flow data
    h, w = flow.shape[:2]
    
    # calculate the magnitude of the optical flow vectors
    flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    
    # reshape the flow magnitude for clustering
    flow_flat = flow_magnitude.reshape((-1, 1))
    
    # use K-Means clustering on flow magnitude
    kmeans = KMeans(n_clusters=2).fit(flow_flat)
    labels = kmeans.labels_.reshape((h, w))
    
    # determines which cluster represents players
    cluster_means = [flow_magnitude[labels == i].mean() for i in range(2)]
    player_cluster = np.argmax(cluster_means)
    
    # create binary mask where the player cluster is 1 and the background is 0
    player_mask = (labels == player_cluster).astype(np.uint8)
    
    # apply morphological operations for noise
    kernel = np.ones((5, 5), np.uint8)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, kernel)
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
    
    return player_mask

def process_and_save_masks(input_folder, flow_file, output_folder):
    # loads optical flow data
    flow_data = np.load(flow_file, allow_pickle=True)
    # retrieves sortled list of jpg in folder
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    # loops through frame and flow data
    for i, (flow, frame_file) in enumerate(zip(flow_data, frame_files[:-1])):
        # makes path to current file frame and reads it
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        
        mask = segment_players(flow)
        
        # saves mask
        mask_output_path = os.path.join(output_folder, f"mask_{i:04d}.jpg")
        cv2.imwrite(mask_output_path, mask * 255)
        print(f"Saved mask to {mask_output_path}")

if __name__ == "__main__":
   if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        process_and_save_masks(input_folder, flow_data_file, output_folder)
