import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import deque
from config import video_tracking_input_path, video_tracking_output_path

def calculate_optical_flow(frame1, frame2, roi):
    # grayscale version of frame 1 & frame 2
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # calculate optical flow between the 2 frames with ROI and Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1[roi[1]:roi[3], roi[0]:roi[2]], 
        gray2[roi[1]:roi[3], roi[0]:roi[2]], 
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow

def segment_players(flow, min_area=1000, max_width=100, max_height=200):
    # gets dimensions of height and width
    h, w = flow.shape[:2]
    
    # calculates the magnitude of optical flow vectors
    flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    
    # reshapes flow magnitude for clustering
    flow_flat = flow_magnitude.reshape((-1, 1))
    
    # use K-Means clustering on flow magnitude
    kmeans = KMeans(n_clusters=2, random_state=0).fit(flow_flat)
    labels = kmeans.labels_.reshape((h, w))
    
    # determine which cluster represents players
    cluster_means = [flow_magnitude[labels == i].mean() for i in range(2)]
    player_cluster = np.argmax(cluster_means)
    
    # create a binary mask where the player cluster is 1 and the background is 0
    player_mask = (labels == player_cluster).astype(np.uint8)
    
    # apply morphological operations for noise
    kernel = np.ones((5, 5), np.uint8)
    
    # morphological opening; erode followed by dilate
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, kernel)
    
    # morphological closing: dilate followed by erode
    player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
    
    # find contours in player mask
    contours, _ = cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter contours by area to remove small noise and by size to remove large boxes
    player_contours = [c for c in contours if cv2.contourArea(c) > min_area and 
                       cv2.boundingRect(c)[2] <= max_width and cv2.boundingRect(c)[3] <= max_height]
    
    return player_contours

def draw_bounding_boxes(frame, contours, roi):
    # loops through contours and calculates bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x + roi[0], y + roi[1]), (x + w + roi[0], y + h + roi[1]), (0, 255, 0), 2)
    return frame

def check_consistency(current_contours, history, threshold=0.2):
    consistent_contours = []
    # loops through current contours in frame
    for current in current_contours:
        x1, y1, w1, h1 = cv2.boundingRect(current)
        area1 = w1 * h1

        # loop through previous contours
        for prev in history:
            x2, y2, w2, h2 = cv2.boundingRect(prev)
            area2 = w2 * h2

            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y

            if overlap_area / min(area1, area2) > threshold:
                consistent_contours.append(current)
                break

    return consistent_contours

def process_video(video_path, output_path):
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # get frame dimensions and fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame dimensions: {frame_width}x{frame_height}, FPS: {fps}")

    # makes sure output path directory exists
    import os
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # used MJPG for better compatibility
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        return

    # define the ROI based on the highlighted area (x1, y1, x2, y2)
    roi = (int(frame_width * 0.05), int(frame_height * 0.2), int(frame_width * 0.95), int(frame_height * 0.8))
    
    # initialize history and read frist frame
    history = deque(maxlen=5)
    ret, frame1 = cap.read()
    frame_count = 0
    # loops through frames
    while ret:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        flow = calculate_optical_flow(frame1, frame2, roi)
        contours = segment_players(flow)
        print(f"Frame {frame_count}: Detected {len(contours)} contours")
        # checks current contour for consistency
        consistent_contours = check_consistency(contours, history)
        print(f"Frame {frame_count}: Consistent {len(consistent_contours)} contours")
        # update history and draw bounding boxes
        history.extend(contours) 
        frame2 = draw_bounding_boxes(frame2, consistent_contours, roi)
        
        # write frame and display
        out.write(frame2)
        cv2.imshow('Frame', frame2)
        
        # updates for next iteration
        frame1 = frame2.copy()
        frame_count += 1
        
        # checks for q key to break loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # releases resources and prints summary
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames. Video saved to {output_path}")

if __name__ == "__main__":
    process_video(video_tracking_input_path, video_tracking_output_path)
