# VisionTrack-Euro2024

Inside the scripts folder, there are 7 files meant to track, extract motion and patterns from football games:

1. extract_frames.py:
Extracts frames from a video and saves them as individual .jpg files and saves them individually in the frames directory. If run with another video frames will just be replaced

2. calculate_optical_flow.py:
Calculates the optical flow between consecutive frames and saves as flow.npy file in output folder

3. segment_and_track_players.py:
Segments players based on optical flow data and saves the segmentation masks and saves in masks folder in output folder

4. visualize_optical_flow.py:
Displays optical flow on extracted frames and saves them in flow_vizualization in output folder

5. visualie_player_tracks.py:
Draws bounding boxes around detected players and saves to tracked folder in output folder

6. check_flow_data.py:
Prints out dimensions, used in debug process

7. video_tracking.py: 
Processes a video to track live player movements, video name is hardcoded in main part of function, ran individual videos to go through one step at a time and make sure every script was working correctly

Used:
Python 3.9.13
OpenCV
NumPy
Scikit-learn
