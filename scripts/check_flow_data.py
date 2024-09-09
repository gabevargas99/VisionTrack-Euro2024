import numpy as np
from config import flow_data_file  # Importing path from config

# Load the flow data file using the path from config.py
flow_data = np.load(flow_data_file, allow_pickle=True)

# Print the shape of the data for debugging
print(flow_data.shape)
