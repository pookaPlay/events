import pickle
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from VecKM import  ExactVecKM
import torch
from rvf_viz import render

if __name__ == '__main__':
    #parser = argparse.ArgumentParser('veckm_events')
    #parser.add_argument('event_file', type=str, help='Path to events.pkl file')
    #args = parser.parse_args()
    #event_filepath = Path(args.event_file)

    height = 480
    width = 640
    radius = 120
    nradius = 100
    N = 10

    cv2.namedWindow('Event Visualization', cv2.WINDOW_AUTOSIZE)

    # Generate synthetic event data 
    # Events are 2d points (x,y) 
    # Sample a circle at N points
    # Circle should be in middle of image (width, height) and have radius "radius" 
    center_x = width / 2
    center_y = height / 2
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)

    events = dict()
    events['x'] = x_coords.astype(np.int32)
    events['y'] = y_coords.astype(np.int32)
    events['p'] = np.ones(N, dtype=np.int32) # Polarity

    #torch.Tensor, (N, self.pt_dim), input point cloud. 
    tevents = torch.from_numpy(np.stack((events['x'], events['y']), axis=1)).float()

    vec = ExactVecKM(2, 256, 10.0)
    event_features = vec.forward(tevents)
    event_features = event_features.detach().numpy()

    print(f"Event features shape: {event_features.shape}")


    img = render(events['x'], events['y'], events['p'], height, width)
    cv2.imshow('Event Visualization', img)

    key = cv2.waitKey(0) # Wait for a key press
    cv2.destroyAllWindows()
        