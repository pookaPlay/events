import pickle
import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from RVF import RVF
from rvf_viz import render

if __name__ == '__main__':
    parser = argparse.ArgumentParser('rvf_events')
    parser.add_argument('event_file', type=str, help='Path to events.pkl file')
    args = parser.parse_args()

    event_filepath = Path(args.event_file)
    
    height = 480
    width = 640

    rvf = RVF()

    with open(event_filepath, 'rb') as f:
        all_events = pickle.load(f)
        #print(all_events)

        cv2.namedWindow('Event Visualization', cv2.WINDOW_AUTOSIZE)        

        min_key = min(all_events.keys())
        max_key = max(all_events.keys())
        print(f"Found event frames with keys from {min_key} to {max_key}.")        

        # Iterate over the range of keys
        for i in range(min_key, max_key + 1):
            if i not in all_events:
                continue
            events = all_events[i]
            
            if i == min_key:
                rvf.Init(events)
            else:
                rvf.Step(events)

            img = render(events['x'], events['y'], events['p'], height, width)

            
            #hist_img = plot_polar_histogram(single_histogram, rvf.max_radius, title=f"Velocity")
            #cv2.imshow('Polar Histogram', hist_img)

            cv2.imshow('Event Visualization', img)
            
            key = cv2.waitKey(0) # Wait 1ms, allows for animation
            if key == ord('q'): # Press 'q' to quit the display loop
                break

        cv2.destroyAllWindows()
        