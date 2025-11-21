import pickle
import argparse
import os
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm

from RVF_Gaussian import RVF_Gaussian
from rvf_viz import RenderSynImage, RenderEventImageGaussian

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_rvf_gaussian')
    parser.add_argument('event_file', type=str, help='Path to events.json file')
    args = parser.parse_args()
 
    event_filepath = Path(args.event_file)
    
    # --- Mouse Callback and Shared State for Interactivity ---
    shared_state = {
        'events': [],
        'event_velocities': [],
        'mouse_pos': (0, 0),
        'closest_event_idx': -1,
        'closest_event_dist': float('inf')
    }

    rvf = RVF_Gaussian(search_radius=16.0, init_var=1.0, proc_var=1.0, alpha=1.0)

    height = 800
    width = 800
    rvfThresh = 0.5

    with open(event_filepath, 'r') as f:
        data = json.load(f)

    totalFrames = data['totalFrames']
    print(f"Total frames: {totalFrames}")
    
    # print(f"{data['frames'][1]['events'][0]}")
    # {'x': 975.306054931405, 'y': 768.8413214736106, 't': 10, 'p': -1}
    title = f"Frames"
    cv2.namedWindow(title)    

    for frame, fi in zip(data['frames'], range(totalFrames)):
        #print(data['frames']:
        print(f"Frame {fi}: {len(frame['events'])}") 
        
        shared_state['events'] = frame['events']

        if fi == 0:
            rvf.Init(frame['events'])
            img = RenderSynImage(frame['events'], height, width)
        else:
            event_peak = rvf.Step(frame['events'])
            img = RenderEventImageGaussian(frame['events'], event_peak, rvf.new_vars, height, width, scale=5.0, rvfThresh=rvfThresh)            
        
        while True:
            display_img = img.copy()
            
            cv2.imshow(title, display_img)

            key = cv2.waitKey(20) # Use a small delay to allow for smooth mouse updates
            if key == ord('q'): # Quit
                cv2.destroyAllWindows()
                exit()
            if key != -1: # Any other key press moves to the next frame
                break

    cv2.destroyAllWindows()
