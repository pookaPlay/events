import pickle
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
from rvf_viz import RenderSynImage
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_events')
    parser.add_argument('event_file', type=str, help='Path to events.pkl file')
    args = parser.parse_args()

    event_filepath = Path(args.event_file)
    
    height = 480
    width = 640

    with open(event_filepath, 'rb') as f:
        all_events = pickle.load(f)        
        output_data = dict()

        cv2.namedWindow('Event Visualization', cv2.WINDOW_AUTOSIZE)        

        min_key = min(all_events.keys())
        max_key = max(all_events.keys())
        print(f"Found event frames with keys from {min_key} to {max_key}.")        

        output_data['totalFrames'] = max_key - min_key + 1
        output_data['frames'] = list()

        # Iterate over the range of keys
        for fi in range(min_key, max_key + 1):
            
            if fi not in all_events:
                continue
            fevents = all_events[fi]            
            events = list()
            
            # convert to event struct format
            localt = fi - min_key
            for j in range(len(fevents['x'])):
                events.append({
                    'x': float(fevents['x'][j]), 
                    'y': float(fevents['y'][j]), 
                    't': localt, 
                    'p': int(fevents['p'][j])
                })
            
            print(f"Frame {fi} has {len(events)} events")
            output_data['frames'].append({
                'events': events
            })
        
            #img = RenderSynImage(events, height, width)
            #cv2.imshow('Event Visualization', img)
                      
            #key = cv2.waitKey(0) # Wait 1ms, allows for animation
            #if key == ord('q'): # Press 'q' to quit the display loop
            #    break

    # save json
    with open('test_events_bak.json', 'w') as fout:
        json.dump(output_data, fout)

        
    #cv2.destroyAllWindows()
        