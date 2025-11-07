import pickle
import argparse
import os
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm

from RVF import RVF
from rvf_viz import RenderSynImage, RenderEventImage, render_polar_histogram

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_rvf')
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

    def mouse_callback(event, x, y, flags, param):
        """Handles mouse events to find and display the closest event ID."""
        if event == cv2.EVENT_MOUSEMOVE:
            shared_state['mouse_pos'] = (x, y)
            min_dist = float('inf')
            closest_idx = -1
            if shared_state['events']:
                # This is a simple linear search. For a large number of events, a k-d tree would be faster.
                for i, e in enumerate(shared_state['events']):
                    dist = np.sqrt((e['x'] - x)**2 + (e['y'] - y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
            shared_state['closest_event_dist'] = min_dist
            shared_state['closest_event_idx'] = closest_idx
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            # On left click, if near an event, show its histogram
            if shared_state['closest_event_idx'] != -1 and shared_state['closest_event_dist'] < 10:
                idx = shared_state['closest_event_idx']

                if shared_state['event_neighbors']:
                    num_neighbors = len(shared_state['event_neighbors'][idx])
                    print(f"Event ID: {idx} has {num_neighbors} neighbors.")

                if shared_state['event_velocities']:
                    hist_obj = shared_state['event_velocities'][idx]
                    hist_img = render_polar_histogram(hist_obj.histogram, rvf.max_radius)
                    hist_window_title = f"Velocity Histogram for Event ID: {idx}"
                    cv2.imshow(hist_window_title, hist_img)

    rvf = RVF()

    height = 1000
    width = 1000

    with open(event_filepath, 'r') as f:
        data = json.load(f)

    totalFrames = data['totalFrames']
    print(f"Total frames: {totalFrames}")
    
    # print(f"{data['frames'][1]['events'][0]}")
    # {'x': 975.306054931405, 'y': 768.8413214736106, 't': 10, 'p': -1}
    title = f"Frames"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)

    for frame, fi in zip(data['frames'], range(totalFrames)):
        #print(data['frames']:
        print(f"Frame {fi}: {len(frame['events'])}") 
        
        shared_state['events'] = frame['events']

        if fi == 0:
            rvf.Init(frame['events'])
            img = RenderSynImage(frame['events'], height, width)
        else:
            event_peak = rvf.Step(frame['events'])
            shared_state['event_velocities'] = rvf.event_velocity
            shared_state['event_neighbors'] = rvf.event_neighbors
            img = RenderEventImage(frame['events'], event_peak, height, width, scale=5.0)            
        
        while True:
            display_img = img.copy()
            
            # If mouse is close to an event, display its ID
            if shared_state['closest_event_idx'] != -1 and shared_state['closest_event_dist'] < 10: # 10 pixel threshold
                idx = shared_state['closest_event_idx']
                event_pos = (int(shared_state['events'][idx]['x']), int(shared_state['events'][idx]['y']))
                text = f"ID: {idx}"
                cv2.putText(display_img, text, (event_pos[0] + 5, event_pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(title, display_img)

            key = cv2.waitKey(20) # Use a small delay to allow for smooth mouse updates
            if key == ord('q'): # Quit
                cv2.destroyAllWindows()
                exit()
            if key != -1: # Any other key press moves to the next frame
                break

    cv2.destroyAllWindows()
