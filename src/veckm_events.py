import pickle
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from VecKM import  ExactVecKM
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('veckm_events')
    parser.add_argument('event_file', type=str, help='Path to events.pkl file')
    args = parser.parse_args()

    event_filepath = Path(args.event_file)
    
    height = 480
    width = 640

    with open(event_filepath, 'rb') as f:
        all_events = pickle.load(f)
        #print(all_events)

        cv2.namedWindow('Event Visualization', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Event Histogram', cv2.WINDOW_AUTOSIZE)        

        min_key = min(all_events.keys())
        max_key = max(all_events.keys())
        print(f"Found event frames with keys from {min_key} to {max_key}.")        

        # Iterate over the range of keys
        for i in range(min_key, max_key + 1):
            if i not in all_events:
                continue
            events = all_events[i]            
            
            #torch.Tensor, (N, self.pt_dim), input point cloud. 
            tevents = torch.from_numpy(np.stack((events['x'], events['y']), axis=1)).float()

            vec = ExactVecKM(2, 256, 10.0)
            event_features = vec.forward(tevents)

            event_features = event_features.detach().numpy()

            if i == min_key:
                past_event_features = event_features
            
            key = cv2.waitKey(0) # Wait 1ms, allows for animation
            if key == ord('q'): # Press 'q' to quit the display loop
                break

        cv2.destroyAllWindows()
        