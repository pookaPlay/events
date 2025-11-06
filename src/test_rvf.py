import pickle
import argparse
import os
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm

from RVF import RVF
from rvf_viz import RenderSynImage, RenderEventImage

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_rvf')
    parser.add_argument('event_file', type=str, help='Path to events.json file')
    args = parser.parse_args()

    event_filepath = Path(args.event_file)
    
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
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    for frame, fi in zip(data['frames'], range(totalFrames)):
        #print(data['frames']:
        print(f"Frame {fi}: {len(frame['events'])}") 
        
        if fi == 0:
            rvf.Init(frame['events'])
            img = RenderSynImage(frame['events'], height, width)
        else:
            event_peak = rvf.Step(frame['events'])
            img = RenderEventImage(frame['events'], event_peak, height, width, scale=2.0)            
            #print(event_peak)
        
        cv2.imshow(title, img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
