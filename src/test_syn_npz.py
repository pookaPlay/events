# load the json event file 
# data/easy_one.json
import pickle
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json

def RenderImage(events, H = 480, W = 640):

    img = np.full((H,W,3), fill_value=0,dtype='uint8')
    #img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')

    # extract data from events
    x = np.array([e['x'] for e in events], dtype=np.int32)
    y = np.array([e['y'] for e in events], dtype=np.int32)    
    pol = np.array([e['p'] for e in events], dtype=np.int32)
        
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    #img[mask==0]=[255,255,255]
    img[mask==0]=[0,0,0]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img
                  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test_syn')
    parser.add_argument('event_file', type=str, help='Path to events.json file', default='data/easy_one.json')
    parser.add_argument('--w', type=int, help='image width', default=1000)
    parser.add_argument('--h', type=int, help='image height', default=1000)
    args = parser.parse_args()
    event_filepath = Path(args.event_file)
    
    height = args.h
    width = args.w
    data = np.load(event_filepath)
    print("Loaded data")
    print("Keys in NPZ file:", data.files)
    for key in data.files:
        # ['evs_norm', 'ev_loc', 'ev']
        eventData = data[key]
        print(f"Shape of {key}: {eventData.shape}")
    
    event_loc = data['ev_loc']
    event_norm = data['evs_norm']
    event_label = data['ev']
    print(f"Event 0: {event_loc[0]} with label {event_label[0]} and norm {event_norm[0]}")
    
    data.close()
    
    
    #totalFrames = data['totalFrames']
    totalFrames = 5
    print(f"Total frames: {totalFrames}")
    
    # print(f"{data['frames'][1]['events'][0]}")
    # {'x': 975.306054931405, 'y': 768.8413214736106, 't': 10, 'p': -1}
    title = f"Frames"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    for frame, fi in zip(data['frames'], range(totalFrames)):
        #print(data['frames']:
        
        print(f"Frame {fi}: {len(frame['events'])}") 
        
        # plot the events 
        img = RenderImage(frame['events'], height, width)        
        
        cv2.imshow(title, img)

        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

