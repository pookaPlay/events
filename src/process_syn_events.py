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

    
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    #img[mask==0]=[255,255,255]
    img[mask==0]=[0,0,0]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img
                  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('veckm_events')
    parser.add_argument('--events', type=str, help='Path to events.json', default='data/easy_one.json')
    parser.add_argument('--out', type=str, help='Path to out.json', default='data/out.json')
    args = parser.parse_args()
    event_filepath = Path(args.events)
    out_filepath = Path(args.out)
    
    height = 1000.0
    width = 1000.0

    with open(event_filepath, 'r') as f:
        data = json.load(f)

    totalFrames = data['totalFrames']
    print(f"Total frames: {totalFrames}")
    

    allminx = np.inf
    allmaxx = -np.inf
    allminy = np.inf
    allmaxy = -np.inf

    for frame, fi in zip(data['frames'], range(totalFrames)):
        #print(data['frames']:        
        events = frame['events']
        
        # extract data from events
        x = np.array([e['x'] for e in events], dtype=np.int32)
        y = np.array([e['y'] for e in events], dtype=np.int32)    
        pol = np.array([e['p'] for e in events], dtype=np.int32)
        
        # Get the min and max x and w     
        minx = np.min(x)
        allminx = min(allminx, minx)

        maxx = np.max(x)
        allmaxx = max(allmaxx, maxx)

        miny = np.min(y)
        allminy = min(allminy, miny)

        maxy = np.max(y)
        allmaxy = max(allmaxy, maxy)


        print(f"Frame: {fi}   x: {minx} -> {maxx} and y: {miny} -> {maxy}")


        # plot the events 
    print(f"Data: {allminx}->{allmaxx} and {allminy}->{allmaxy}")
    # Scale the evnts to 1000 by 1000

    for frame, fi in zip(data['frames'], range(totalFrames)):
        #print(data['frames']:        
        events = frame['events']    
        for event in events:
            event['x'] = width * float(event['x'] - allminx) / float(allmaxx - allminx)
            event['y'] = height * float(event['y'] - allminy) / float(allmaxy - allminy)

            # convert back to int with floor and max sure max is width-1
            event['x'] = int(np.floor(event['x']))
            event['y'] = int(np.floor(event['y']))
            event['x'] = min(event['x'], width-1)
            event['y'] = min(event['y'], height-1)
            event['x'] = max(event['x'], 0)
            event['y'] = max(event['y'], 0)


    # Save the json 
    with open(out_filepath, 'w') as f:
        json.dump(data, f)

            






