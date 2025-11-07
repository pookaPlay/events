import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import skvideo.io
from tqdm import tqdm

from eventreader import EventReader
import cv2


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=0,dtype='uint8')
    #img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    #img[mask==0]=[255,255,255]
    img[mask==0]=[0,0,0]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('extract_events')
    parser.add_argument('event_file', type=str, help='Path to events.h5 file')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to directory to write PNG frames (optional)')
    parser.add_argument('--output_file', type=str, default=None, help='Path to write video file (optional)')
    parser.add_argument('--output_events', type=str, default=None, help='Path to write numpy events (optional)')
    parser.add_argument('--delta_time_ms', '-dt', type=float, default=50.0, help='Time window (in milliseconds) to summarize events for visualization')
    parser.add_argument('--step_time_ms', '-st', type=float, default=50.0, help='Time window (in milliseconds) to summarize events for visualization')
    parser.add_argument('--min_frame', type=int, default=0, help='Minimum frame to render')
    parser.add_argument('--max_frame', type=int, default=500, help='Maximum number of frames to render')
    parser.add_argument('--display', action='store_true', help='Display frames interactively and wait for key press')
    parser.add_argument('--print_events', action='store_true', help='If displaying, print the current events to console')
    args = parser.parse_args()

    event_filepath = Path(args.event_file)
    output_dir = Path(args.output_dir) if args.output_dir else None
    video_filepath = Path(args.output_file) if args.output_file else None
    events_filepath = Path(args.output_events) if args.output_events else None
    dt = args.delta_time_ms
    st = args.step_time_ms

    height = 480
    width = 640

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = None
    if video_filepath:
        assert video_filepath.parent.is_dir(), "Directory {} does not exist".format(str(video_filepath.parent))
        writer = skvideo.io.FFmpegWriter(video_filepath)
    
    if args.display:
        cv2.namedWindow('Event Visualization', cv2.WINDOW_AUTOSIZE)
    all_events = dict()

    # Disable tqdm progress bar if displaying interactively to avoid console clutter
    for i, events in enumerate(tqdm(EventReader(event_filepath, dt, st), disable=args.display)):
        if args.max_frame is not None and i >= args.max_frame:
            break
        if args.min_frame is not None and i < args.min_frame:
            continue

        all_events[i] = events

        if args.display and args.print_events:
            print(f"--- Events for frame {i} ---")
            print(events)

        p = events['p']
        x = events['x']
        y = events['y']
        
        
        # events where p == 1
        # pevents = events[events['p'] == 1]
        print(f"frame {i} has {len(events['p'])} events")

        # t = events['t'] # 't' is not used in the render function
        img = render(x, y, p, height, width)
        
        if output_dir:
            frame_path = output_dir / f'frame_{i:05d}.png'
            cv2.imwrite(str(frame_path), img)

        if writer:
            writer.writeFrame(img)

        if args.display:
            cv2.imshow('Event Visualization', img)         
                        
            key = cv2.waitKey(0) # Wait indefinitely for a key press
            if key == ord('q'): # Press 'q' to quit the display loop
                break

    if events_filepath is not None:
        print(f"Saving events to {events_filepath}...")
        with open(events_filepath, 'wb') as f:
            pickle.dump(all_events, f)

    if writer:
        writer.close()
    if args.display:
        cv2.destroyAllWindows()
