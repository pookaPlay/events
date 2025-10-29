import pickle
import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from RVF import RVF
from rvf_viz import render
from velocity_histogram import VelocityHistogram
from rvf_viz import render_polar_histogram

if __name__ == '__main__':

    height = 480
    width = 640
    num_radius_bins = 8
    num_angle_bins = 16
    max_radius = 10.0  # Example: max speed of 10 pixels/second

    cv2.namedWindow('Event Histogram', cv2.WINDOW_AUTOSIZE)        
    
    v1 = VelocityHistogram(num_radius_bins, num_angle_bins, max_radius)    
    v2 = VelocityHistogram(num_radius_bins, num_angle_bins, max_radius)
    vm = VelocityHistogram(num_radius_bins, num_angle_bins, max_radius)
    
    rad1 = np.array([5.0])
    ang1 = np.array([0.0])
    v1.add_events(rad1, ang1)

    rad2 = np.array([5.0])
    ang2 = np.array([0.0])
    v2.add_events(rad2, ang2)

    vm.multiply(v1, v2)
    #vm.add(v1, v2)
    
    v1img = render_polar_histogram(v1.histogram, max_radius)
    vmimg = render_polar_histogram(vm.histogram, max_radius)

    cv2.imshow('Event Histogram', v1img)
    cv2.imshow('Event Histogram Post', vmimg)
            
    key = cv2.waitKey(0) # Wait 1ms, allows for animation
    #if key == ord('q'): # Press 'q' to quit the display loop
    #    break

    cv2.destroyAllWindows()
        