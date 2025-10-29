

from velocity_histogram import VelocityHistogram
from rvf_viz import plot_polar_histogram

import copy
import numpy as np
from scipy.spatial import cKDTree

class RVF:
    def __init__(self):
        print(f"RVF")
        self.height = 480
        self.width = 640
        self.num_events = 0
        # Parameters for our velocity histograms
        self.num_radius_bins = 8
        self.num_angle_bins = 12
        self.max_radius = 10.0  # Example: max speed of 10 pixels/second

        self.pastPoints = dict()
        self.past_events_tree = None
        
        self.search_radius = 5.0

    def Init(self, events):
        self.num_events = events['x'].shape[0]
        assert self.num_events == events['y'].shape[0]
        self.pastPoints = copy.deepcopy(events)
        print(f"Initilized with {self.num_events} events")

        # Build the spatial index on the initial points
        if self.num_events > 0:
            past_coords = np.stack((self.pastPoints['x'].astype(int), self.pastPoints['y'].astype(int)), axis=1)
            self.past_events_tree = cKDTree(past_coords)
        else:
            self.past_events_tree = None

    def Step(self, events):
        self.num_events = events['x'].shape[0]
        assert self.num_events == events['y'].shape[0]

        print(f"Stepping with {self.num_events} events")

        total_neighbors = 0

        if self.num_events > 0 and self.past_events_tree is not None:            
            # for each event 
            for i in range(self.num_events):
                current_event_coord = (int(events['x'][i]), int(events['y'][i]))

                indices_of_nearby_past_events = self.past_events_tree.query_ball_point(current_event_coord, r=self.search_radius)
                total_neighbors += len(indices_of_nearby_past_events)
                
                nearby_past_events_x = self.pastPoints['x'][indices_of_nearby_past_events].astype(int)
                nearby_past_events_y = self.pastPoints['y'][indices_of_nearby_past_events].astype(int)
                dx = current_event_coord[0] - nearby_past_events_x
                dy = current_event_coord[1] - nearby_past_events_y
                # convert to polar
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)

        print(f"Total neighbors: {total_neighbors}") 

        self.pastPoints = copy.deepcopy(events)

        if self.pastPoints['x'].shape[0] > 0:
            past_coords = np.stack((self.pastPoints['x'].astype(int), self.pastPoints['y'].astype(int)), axis=1)
            self.past_events_tree = cKDTree(past_coords)
        else:
            self.past_events_tree = None
