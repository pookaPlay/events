from velocity_histogram import VelocityHistogram
from rvf_viz import render_polar_histogram
import copy
import numpy as np
from scipy.spatial import cKDTree
import cv2

class RVF:
    def __init__(self):
        print(f"RVF")
        self.height = 480
        self.width = 640
        self.num_events = 0
        # Parameters for our velocity histograms
        self.num_radius_bins = 8
        self.num_angle_bins = 16
        self.max_radius = 10.0  # Example: max speed of 10 pixels/second

        self.pastPoints = dict()
        self.past_events_tree = None
        
        self.search_radius = self.max_radius

    def Init(self, events):
        self.num_events = events['x'].shape[0]
        assert self.num_events == events['y'].shape[0]
        self.pastPoints = copy.deepcopy(events)
        print(f"Initilized with {self.num_events} events")

        # Build the spatial index on the initial points
        if self.num_events > 0:
            past_coords = np.stack((self.pastPoints['x'].astype(int), self.pastPoints['y'].astype(int)), axis=1)
            self.past_events_tree = cKDTree(past_coords)
            
            self.past_events_velocity = [VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)
                                        for _ in range(self.num_events)]

        else:
            self.past_events_tree = None
            self.past_events_velocity = None

    def Step(self, events):
        self.num_events = events['x'].shape[0]
        assert self.num_events == events['y'].shape[0]

        print(f"Stepping with {self.num_events} events")

        # Initialize a list of VelocityHistogram objects, one for each event in the current step.
        self.event_velocity = [VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)
                               for _ in range(self.num_events)]
        
        total_neighbors = 0
        total_peaks = 0
        flagEvents = []

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

                for ni in range(r.shape[0]):
                    nhist = self.past_events_velocity[indices_of_nearby_past_events[ni]]      
                    nvel = VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)
                    nvel.add_events(r[ni], theta[ni])

                    nmod = VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)                    
                    nmod.multiply(nvel, nhist)
                    self.event_velocity[i].add(nmod)

                self.event_velocity[i].normalize()
                peakVal = self.event_velocity[i].peak()
                if peakVal > 0.2:
                    total_peaks += 1
                    flagEvents.append(i)

                if 0: 
                    print("Plotting histogram")
                    img = render_polar_histogram(self.event_velocity[i].histogram, self.max_radius)
                    cv2.imshow('Event Histogram', img)

        
        print(f"Total Peaks: {total_peaks}")
        print(f"Total neighbors: {total_neighbors}") 

        self.pastPoints = copy.deepcopy(events)

        # Build the spatial index on the initial points
        if self.pastPoints['x'].shape[0] > 0:
            past_coords = np.stack((self.pastPoints['x'].astype(int), self.pastPoints['y'].astype(int)), axis=1)
            self.past_events_tree = cKDTree(past_coords)
            self.past_events_velocity = copy.deepcopy(self.event_velocity)
        else:
            self.past_events_tree = None
            self.past_events_velocity = None

        return flagEvents