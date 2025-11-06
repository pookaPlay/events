from velocity_histogram import VelocityHistogram
from rvf_viz import render_polar_histogram
import copy
import numpy as np
from scipy.spatial import cKDTree
import cv2

class RVF:
    def __init__(self):
        print(f"RVF")
        self.height = 1000
        self.width = 1000
        self.num_events = 0
        # Parameters for our velocity histograms
        self.num_radius_bins = 8
        self.num_angle_bins = 16
        self.max_radius = 50.0  # max speed

        self.pastPoints = dict()
        self.past_events_tree = None
        
        self.search_radius = self.max_radius

    def Init(self, events):
        self.num_events = len(events)
        
        self.pastPoints = copy.deepcopy(events)
        print(f"Initilized with {self.num_events} events")

        # Build the spatial index on the initial points
        if self.num_events > 0:
            x = [e['x'] for e in events]
            y = [e['y'] for e in events]
            past_coords = np.array(list(zip(x, y)), dtype=np.int32)
            print(f"Init of points: {past_coords.shape}")

            self.past_events_tree = cKDTree(past_coords)
            
            self.past_events_velocity = [VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)
                                        for _ in range(self.num_events)]

        else:
            self.past_events_tree = None
            self.past_events_velocity = None

    def Step(self, events):
        self.num_events = len(events)

        print(f"Stepping with {self.num_events} events")

        # Initialize a list of VelocityHistogram objects, one for each event in the current step.
        self.event_velocity = [VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)
                               for _ in range(self.num_events)]
        # Use the peak for viz        
        self.event_peak = [None for _ in range(self.num_events)]

        total_neighbors = 0
        total_peaks = 0
        flagEvents = []

        if self.num_events > 0 and self.past_events_tree is not None:            
            # for each event 
            for i in range(self.num_events):
                current_event_coord = (events[i]['x'], events[i]['y'])
                int_event_coord = (int(events[i]['x']), int(events[i]['y']))

                indices_of_nearby_past_events = self.past_events_tree.query_ball_point(int_event_coord, r=self.search_radius)
                #print(f"Found {len(indices_of_nearby_past_events)} neighbors")
                #print(indices_of_nearby_past_events)

                total_neighbors += len(indices_of_nearby_past_events)
                
                nearby_past_events_x = np.array([self.pastPoints[i]['x'] for i in indices_of_nearby_past_events])
                nearby_past_events_y = np.array([self.pastPoints[i]['y'] for i in indices_of_nearby_past_events])

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

                val, radius, angle = self.event_velocity[i].peak()
                self.event_peak[i] = (val, radius, angle)

        print(f"  Avg # Neighbors: {total_neighbors/self.num_events}")        

        self.pastPoints = copy.deepcopy(events)
        # Build the spatial index on the initial points
        if self.num_events > 0:
            x = [e['x'] for e in events]
            y = [e['y'] for e in events]
            new_coords = np.array(list(zip(x, y)))
            print(f"Init of points: {new_coords.shape}")

            self.past_events_tree = cKDTree(new_coords)            
            self.past_events_velocity = copy.deepcopy(self.event_velocity)
        else:
            print(f"No events found this step!!!!!")
            self.past_events_tree = None
            self.past_events_velocity = None


        return self.event_peak