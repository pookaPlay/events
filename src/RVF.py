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
        self.num_radius_bins = 16
        self.num_angle_bins = 16
        self.max_radius = 16.0  # max speed
        
        self.alpha = 1
        

        self.pastPoints = dict()
        self.past_events_tree = None
        
        self.search_radius = self.max_radius
    
    def GetEventVelocity(self, ei):
        return self.event_velocity[ei]

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
        self.event_neighbors = [[] for _ in range(self.num_events)]

        total_neighbors = 0
        total_peaks = 0
        flagEvents = []

        if self.num_events > 0 and self.past_events_tree is not None:            
            # for each event 
            for ei in range(self.num_events):
                current_event_coord = (events[ei]['x'], events[ei]['y'])
                int_event_coord = (int(events[ei]['x']), int(events[ei]['y']))

                indices_of_nearby_past_events = self.past_events_tree.query_ball_point(int_event_coord, r=self.search_radius)
                
                total_neighbors += len(indices_of_nearby_past_events)
                self.event_neighbors[ei] = indices_of_nearby_past_events
                
                nearby_past_events_x = np.array([self.pastPoints[i]['x'] for i in indices_of_nearby_past_events])
                nearby_past_events_y = np.array([self.pastPoints[i]['y'] for i in indices_of_nearby_past_events])

                dx = current_event_coord[0] - nearby_past_events_x
                dy = current_event_coord[1] - nearby_past_events_y
                # convert to polar
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)

                for ni in range(r.shape[0]):
                    nhist = self.past_events_velocity[indices_of_nearby_past_events[ni]]
                    nhist.smooth()                    

                    nevent_coord = (nearby_past_events_x[ni], nearby_past_events_y[ni])
                    nevent_pred  = nhist.PredictNextLocation(nevent_coord)
                    ndx = current_event_coord[0] - nevent_pred[0]
                    ndy = current_event_coord[1] - nevent_pred[1]

                    # convert to polar
                    pr = np.sqrt(ndx**2 + ndy**2)
                    ptheta = np.arctan2(ndy, ndx)
                    pweight = 1.0 - (pr / self.max_radius)

                    nvel = VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)
                    nvel.add_events_normalize(r[ni], theta[ni])
                    nvel.smooth()

                    nmod = VelocityHistogram(self.num_radius_bins, self.num_angle_bins, self.max_radius)                    
                    #nmod.multiply_normalize(nvel, nhist)
                    nmod.multiply_alpha_normalize(nhist, self.alpha, nvel)
                                        
                    # When r[ni] is 0, weight is 1. When r[ni] is max_radius, weight is 0.
                    weight = 1.0 - (r[ni] / self.max_radius)
                    if ei == 6:
                        print(f" Loc weight {weight} and predicted {pweight}")

                    # Apply the weight to the histogram before adding it.
                    nmod.histogram *= pweight
                    self.event_velocity[ei].add(nmod)                    

                self.event_velocity[ei].normalize()
                #(val, radius, angle)
                self.event_peak[ei] = self.event_velocity[ei].peak()
                

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
    