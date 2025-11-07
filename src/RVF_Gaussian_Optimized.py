import copy
import numpy as np
from scipy.spatial import cKDTree
import cv2

class RVF_Gaussian_Optimized:
    def __init__(self, search_radius=8.0, init_var=1.0, proc_var=1.0, alpha=1.0, spatial_search_radius=50.0):
        print(f"RVF")
        self.num_events = 0
        self.search_radius = search_radius
        self.spatial_search_radius = spatial_search_radius
        self.init_var = init_var
        self.var = proc_var        
        self.alpha = alpha
        self.runSpatial = False
        
        self.pastPoints = dict()
        self.past_events_tree = None
                
    def Init(self, events):
        self.num_events = len(events)        
        self.pastPoints = copy.deepcopy(events)
        
        print(f"Initilizing with {self.num_events} events")
        
        # Build the spatial index on the initial points
        if self.num_events > 0:
            x = [e['x'] for e in events]
            y = [e['y'] for e in events]
            past_coords = np.array(list(zip(x, y)), dtype=np.int32)
            print(f"Init of points: {past_coords.shape}")

            self.past_events_tree = cKDTree(past_coords)
            
            self.past_means = np.zeros((self.num_events, 2))
            self.past_vars = np.ones(self.num_events) * self.init_var
            self.past_weights = np.zeros(self.num_events)

        else:
            self.past_events_tree = None
            self.past_means = None
            self.past_vars = None
            self.past_weights = None

    def Step(self, events):
        self.num_events = len(events)

        print(f"Stepping with {self.num_events} events")
        if self.num_events > 0:
            self.new_means = np.zeros((self.num_events, 2))
            self.new_vars = np.ones(self.num_events) * self.init_var
            self.new_weights = np.zeros(self.num_events)

            total_neighbors = 0
            total_peaks = 0        

            # Use the peak for viz        
            self.event_peak = [None for _ in range(self.num_events)]
            self.event_neighbors = [[] for _ in range(self.num_events)]

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

                    # Check if there are any neighbors before proceeding
                    if dx.shape[0] > 0:
                        # Extract past means and variances for all nearby events
                        nmeans_neighbors = self.past_means[indices_of_nearby_past_events] # (num_neighbors, 2)
                        nvars_neighbors = self.past_vars[indices_of_nearby_past_events]   # (num_neighbors,)

                        # Vectorized calculations for all neighbors of the current event
                        myvars_neighbors = self.var * nvars_neighbors / (nvars_neighbors + self.var)

                        # myx and myy are vectors (num_neighbors,)
                        myx_neighbors = dx * myvars_neighbors / self.var + nmeans_neighbors[:, 0] * myvars_neighbors / nvars_neighbors
                        myy_neighbors = dy * myvars_neighbors / self.var + nmeans_neighbors[:, 1] * myvars_neighbors / nvars_neighbors
                        
                        # scale_factor is a vector (num_neighbors,)
                        scale_factors_neighbors = np.sqrt(dx**2 + dy**2) / self.var
                        scale_factors_neighbors += np.sqrt(nmeans_neighbors[:, 0]**2 + nmeans_neighbors[:, 1]**2) / nvars_neighbors
                        scale_factors_neighbors -= np.sqrt(myx_neighbors**2 + myy_neighbors**2) / myvars_neighbors                    

                        myws_neighbors = 1.0 / (self.var + nvars_neighbors) *  np.exp( -0.5 * scale_factors_neighbors)

                        # Accumulate weighted values
                        mx = np.sum(myx_neighbors * myws_neighbors)
                        my = np.sum(myy_neighbors * myws_neighbors)
                        mv = np.sum(myvars_neighbors * myws_neighbors)
                        mw = np.sum(myws_neighbors)
                    else:
                        mx, my, mv, mw = 0.0, 0.0, 0.0, 0.0

                    # After iterating through all neighbors, if any were found,
                    # calculate the final mean and variance for the current event.
                    if mw > 0:
                        self.new_means[ei, 0] = mx / mw
                        self.new_means[ei, 1] = my / mw
                        self.new_vars[ei] = mv / mw
                        self.new_weights[ei] = mw / dx.shape[0]
                        self.event_peak[ei] = (self.new_weights[ei], self.new_means[ei, 0], self.new_means[ei, 1])
                    else:
                        # Handle case with no neighbors
                        self.event_peak[ei] = None

            print(f"  Avg # Neighbors: {total_neighbors/self.num_events}")        

            ######################
            # Build the spatial index on the initial points        
            self.pastPoints = copy.deepcopy(events)
            x = [e['x'] for e in events]
            y = [e['y'] for e in events]
            new_coords = np.array(list(zip(x, y)), dtype=np.int32)
            print(f"Init of points: {new_coords.shape}")

            self.past_events_tree = cKDTree(new_coords)            

            self.past_means = copy.deepcopy(self.new_means)
            self.past_vars = copy.deepcopy(self.new_vars)
            self.past_weights = copy.deepcopy(self.new_weights)

            # Now do spatial version
            if self.runSpatial:
                total_spatial_neighbors = 0
                for ei in range(new_coords.shape[0]):
                    indices_of_nearby_events = self.past_events_tree.query_ball_point(new_coords[ei], r=self.spatial_search_radius)
                    total_spatial_neighbors += len(indices_of_nearby_events)
            
        else:
            print(f"No events found this step!!!!!")
            self.past_events_tree = None
            self.past_means = None
            self.past_vars = None
            self.past_weights = None
            

        return self.event_peak
    