import copy
import numpy as np
from scipy.spatial import cKDTree
import cv2

class RVF_Gaussian:
    def __init__(self, search_radius=8.0, init_var=1.0, proc_var=1.0, alpha=1.0, spatial_search_radius=50.0):
        print(f"RVF")
        self.height = 800
        self.width = 800
        self.num_events = 0
        self.search_radius = search_radius
        self.spatial_search_radius = spatial_search_radius

        self.init_var = init_var
        self.var = proc_var        
        self.alpha = alpha

        self.runSpatial = False
        self.runPDA = True

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

                    # accumulate for RVF-PDA
                    mx = 0.0
                    my = 0.0
                    mv = 0.0
                    mw = 0.0
                    # RVF-NN
                    bestw = 0.0

                    for ni in range(dx.shape[0]):
                        nmean = self.past_means[indices_of_nearby_past_events[ni]]
                        nvar = self.past_vars[indices_of_nearby_past_events[ni]]

                        myvar = self.var * nvar / (nvar + self.var)
                        myx = dx[ni] * myvar / self.var + nmean[0] * myvar / nvar
                        myy = dy[ni] * myvar / self.var + nmean[1] * myvar / nvar
                        
                        # Use the specific neighbor's distance, not the whole array
                        scale_factor = np.sqrt(dx[ni]**2 + dy[ni]**2) / self.var
                        scale_factor += np.sqrt(nmean[0]*nmean[0] + nmean[1]*nmean[1]) / nvar
                        scale_factor -= np.sqrt(myx*myx + myy*myy) / myvar                    

                        myw = 1.0 / (self.var + nvar) *  np.exp( -0.5 * scale_factor)

                        if self.runPDA:
                            # Accumulate weighted values
                            mx += myx * myw
                            my += myy * myw
                            mv += myvar * myw
                            mw += myw
                        else:
                            if myw > mw:
                                mw = myw
                                mx = myx
                                my = myy
                                mv = myvar

                    # After iterating through all neighbors, if any were found,
                    # calculate the final mean and variance for the current event.
                    if mw > 0:
                        if self.runPDA:
                            self.new_means[ei, 0] = mx / mw
                            self.new_means[ei, 1] = my / mw
                            self.new_vars[ei] = mv / mw
                            self.new_weights[ei] = mw / dx.shape[0]
                            self.event_peak[ei] = (self.new_weights[ei], self.new_means[ei, 0], self.new_means[ei, 1])
                        else:
                            self.new_means[ei, 0] = mx
                            self.new_means[ei, 1] = my
                            self.new_vars[ei] = mv
                            self.new_weights[ei] = mw
                            self.event_peak[ei] = (self.new_weights[ei], self.new_means[ei, 0], self.new_means[ei, 1])
                    else:
                        # Handle case with no neighbors
                        self.event_peak[ei] = None

            print(f"  Avg # Neighbors: {total_neighbors/self.num_events}")        

            self.pastPoints = copy.deepcopy(events)
            # Build the spatial index on the initial points
            x = [e['x'] for e in events]
            y = [e['y'] for e in events]
            new_coords = np.array(list(zip(x, y)), dtype=np.int32)            

            self.past_events_tree = cKDTree(new_coords)            
            
            self.spatial_means = np.zeros((self.num_events, 2))
            self.spatial_vars = np.ones(self.num_events) * self.init_var
            self.spatial_weights = np.zeros(self.num_events)

            # Now do spatial version
            if self.runSpatial:
                print(f"Running spatial smoothing")
                total_spatial_neighbors = 0
                for ei in range(new_coords.shape[0]):
                    indices_of_nearby_events = self.past_events_tree.query_ball_point(new_coords[ei], r=self.spatial_search_radius)
                    total_spatial_neighbors += len(indices_of_nearby_events)

                    if len(indices_of_nearby_events) > 0: 

                        # for each neighbor accumulate 
                        for ni in range(len(indices_of_nearby_events)):
                            weighted_mean = self.new_means[indices_of_nearby_events[ni]] * self.new_weights[indices_of_nearby_events[ni]]
                            self.spatial_means[ei] += weighted_mean                        
                            weighted_var = self.new_vars[indices_of_nearby_events[ni]] * self.new_weights[indices_of_nearby_events[ni]]                        
                            self.spatial_vars[ei] += weighted_var
                            self.spatial_weights[ei] += self.new_weights[indices_of_nearby_events[ni]]
                        

                        self.spatial_means[ei] /= self.spatial_weights[ei]
                        self.spatial_means[ei] /= self.spatial_weights[ei]

                print(f"  Avg # Spatial Neighbors: {total_spatial_neighbors/new_coords.shape[0]}")        
                self.past_means = copy.deepcopy(self.spatial_means)
                self.past_vars = copy.deepcopy(self.spatial_vars)
                self.past_weights = copy.deepcopy(self.spatial_weights)

            else:
                self.past_means = copy.deepcopy(self.new_means)
                self.past_vars = copy.deepcopy(self.new_vars)
                self.past_weights = copy.deepcopy(self.new_weights)



        else:
            print(f"No events found this step!!!!!")
            self.past_events_tree = None
            self.past_means = None
            self.past_vars = None
            self.past_weights = None
            

        return self.event_peak
    