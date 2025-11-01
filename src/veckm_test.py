import pickle
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from VecKM import  ExactVecKM
import torch
from rvf_viz import render

if __name__ == '__main__':
    #parser = argparse.ArgumentParser('veckm_events')
    #parser.add_argument('event_file', type=str, help='Path to events.pkl file')
    #args = parser.parse_args()
    #event_filepath = Path(args.event_file)

    height = 480
    width = 640
    radius = 120
    nradius = 100
    N = 64
    
    # Generate synthetic event data 
    # Events are 2d points (x,y) 
    # Sample a circle at N points
    # Circle should be in middle of image (width, height) and have radius "radius" 
    center_x = width / 2
    center_y = height / 2
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_coords = center_x + radius * np.cos(angles)
    y_coords = center_y + radius * np.sin(angles)

    events = dict()
    tevents = dict()

    events[0] = dict()
    events[0]['x'] = x_coords.astype(np.int32)
    events[0]['y'] = y_coords.astype(np.int32)
    events[0]['p'] = np.ones(N, dtype=np.int32) # Polarity
    
    # Generate synthetic event data 
    # for points drawn uniformly at random across the image
    rand_x_coords = np.random.randint(0, width, N)
    rand_y_coords = np.random.randint(0, height, N)

    events[1] = dict()
    events[1]['x'] = rand_x_coords.astype(np.int32)
    events[1]['y'] = rand_y_coords.astype(np.int32)
    events[1]['p'] = np.ones(N, dtype=np.int32) # Polarity    

    # Generate synthetic event data 
    # for regularly sampled points on the 2d image of size sqrt(N) by sqrt(N)
    grid_size = int(np.sqrt(N))
    x_ticks = np.linspace(0, width - 1, grid_size, dtype=np.int32)
    y_ticks = np.linspace(0, height - 1, grid_size, dtype=np.int32)
    xv, yv = np.meshgrid(x_ticks, y_ticks)
    events[2] = dict()
    events[2]['x'] = xv.flatten()
    events[2]['y'] = yv.flatten()
    events[2]['p'] = np.ones(N, dtype=np.int32)
    
    # Generate synthetic event data 
    # Events are 2d points (x,y) 
    # Sample a circle at N points
    # Circle should be in middle of image (width, height) and have radius "radius" 
    center_x = (width / 2) + (width / 4)
    center_y = (height / 2) + (height / 4)
    radiusFactor = 0.75
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_coords = center_x + radiusFactor * radius * np.cos(angles)
    y_coords = center_y + radiusFactor * radius * np.sin(angles)

    events[3] = dict()
    events[3]['x'] = x_coords.astype(np.int32)
    events[3]['y'] = y_coords.astype(np.int32)
    events[3]['p'] = np.ones(N, dtype=np.int32) # Polarity

    # Generate synthetic event data 
    # Events are 2d points (x,y) 
    # Sample a circle at N points
    # Circle should be in middle of image (width, height) and have radius "radius" 
    center_x = (width / 2) - (width / 4)
    center_y = (height / 2) - (height / 4)
    radiusFactor = 0.75
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_coords = center_x + radiusFactor * radius * np.cos(angles)
    y_coords = center_y + radiusFactor * radius * np.sin(angles)

    events[4] = dict()
    events[4]['x'] = x_coords.astype(np.int32)
    events[4]['y'] = y_coords.astype(np.int32)
    events[4]['p'] = np.ones(N, dtype=np.int32) # Polarity

    # Generate synthetic event data 
    # Events are 2d points (x,y) 
    # Sample a square in middle of image (width, height) and have side length "side" 
    center_x = (width / 2) 
    center_y = (height / 2)
    side = 100
    
    # Calculate number of points per side
    points_per_side = N // 4
    rem = N % 4

    # Generate points for each side of the square
    half_side = side / 2
    x_coords_sq = np.array([])
    y_coords_sq = np.array([])

    # Distribute points, including remainder
    pts_top = points_per_side + (1 if rem > 0 else 0)
    pts_right = points_per_side + (1 if rem > 1 else 0)
    pts_bottom = points_per_side + (1 if rem > 2 else 0)
    pts_left = points_per_side

    # Top side (left to right)
    x_coords_sq = np.append(x_coords_sq, np.linspace(center_x - half_side, center_x + half_side, pts_top, endpoint=False))
    y_coords_sq = np.append(y_coords_sq, np.full(pts_top, center_y - half_side))

    # Right side (top to bottom)
    x_coords_sq = np.append(x_coords_sq, np.full(pts_right, center_x + half_side))
    y_coords_sq = np.append(y_coords_sq, np.linspace(center_y - half_side, center_y + half_side, pts_right, endpoint=False))

    # Bottom side (right to left)
    x_coords_sq = np.append(x_coords_sq, np.linspace(center_x + half_side, center_x - half_side, pts_bottom, endpoint=False))
    y_coords_sq = np.append(y_coords_sq, np.full(pts_bottom, center_y + half_side))

    # Left side (bottom to top)
    x_coords_sq = np.append(x_coords_sq, np.full(pts_left, center_x - half_side))
    y_coords_sq = np.append(y_coords_sq, np.linspace(center_y + half_side, center_y - half_side, pts_left, endpoint=False))

    events[5] = dict()
    events[5]['x'] = x_coords_sq.astype(np.int32)
    events[5]['y'] = y_coords_sq.astype(np.int32)
    events[5]['p'] = np.ones(N, dtype=np.int32) # Polarity

    print(f"Events 5: {events[5]['x'].shape}")
    print(f"Events 5: {events[5]['y'].shape}")
    print(f"Events 5: {events[5]['p'].shape}")


    event_features = dict()

    for pici in range(6):
        
        tevents = torch.from_numpy(np.stack((events[pici]['x'], events[pici]['y']), axis=1)).float()    
        # The radius must be large enough to find neighbors. For the 5x5 grid, points are ~160px apart.
        vec = ExactVecKM(2, 256, 170.0)
        event_features[pici] = vec.forward(tevents)        

        print(f"Event features shape: {event_features[pici].shape}")
        # G = G / torch.norm(G, dim=-1, keepdim=True) * self.sqrt_d               # Complex(n, d)

        img = render(events[pici]['x'], events[pici]['y'], events[pici]['p'], height, width)
        
        title = f"Event {pici}"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, img)

    def calculate_and_print_similarity(set_a, set_b, name_a, name_b, enc_dim):
        
        ## shuffle the order of set b
        #perm = torch.randperm(set_b.shape[0])
        #set_b = set_b[perm]
        
        # This implements Strategy 2: a correspondence-free comparison (Chamfer-like).
        # For each point in set_a, we find its best match in set_b.

        # 1. Compute the pairwise similarity matrix between all points in A and all points in B.
        # The shape will be (num_points_a, num_points_b).
        # einsum 'ai,bi->ab' computes the dot product between each vector in a and each in b.
        pairwise_dot_products = torch.einsum('ai,bi->ab', set_a.conj(), set_b)
        pairwise_similarity = torch.abs(pairwise_dot_products) / enc_dim

        # 2. For each point in set_a, find the similarity of its best match in set_b.
        # We do this by taking the maximum value along each row of the similarity matrix.
        best_matches_for_a, _ = torch.max(pairwise_similarity, dim=1)

        # 3. The overall similarity is the average of these best-match scores.
        similarity = best_matches_for_a

        avg_similarity = torch.mean(similarity)
        print(f"Average similarity between {name_a} and {name_b}: {avg_similarity.item():.4f}")

    # do some comparison of features      #= event_features[pici].detach().numpy()
    feature_sets = {
        "circle 1": event_features[0],
        "random points": event_features[1],
        "grid points": event_features[2],
        "circle 2": event_features[3],
        "circle 3": event_features[4],
        "square": event_features[5]
    }
    enc_dim = feature_sets["circle 1"].shape[1]
    
    calculate_and_print_similarity(feature_sets["circle 1"], feature_sets["circle 2"], "circle 1", "circle 2", enc_dim)
    calculate_and_print_similarity(feature_sets["circle 1"], feature_sets["circle 3"], "circle 1", "circle 3", enc_dim)
    calculate_and_print_similarity(feature_sets["circle 2"], feature_sets["circle 3"], "circle 2", "circle 3", enc_dim)
    calculate_and_print_similarity(feature_sets["circle 1"], feature_sets["random points"], "circle 1", "random points", enc_dim)
    calculate_and_print_similarity(feature_sets["circle 1"], feature_sets["grid points"], "circle 1", "grid points", enc_dim)
    calculate_and_print_similarity(feature_sets["circle 1"], feature_sets["square"], "circle 1", "square", enc_dim)
    calculate_and_print_similarity(feature_sets["grid points"], feature_sets["square"], "grid points", "square", enc_dim)

    # --- How similar are points on the SAME circle to each other? ---
    set1 = feature_sets["circle 1"]
    # Calculate the full pairwise similarity matrix (N x N) for the first circle.
    pairwise_similarity_circle1 = torch.abs(torch.einsum('ai,bi->ab', set1.conj(), set1)) / enc_dim
    # The diagonal of this matrix is 1.0 (self-similarity). We want the average of the off-diagonal elements.
    N = set1.shape[0]
    # Sum all elements and subtract the trace (sum of diagonal), then divide by the number of off-diagonal elements.
    sum_off_diagonal = torch.sum(pairwise_similarity_circle1) - torch.trace(pairwise_similarity_circle1)
    num_off_diagonal = N * (N - 1)
    avg_self_similarity = sum_off_diagonal / num_off_diagonal
    print(f"\nAverage self-similarity of points on circle 1: {avg_self_similarity.item():.4f}")

    key = cv2.waitKey(0) # Wait for a key press        
    cv2.destroyAllWindows()
