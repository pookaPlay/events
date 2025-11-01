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

    # do some comparison of features      #= event_features[pici].detach().numpy()
    set1 = event_features[0] # Features for the first circle
    set2 = event_features[3] # Features for the second, smaller circle
    set3 = event_features[4] # Features for the second, smaller circle
    set_rand = event_features[1] # Features for the random points    
    set_grid = event_features[2] # Features for the grid points
    set_square = event_features[5] # Features for the square points
    
    enc_dim = set1.shape[1]

    # Compare the two circles using the dot product.
    # We use einsum for a batched dot product: sum(v1_i.conj() * v2_i) for each point i
    # This computes the dot product for each corresponding pair of feature vectors.
    # The feature vectors from VecKM are normalized to sqrt(enc_dim).
    # So, the product of their norms is enc_dim.
    dot_products_circles = torch.einsum('...i,...i->...', set1.conj(), set2)
    similarity_circles = torch.abs(dot_products_circles) / enc_dim
    avg_similarity_circles = torch.mean(similarity_circles)
    print(f"Average similarity between circle 1 and circle 2: {avg_similarity_circles.item():.4f}")

    dot_products_circles = torch.einsum('...i,...i->...', set1.conj(), set3)
    similarity_circles = torch.abs(dot_products_circles) / enc_dim
    avg_similarity_circles = torch.mean(similarity_circles)
    print(f"Average similarity between circle 1 and circle 3: {avg_similarity_circles.item():.4f}")

    dot_products_circles = torch.einsum('...i,...i->...', set2.conj(), set3)
    similarity_circles = torch.abs(dot_products_circles) / enc_dim
    avg_similarity_circles = torch.mean(similarity_circles)
    print(f"Average similarity between circle 2 and circle 3: {avg_similarity_circles.item():.4f}")

    # For contrast, let's compare the first circle to the set of random points.
    # We expect a much lower similarity score here.
    dot_products_rand = torch.einsum('...i,...i->...', set1.conj(), set_rand)
    similarity_rand = torch.abs(dot_products_rand) / enc_dim
    avg_similarity_rand = torch.mean(similarity_rand)
    print(f"Average similarity between circle 1 and random points: {avg_similarity_rand.item():.4f}")

    dot_products_grid = torch.einsum('...i,...i->...', set1.conj(), set_grid)
    similarity_grid = torch.abs(dot_products_grid) / enc_dim
    avg_similarity_grid = torch.mean(similarity_grid)
    print(f"Average similarity between circle 1 and grid points: {avg_similarity_grid.item():.4f}")

    dot_products_square = torch.einsum('...i,...i->...', set1.conj(), set_square)
    similarity_square = torch.abs(dot_products_square) / enc_dim
    avg_similarity_square = torch.mean(similarity_square)
    print(f"Average similarity between circle 1 and square points: {avg_similarity_square.item():.4f}")

    dot_products_square = torch.einsum('...i,...i->...', set_grid.conj(), set_square)
    similarity_square = torch.abs(dot_products_square) / enc_dim
    avg_similarity_square = torch.mean(similarity_square)
    print(f"Average similarity between square and grid points: {avg_similarity_square.item():.4f}")

    key = cv2.waitKey(0) # Wait for a key press        
    cv2.destroyAllWindows()
