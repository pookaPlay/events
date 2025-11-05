import pickle
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from VecKM import  ExactVecKM, get_adj_matrix
import torch
from rvf_viz import render

def calculate_corresponding_similarity(set_a, set_b, randperm = False):
    
    assert set_a.shape[0] == set_b.shape[0], "Sets must have the same number of points for correspondence-based similarity."
    assert set_a.shape[1] == set_b.shape[1], "Sets must have the same dimension."
    enc_dim = set_a.shape[1]

    # permute set_b to test how invariant
    if randperm:
        #perm = torch.randperm(set_b.shape[0])
        # permutation for one next in sequence
        perm = (torch.arange(set_b.shape[0]) + 100) % set_b.shape[0]
        set_b = set_b[perm]

    # Calculate dot product between corresponding vectors.
    dot_products = torch.einsum('ai,ai->a', set_a.conj(), set_b)
    similarities = torch.abs(dot_products) / enc_dim
    
    # Now do something
    avg_similarity = torch.mean(similarities)

    return avg_similarity    

def calculate_pairwise_similarity(set_a, set_b):
    
    #assert set_a.shape[0] == set_b.shape[0], "Sets must have the same number of points for correspondence-based similarity."
    assert set_a.shape[1] == set_b.shape[1], "Sets must have the same dimension."
    enc_dim = set_a.shape[1]

    pairwise_dot_products = torch.einsum('ai,bi->ab', set_a.conj(), set_b)
    pairwise_similarity = torch.abs(pairwise_dot_products) / enc_dim
    
    # The diagonal of this matrix is 1.0 (self-similarity). We want the average of the off-diagonal elements.
    N = set_a.shape[0]
    # Sum all elements and subtract the trace (sum of diagonal), then divide by the number of off-diagonal elements.
    diag = torch.trace(pairwise_similarity)
    sum_off_diagonal = torch.sum(pairwise_similarity) - diag
    num_off_diagonal = N * (N - 1)
    avg_self_similarity = sum_off_diagonal / num_off_diagonal    
    diag_self = diag / N
    return avg_self_similarity, diag_self



if __name__ == '__main__':
    #parser = argparse.ArgumentParser('veckm_events')
    #parser.add_argument('event_file', type=str, help='Path to events.pkl file')
    #args = parser.parse_args()
    #event_filepath = Path(args.event_file)
    
    height = 480
    width = 640
    radius = 64
    alpha = 5.0
    
    # Test multiple neighborhood radii to find optimal
    # Also test different point densities
    nradius_candidates = [3, 5, 8, 10, 15, 20]
    embdim = 256
    # Test with different numbers of points to see if sparsity helps
    N_candidates = [100, 200, 500, 1000]
    N = 200  # Start with moderate density
    
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
    np.random.seed(42)  # For reproducibility
    rand_x_coords = np.random.randint(0, width, N)
    rand_y_coords = np.random.randint(0, height, N)

    events[1] = dict()
    events[1]['x'] = rand_x_coords.astype(np.int32)
    events[1]['y'] = rand_y_coords.astype(np.int32)
    events[1]['p'] = np.ones(N, dtype=np.int32) # Polarity    

    vec = ExactVecKM(2, embdim, radius, alpha)
    event_features = dict()
    
    for pici in range(2):
        tevents = torch.from_numpy(np.stack((events[pici]['x'], events[pici]['y']), axis=1)).float()    
        event_features[pici] = vec.forward(tevents)        

        img = render(events[pici]['x'], events[pici]['y'], events[pici]['p'], height, width)
        title = f"Event {pici} (radius={radius})"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, img)

    set1 = event_features[0]  # Circle
    set2 = event_features[1]  # Noise

    enc_dim = set1.shape[1]
        
    print(f"\n=== Similarity Comparisons ===")
    print(f"Comparing circle to circle")
    corr1 = calculate_corresponding_similarity(set1, set1, randperm=False)
    corr2 = calculate_corresponding_similarity(set1, set1, randperm=True)        
    pair, diag = calculate_pairwise_similarity(set1, set1)    
    print(f"S1-S1 Pairwise: {pair.item():.4f} and {diag.item():.4f}     Corresponding: {corr1.item():.4f} and {corr2.item():.4f}")

    print(f"\nComparing circle to uniform") 
    corr1 = calculate_corresponding_similarity(set1, set2, randperm=False)
    corr2 = calculate_corresponding_similarity(set1, set2, randperm=True)        
    pair, diag = calculate_pairwise_similarity(set1, set2)    
    print(f"S1-S2 Pairwise : {pair.item():.4f} and {diag.item():.4f}    Corresponding : {corr1.item():.4f} and {corr2.item():.4f}")

    print(f"\nComparing uniform to uniform") 
    corr1 = calculate_corresponding_similarity(set2, set2, randperm=False)
    corr2 = calculate_corresponding_similarity(set2, set2, randperm=True)        
    pair, diag = calculate_pairwise_similarity(set2, set2)    
    print(f"S1-S2 Pairwise : {pair.item():.4f} and {diag.item():.4f}    Corresponding : {corr1.item():.4f} and {corr2.item():.4f}")

    key = cv2.waitKey(0) # Wait for a key press        
    cv2.destroyAllWindows()
