import numpy as np
import matplotlib
# Use a non-interactive backend to prevent matplotlib from trying to open a window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=0,dtype='uint8')
    #img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    #img[mask==0]=[255,255,255]
    img[mask==0]=[0,0,0]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img

def render_flags(img, flagx, flagy):
    # Define circle properties
    color = (0, 255, 0)  # Green color in BGR format
    radius = 5           # Radius of the circle
    thickness = 1        # Thickness of the circle line (-1 for filled circle)
    for i in range(flagx.shape[0]):
        center = (int(flagx[i]), int(flagy[i]))
        cv2.circle(img, center, radius, color, thickness)

    return img

def render_polar_histogram(histogram, max_radius):

    if np.sum(histogram) == 0:
        # Return a black image if there's no data to plot
        return np.zeros((480, 640, 3), dtype=np.uint8)

    num_radius_bins, num_angle_bins = histogram.shape

    # Create the angle and radius grid for the plot
    theta = np.linspace(0.0, 2 * np.pi, num_angle_bins + 1)  # Angles in radians
    radii = np.linspace(0.0, max_radius, num_radius_bins + 1) # Radii

    # Create a figure and a polar subplot
    fig, ax = plt.subplots(figsize=(6.4, 4.8), subplot_kw={'projection': 'polar'})
    
    # Use pcolormesh to draw the histogram.
    # pcolormesh(Theta, R, C) expects C to have shape (num_radii, num_angles)
    c = ax.pcolormesh(theta, radii, histogram, cmap='viridis', shading='auto')

    ax.set_title("pretty")
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top
    ax.set_theta_direction(-1)       # Make angles go clockwise
    fig.colorbar(c, ax=ax, label="Event Count")

    # Convert the matplotlib plot to a numpy array
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    img_argb = np.frombuffer(buf, dtype=np.uint8)
    img_argb = img_argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert ARGB (matplotlib) to BGR (OpenCV), discarding the alpha channel
    img_bgr = cv2.cvtColor(img_argb, cv2.COLOR_RGBA2BGR)

    plt.close(fig)  # Close the figure to free memory

    return img_bgr
