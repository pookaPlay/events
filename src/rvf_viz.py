import numpy as np
import matplotlib
# Use a non-interactive backend to prevent matplotlib from trying to open a window
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

def RenderSynImage(events, H = 1000, W = 1000):

    img = np.full((H,W,3), fill_value=0,dtype='uint8')
    #img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')

    # extract data from events
    x = np.array([e['x'] for e in events], dtype=np.int32)
    y = np.array([e['y'] for e in events], dtype=np.int32)    
    pol = np.array([e['p'] for e in events], dtype=np.int32)
        
    pol[pol==0]=-1

    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)

    mask[y[mask1],x[mask1]]=pol[mask1]
    #img[mask==0]=[255,255,255]
    img[mask==0]=[0,0,0]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img

def RenderEventImage(events, peaks, H=1000, W=1000, scale=1.0):
    """
    Renders events as points and their corresponding velocity peaks as arrows.

    Args:
        events: A list of event dictionaries, each with 'x' and 'y'.
        peaks: A list of (value, radius, angle) tuples for each event.
        H: Image height.
        W: Image width.
        scale: A scaling factor for the length of the velocity arrows.
    """
    
    MAX_PEAK = np.max([peak[0] for peak in peaks])
    print(f"RenderEventIMage has max val of {MAX_PEAK}")

    img = np.full((H,W,3), fill_value=0,dtype='uint8')

    for i, event in enumerate(events):
        start_point = (int(event['x']), int(event['y']))

        # Draw the event as a small circle
        cv2.circle(img, start_point, radius=1, color=(0, 0, 255), thickness=-1) # Red dot for event

        peak = peaks[i]
        if peak is not None:
            val, radius, angle = peak

            # --- Visualize Uncertainty (val) ---
            # Map 'val' to a color. High val = high certainty = bright green. Low val = low certainty = yellow/red.
            # The max possible value for 'val' depends on the histogram size, but we can normalize it for visualization.
            # Let's assume a reasonable upper bound for 'val' for good color mapping, e.g., 0.1.
            # You might need to tune this based on typical values you observe.
            # expect max 0.4
            norm_val = min(val, MAX_PEAK) / MAX_PEAK
            
            # Interpolate from red (low certainty) to green (high certainty) in BGR format
            color = (0, int(255 * norm_val), int(255 * (1 - norm_val)))

            arrow_length = radius * scale
            end_point_x = int(start_point[0] + arrow_length * np.cos(angle))
            end_point_y = int(start_point[1] + arrow_length * np.sin(angle))
            end_point = (end_point_x, end_point_y)
            cv2.arrowedLine(img, start_point, end_point, color=color, thickness=1, tipLength=0.3)

    return img


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
    img[mask==1]=[255,255,255]
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
