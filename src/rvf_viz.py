import numpy as np
import matplotlib
# Use a non-interactive backend to prevent matplotlib from trying to open a window
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def plot_polar_histogram(histogram_data: np.ndarray, max_radius: float, title="Velocity Histogram") -> np.ndarray:
    """
    Visualizes a 2D polar histogram using matplotlib and returns it as an image array.

    Args:
        histogram_data (np.ndarray): The 2D histogram data (num_radius_bins, num_angle_bins).
        max_radius (float): The maximum radius used for the histogram.
        title (str): The title for the plot.

    Returns:
        np.ndarray: An BGR image of the plot as a numpy array, suitable for cv2.
    """
    if np.sum(histogram_data) == 0:
        # Return a black image if there's no data to plot
        return np.zeros((480, 640, 3), dtype=np.uint8)

    num_radius_bins, num_angle_bins = histogram_data.shape

    # Create the angle and radius grid for the plot
    theta = np.linspace(0.0, 2 * np.pi, num_angle_bins + 1)  # Angles in radians
    radii = np.linspace(0.0, max_radius, num_radius_bins + 1) # Radii

    # Create a figure and a polar subplot
    fig, ax = plt.subplots(figsize=(6.4, 4.8), subplot_kw={'projection': 'polar'})
    
    # Use pcolormesh to draw the histogram. The data needs to be transposed.
    # pcolormesh(Theta, R, C) expects C to have shape (num_angles, num_radii)
    c = ax.pcolormesh(theta, radii, histogram_data.T, cmap='viridis', shading='auto')

    ax.set_title(title)
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top
    ax.set_theta_direction(-1)       # Make angles go clockwise
    fig.colorbar(c, ax=ax, label="Event Count")

    # Convert the matplotlib plot to a numpy array
    fig.canvas.draw()
    img_rgba = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB (matplotlib) to BGR (OpenCV)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGB2BGR)

    plt.close(fig)  # Close the figure to free memory

    return img_bgr
