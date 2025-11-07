import numpy as np
from scipy.ndimage import convolve

class VelocityHistogram:
    """
    A 2D histogram for polar data, parameterized by radius and angle.
    Angles are expected in radians.
    """
    def __init__(self, num_radius_bins: int, num_angle_bins: int, max_radius: float):
        """
        Initializes the polar histogram.

        Args:
            num_radius_bins: The number of bins for the radius (speed).
            num_angle_bins: The number of bins for the angle (direction).
            max_radius: The maximum radius value. Radii larger than this will be clipped.
        """
        if not (num_radius_bins > 0 and num_angle_bins > 0 and max_radius > 0):
            raise ValueError("Number of bins and max_radius must be positive.")
            
        self.num_radius_bins = num_radius_bins
        self.num_angle_bins = num_angle_bins
        self.max_radius = float(max_radius)
        
        self.smooth_radius = 3
        self.smooth_angle = 3

        # Create 1D Gaussian kernels for radius and angle
        gauss_r = np.exp(-(np.linspace(-self.smooth_radius, self.smooth_radius, 2 * self.smooth_radius + 1) ** 2) / (2 * (self.smooth_radius/2) ** 2))
        gauss_a = np.exp(-(np.linspace(-self.smooth_angle, self.smooth_angle, 2 * self.smooth_angle + 1) ** 2) / (2 * (self.smooth_angle/2) ** 2))

        # Create a 2D kernel by taking the outer product
        self.smooth_kernel = np.outer(gauss_r, gauss_a)

        # The histogram data structure, initialized to zeros
        self.clear()

    def clear(self):
        """Resets the histograms to uniform."""
        # The histogram data structure, initialized to zeros
        self.histogram = np.ones((self.num_radius_bins, self.num_angle_bins), dtype=np.float64)
        self.normalize()

    def add_events_normalize(self, radii: np.ndarray, angles: np.ndarray):
        """
        Add polar data points to the histogram.

        Args:
            radii: A numpy array of radius values.
            angles: A numpy array of angle values in radians.
        """
        if radii.size == 0:
            return

        # Normalize angles to be within [0, 2*pi)
        angles = np.mod(angles, 2 * np.pi)

        # --- Bilinear Interpolation for Smoother Histogram ---

        # Calculate continuous bin coordinates
        r_cont = np.clip(radii, 0, self.max_radius) / self.max_radius * self.num_radius_bins
        a_cont = angles / (2 * np.pi) * self.num_angle_bins

        # Get the indices of the four surrounding bins
        r_idx0 = np.floor(r_cont - 0.5).astype(int)
        a_idx0 = np.floor(a_cont - 0.5).astype(int)

        # Calculate the weights for each of the 4 bins
        # The weights are the areas of the rectangles opposite to each corner
        dr = r_cont - (r_idx0 + 0.5)
        da = a_cont - (a_idx0 + 0.5)

        w00 = (1 - dr) * (1 - da)
        w01 = (1 - dr) * da
        w10 = dr * (1 - da)
        w11 = dr * da

        # Get the indices of the four bins, handling boundaries
        r_idx0_c = np.clip(r_idx0, 0, self.num_radius_bins - 1)
        r_idx1_c = np.clip(r_idx0 + 1, 0, self.num_radius_bins - 1)
        
        # Handle angle wrapping
        a_idx0_c = np.mod(a_idx0, self.num_angle_bins)
        a_idx1_c = np.mod(a_idx0 + 1, self.num_angle_bins)

        # Add the weighted contributions to the four neighboring bins
        # np.add.at is used for safe, unbuffered addition to the same bin from multiple points
        np.add.at(self.histogram, (r_idx0_c, a_idx0_c), w00)
        np.add.at(self.histogram, (r_idx0_c, a_idx1_c), w01)
        np.add.at(self.histogram, (r_idx1_c, a_idx0_c), w10)
        np.add.at(self.histogram, (r_idx1_c, a_idx1_c), w11)
        self.normalize()        

    def multiply_alpha_normalize(self, v1, alpha, v2):
        # raise each element in histogram to the alpha power
        self.histogram = np.power(v1.histogram, alpha)
        self.histogram = self.histogram * v2.histogram
        self.normalize()

    def multiply_normalize(self, v1, v2):
        self.histogram = v1.histogram * v2.histogram
        self.normalize()

    def multiply(self, v1, v2):
        self.histogram = v1.histogram * v2.histogram        

    def add_normalize(self, v1, v2):        
        self.histogram = v1.histogram + v2.histogram
        self.normalize()

    def add(self, v1):
        self.histogram = self.histogram + v1.histogram

    def normalize(self):
        if np.sum(self.histogram) > 1e-10:            
            self.histogram /= np.sum(self.histogram)
        else:
            self.clear()            

    def smooth(self):
        """
        Applies a 2D convolution to smooth the histogram.
        It handles the circular nature of the angle dimension by padding before convolution.
        """
        # The angle dimension is circular. We pad the histogram with wrapped angle data
        # to ensure the convolution handles the edges correctly.
        pad_width = self.smooth_angle // 2
        padded_hist = np.pad(self.histogram, ((0, 0), (pad_width, pad_width)), mode='wrap')

        # Perform the 2D convolution on the padded histogram
        smoothed_padded_hist = convolve(padded_hist, self.smooth_kernel, mode='constant', cval=0.0)

        # Remove the padding to get back to the original histogram size
        self.histogram = smoothed_padded_hist[:, pad_width:-pad_width]
        self.normalize()

    def peak(self):
        # get max value and index
        val = np.max(self.histogram)
        # get radius and angle
        idx = np.unravel_index(np.argmax(self.histogram), self.histogram.shape)
        radius = idx[0] * self.max_radius / self.num_radius_bins
        angle = idx[1] * 2 * np.pi / self.num_angle_bins
        
        return val, radius, angle
        
    def PredictNextLocation(self, loc):
        """
        Predicts the next location of a point based on histogram.

        Args:
            loc: The current (x, y) location of the point as a tuple or list.

        Returns:
            The predicted next (x, y) location as a tuple.
        """
        # 1. Create arrays representing the center radius and angle for each bin.
        # The shape of each will be (num_radius_bins, num_angle_bins).
        radius_vals = (np.arange(self.num_radius_bins) + 0.5) * (self.max_radius / self.num_radius_bins)
        angle_vals = (np.arange(self.num_angle_bins) + 0.5) * (2 * np.pi / self.num_angle_bins)
        
        # Create a grid of radius and angle values for each bin in the histogram.
        # `indexing='ij'` ensures the grid matches the histogram's (radius, angle) layout.
        radii_grid, angles_grid = np.meshgrid(radius_vals, angle_vals, indexing='ij')

        # 2. Convert the polar velocities of each bin to Cartesian vectors (vx, vy).
        vx_grid = radii_grid * np.cos(angles_grid)
        vy_grid = radii_grid * np.sin(angles_grid)

        # 3. Calculate the expected velocity by taking a weighted sum.
        # The histogram values are the weights. Since the histogram is normalized,
        # this is the expected value of the velocity distribution.
        expected_vx = np.sum(vx_grid * self.histogram)
        expected_vy = np.sum(vy_grid * self.histogram)

        # 4. Calculate the predicted next location by adding the expected velocity vector.
        return (loc[0] + expected_vx, loc[1] + expected_vy)
