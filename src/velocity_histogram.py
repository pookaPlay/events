import numpy as np

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
        
        self.clear()

    def add_events(self, radii: np.ndarray, angles: np.ndarray):
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


    def multiply(self, v1, v2):
        self.histogram = v1.histogram * v2.histogram
        self.normalize()

    def add(self, v1, v2 = None):
        if v2 is not None:
            self.histogram = v1.histogram + v2.histogram
            self.normalize()
        else:
            self.histogram = self.histogram + v1.histogram

    def normalize(self):
        if np.sum(self.histogram) > 1e-10:            
            self.histogram /= np.sum(self.histogram)
        else:
            self.clear()            

    def peak(self):
        # get max value and index
        val = np.max(self.histogram)
        # get radius and angle
        idx = np.unravel_index(np.argmax(self.histogram), self.histogram.shape)
        radius = idx[0] * self.max_radius / self.num_radius_bins
        angle = idx[1] * 2 * np.pi / self.num_angle_bins
        
        return val, radius, angle
        

    def clear(self):
        """Resets the histograms to uniform."""
        # The histogram data structure, initialized to zeros
        self.histogram = np.ones((self.num_radius_bins, self.num_angle_bins), dtype=np.float64)
        self.normalize()
