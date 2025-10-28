import numpy as np

class VelocityHistogram:
    """
    A 2D histogram for polar data, parameterized by radius and angle.
    Angles are expected in degrees [0, 360).
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
        
        # The histogram data structure, initialized to zeros
        self.histogram = np.zeros((num_radius_bins, num_angle_bins), dtype=np.int32)

    def add(self, radii: np.ndarray, angles: np.ndarray):
        """
        Add polar data points to the histogram.

        Args:
            radii: A numpy array of radius values.
            angles: A numpy array of angle values in degrees.
        """
        # Ensure angles are within [0, 360)
        angles = np.mod(angles, 360)

        # Calculate bin indices for radius
        # Clip radii to handle values exactly at max_radius or above
        radius_indices = np.floor(radii / self.max_radius * self.num_radius_bins).astype(int)
        radius_indices = np.clip(radius_indices, 0, self.num_radius_bins - 1)

        # Calculate bin indices for angle
        angle_indices = np.floor(angles / 360.0 * self.num_angle_bins).astype(int)
        angle_indices = np.clip(angle_indices, 0, self.num_angle_bins - 1)

        # Use np.add.at for an efficient and safe way to increment bins
        np.add.at(self.histogram, (radius_indices, angle_indices), 1)

    def clear(self):
        """Resets the histogram counts to zero."""
        self.histogram.fill(0)