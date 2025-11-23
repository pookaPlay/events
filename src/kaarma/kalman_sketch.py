import numpy as np


class KalmanFilter:
	"""A simple linear Kalman Filter for 2D position+velocity.

	State vector: [x, y, vx, vy]
	Observation: [x, y]
	"""

	def __init__(self, dt=1.0, process_var=1e-2, meas_var=1.0):
		self.dt = float(dt)

		# State transition (constant velocity model)
		self.F = np.array([
			[1, 0, self.dt, 0],
			[0, 1, 0, self.dt],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		], dtype=float)

		# Observation matrix: we observe x,y only
		self.H = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0]
		], dtype=float)

		# Process noise covariance (Q)
		q = float(process_var)
		# For a constant velocity model, use a simple diagonal-scaled Q
		self.Q = q * np.eye(4)

		# Measurement noise covariance (R)
		r = float(meas_var)
		self.R = r * np.eye(2)

		# Initial state and covariance
		self.x = np.zeros((4, 1), dtype=float)
		self.P = np.eye(4, dtype=float) * 1e3

	def set_initial(self, pos, vel=(0.0, 0.0), P0=None):
		"""Set initial state from position (x,y) and velocity (vx,vy)."""
		self.x = np.array([[float(pos[0])], [float(pos[1])], [float(vel[0])], [float(vel[1])]], dtype=float)
		if P0 is not None:
			self.P = np.array(P0, dtype=float)

	def predict(self, dt=None):
		"""Predict step: optionally update dt and F before predicting."""
		if dt is not None and dt != self.dt:
			self.dt = float(dt)
			self.F = np.array([
				[1, 0, self.dt, 0],
				[0, 1, 0, self.dt],
				[0, 0, 1, 0],
				[0, 0, 0, 1]
			], dtype=float)

		# x = F x
		self.x = self.F.dot(self.x)
		# P = F P F^T + Q
		self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
		return self.x.copy(), self.P.copy()

	def update(self, z):
		"""Update with measurement z = [x, y] (array-like)."""
		z = np.asarray(z, dtype=float).reshape((2, 1))
		# Innovation
		y = z - self.H.dot(self.x)
		S = self.H.dot(self.P).dot(self.H.T) + self.R
		# Kalman gain
		K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
		# Update state
		self.x = self.x + K.dot(y)
		# Update covariance
		I = np.eye(self.P.shape[0])
		self.P = (I - K.dot(self.H)).dot(self.P)
		return self.x.copy(), self.P.copy()


def cov_to_ellipse_params(cov_xy, nsig=2.0):
	"""Return ellipse parameters (angle in degrees, width, height) for given 2x2 covariance."""
	# eigenvalues and eigenvectors
	vals, vecs = np.linalg.eigh(cov_xy)
	# sort largest first
	order = vals.argsort()[::-1]
	vals = vals[order]
	vecs = vecs[:, order]
	# angle of largest eigenvector
	angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
	width, height = 2 * nsig * np.sqrt(np.maximum(vals, 0))
	return angle, width, height

