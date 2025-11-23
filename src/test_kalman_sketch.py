"""
Test and visualize a simple Kalman filter on synthetic 2D data.

Run as a script to step through frames and see prediction, observation,
and uncertainty ellipse.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from kaarma.kalman_sketch import KalmanFilter, cov_to_ellipse_params


def generate_synthetic(T=100, dt=1.0, init_pos=(50, 50), velocity=(1.5, 0.7), process_noise=0.0, meas_noise=2.0):
    x = np.zeros((T, 2), dtype=float)
    vel = np.array(velocity, dtype=float)
    pos = np.array(init_pos, dtype=float)
    for t in range(T):
        # simple constant velocity motion
        pos = pos + vel * dt
        # optionally add process noise to ground truth
        if process_noise > 0:
            pos = pos + np.random.randn(2) * process_noise
        x[t] = pos.copy()
    # measurements
    z = x + np.random.randn(*x.shape) * meas_noise
    return x, z


def plot_frame(ax, true_pos, meas, pred_mean, pred_P):
    ax.clear()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    # true position
    ax.scatter(true_pos[0], true_pos[1], c='g', label='true')
    # measurement
    ax.scatter(meas[0], meas[1], c='r', label='meas')
    # prediction
    ax.scatter(pred_mean[0], pred_mean[1], c='b', label='pred')

    # draw covariance ellipse for position submatrix
    cov_xy = pred_P[:2, :2]
    angle, width, height = cov_to_ellipse_params(cov_xy, nsig=10.0)

    from matplotlib.patches import Ellipse
    ell = Ellipse(xy=(pred_mean[0], pred_mean[1]), width=width, height=height, angle=angle,
                  edgecolor='b', facecolor='none', linestyle='--', linewidth=1.5)
    ax.add_patch(ell)

    ax.legend(loc='upper right')
    ax.set_title('Kalman filter: true(g) meas(r) pred(b)')


def run_demo():
    # generate synthetic data
    T = 200
    true, meas = generate_synthetic(T=T, dt=1.0, init_pos=(20, 30), velocity=(1.2, 0.9), process_noise=0.0, meas_noise=3.0)

    kf = KalmanFilter(dt=1.0, process_var=0.01, meas_var=9.0)
    # initialize near first measurement
    kf.set_initial(pos=meas[0], vel=(0.0, 0.0), P0=np.eye(4) * 50.0)

    fig, ax = plt.subplots(figsize=(6, 6))

    pred_history = []
    for t in range(T):
        kf.predict()
        pred_mean, pred_P = kf.update(meas[t])
        pm = pred_mean.flatten()
        pred_history.append(pm)

        plot_frame(ax, true[t], meas[t], pm[:2], pred_P)
        plt.pause(0.05)

    plt.show()


if __name__ == '__main__':
    run_demo()
