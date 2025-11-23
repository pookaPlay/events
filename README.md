# Spatial-Temporal Event Experiments
====================================


# DSEC dataset notes/updates
conda install hdf5-plugin (for blosc) 
import hdf5-plugin in eventreader 

Manually patch scipy.video (abstract.py) to update for newer numpy: tostring() -> tobyte()

# Cmds
Minimum time for frame is about -dt 0.1  (ms) with about 1.5k events image (300k pixels)
Sensible time might be 1ms duration... ~15k events image (640x480 pixels)

python ..\src\extract_events.py c:\data\events\interlaken_00\interlaken_00_c_events_right\events.h5 -dt 0.1 -st 1 --min_frame 200 --max_frame 500 --output_events test_events.pkl --display

This is a different format to the js simulator so we convert with
python test_events.py test_events.pkl => generates test_events.json

python test_rvf.py test_events.json


# Kalman filter example 
Write a basic kalman filter with predict and update steps. Write a test for the kalman filter using 2d synthetic data (in time) with a noisy point moving across the screen. During the test I want to step through the data and visualize the prediction and associated uncertainty.
