# events
Spatio temporal experiments

# DSEC dataset notes/updates
============================
conda install hdf5-plugin (for blosc) 
import hdf5-plugin in eventreader 

Manually patch scipy.video (abstract.py) to update for newer numpy: tostring() -> tobyte()

# Cmds

Minimum time for frame is about -dt 0.1  (ms) with about 1.5k events image (300k pixels)
Sensible time might be 1ms duration... ~15k events image (640x480 pixels)

python ..\src\extract_events.py c:\data\events\interlaken_00\interlaken_00_c_events_right\events.h5 -dt 0.1 -st 1 --min_frame 200 --max_frame 500 --output_events test_events.pkl --display
