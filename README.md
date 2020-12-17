# Patch Match Stereo - 3D Computer Vision


## Patch Match
A randomized algorithm for quickly finding approximate nearest neighbor matches between image patches.
More information on https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/

## Project Description
### Objective
The goal of the assignment is to implement the multi-view patch algorithm for depthmap
generation. 

### Dataset
To this end you shall download the datasets fountain-P11, Herz-Jesu-P8, entry-P10
from the Strecha MVS evaluation website
 https://icwww.epfl.ch/~marquez/multiview/denseMVS.html
and use the provided camera pose and calibration information provided for each image.
Use the Patchmatch sample and propagation scheme alternating among the four image directions
(left-to-right, top-to-bottom, right-to-left, bottom to top) and report the progress after each
propagation direction.

### Procedure
a) Select three images from each of the datasets/scenes and generate for each a depth map
Show the resulting depth maps after each iteration.
b) Report the accuracy of each generated depth map compared to the available ground truth, by
1. Report the average pixel error for each of the depth map
2. Generate an error map (an image where the magnitude of the estimation error is stored at
the pixel position) using Matlab’s “jet” colormap for visualization
3. Plot the cumulative error distribution for each depth map
Notes:
- Start from a random depth initialization for each pixel in the depth map
- Use any window size photo-consistency measure you deem adequate. Justify your design
choice
Bonus: Use multiple photo-consistency measures and compare performance

### Running information
Expects a folder named 'data' with the relevant scenes from the Strecha Dataset, in the same repository.
