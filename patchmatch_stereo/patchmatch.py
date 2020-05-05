import os
import numpy as np

class PatchMatch:
    def __init__(self, images, p_matrices, intrinsics):
        """"""
        self.images = images
        self.p_matrices = p_matrices
        self.intrinsics = intrinsics
        # initialize random depthmap
        self.depthmap = np.random.random((images[0].shape))
    
    def get_corresponding_img(self):
        """"""

    def run_iteration(self, image, direction):
        """Step1: get neighbor's depth value and a random value.
           Step2: Project the pixel to 3d space and get 3 3D points (use x = fX/Z)
           Step3: backproject all 3D points to the other image (multiply by p matrix)
           Step4: evaluate photoconsistency on 3 options
           Step5: Keep best option"""
           
        #    self.depthmap[i,j]
        
        
    
    
