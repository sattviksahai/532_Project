import os
import numpy as np
from dataset import Dataset
from tqdm import tqdm
from photoconsistency import SAD

class PatchMatch:
    def __init__(self, images, p_matrices, intrinsics):
        """"""
        self.images = images
        self.p_matrices = p_matrices
        self.intrinsics = intrinsics
        self.ref_image = self.images['image'][0]
        self.ref_pmat = self.p_matrices.loc[self.p_matrices['name'] == self.images['name'][0]]['pmat'].values[0]
        self.ref_k = self.intrinsics.loc[self.intrinsics['name'] == self.images['name'][0]]['intrinsics'].values[0]
        # initialize random depthmap
        self.depthmap = np.random.rand(images['image'][0].shape[0], images['image'][0].shape[1])
        print("TODO: how to set limits for sampling depth?")
        print("TODO: implement get_corresponding_img")
        self.depth_lowerlim = 0
        self.depth_upperlim = 10
        self.state_table = ['top_to_bottom',
                            'bottom_to_top',
                            'left_to_right',
                            'right_to_left']
        self.other_img = self.images['name'][1]
        self.other_p_mat = self.p_matrices.loc[self.p_matrices['name'] == self.images['name'][1]]['pmat'].values[0]
    
    def get_corresponding_img(self, x,y):
        """"""
        return self.other_img, self.other_p_mat

    def project_to(self, loc, pmat, depth):

        print("TODO: check if its correct to use K and then also f/Z")
        """"""
        x = np.dot(self.ref_k, loc)
        x = x/x[2]
        X = np.ones(4)
        X[0] = (depth*x[0])/self.ref_k[0,0]
        X[1] = (depth*x[1])/self.ref_k[1,1]
        X[2] = depth
        x_ = np.dot(pmat, X)
        x_ = x_/x_[2]
        # print(x_)
        return x_[:-1]

    def eval_photoconsistency(self, original_pos, other_img, positions):
        """"""
        # scores = []
        # for pos in positions:
        #     scores.append(SAD(self.ref_image[original_pos[0], original_pos[1]], ))
        return 0

    def run_iteration(self, direction):
        """Step1: get neighbor's depth value and a random value.
           Step2: Project the pixel to 3d space and get 3 3D points (use x = fX/Z)
           Step3: backproject all 3D points to the other image (multiply by p matrix)
           Step4: evaluate photoconsistency on 3 options
           Step5: Keep best option"""
        
        for i in tqdm(range(self.depthmap.shape[0])):
            for j in range(self.depthmap.shape[1]):
                # Step1
                if direction == 'top_to_bottom':
                    if i != 0:
                        neighbor = self.depthmap[i-1,j]
                    else:
                        continue
                elif direction == 'bottom_to_top':
                    if i != self.depthmap.shape[0]-1:
                        neighbor = self.depthmap[i+1,j]
                    else:
                        continue
                elif direction == 'left_to_right':
                    if j != 0:
                        neighbor = self.depthmap[i,j-1]
                    else:
                        continue
                elif direction == 'right_to_left':
                    if j != self.depthmap.shape[1]-1:
                        neighbor = self.depthmap[i,j+1]
                    else:
                        continue
                else:
                    raise "invalid state"
                rand_val = np.random.uniform(low=self.depth_lowerlim, high=self.depth_upperlim)
                other_img, other_pmat = self.get_corresponding_img(i, j)
                # project with all depth hypotheses
                # # Step2&3
                neighbor_pos = self.project_to(np.array([i,j,1]), other_pmat, neighbor)
                rand_pos = self.project_to(np.array([i,j,1]), other_pmat, rand_val)
                original_pos = self.project_to(np.array([i,j,1]), other_pmat, self.depthmap[i,j])

                # Step4
                best_depth = self.eval_photoconsistency((i,j), other_img, [neighbor_pos, rand_pos, original_pos])

                # # Step5
                # if best_depth == 0:
                #     self.depthmap[i,j] = neighbor
                # elif best_depth == 1:
                #     self.depthmap[i,j] = rand_val
                # # else leave depth as it is

    def run(self, num_iterations):
        for it in range(num_iterations):
            self.run_iteration(self.state_table[it%len(self.state_table)])

        
        
    
if __name__ == "__main__":
    # read dataset
    data_path = "data/"
    mvs_dataset = Dataset(data_path)

    f_images = mvs_dataset.get_images('fountain')
    f_p_mats = mvs_dataset.get_p_matrices('fountain')
    f_intrinsics = mvs_dataset.get_intrinsics('fountain')

    pm_fountain = PatchMatch(f_images, f_p_mats, f_intrinsics)
    pm_fountain.run(1)
