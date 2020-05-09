import os
import numpy as np
from dataset import Dataset
from tqdm import tqdm
from photoconsistency import SAD

class PatchMatch:
    def __init__(self, images, p_matrices, intrinsics, rotations):
        """"""
        self.images = images
        self.p_matrices = p_matrices
        self.intrinsics = intrinsics
        self.rotations = rotations
        self.ref_image = self.images['image'][0]
        self.ref_pmat = self.p_matrices.loc[self.p_matrices['name'] == self.images['name'][0]]['pmat'].values[0]
        self.ref_k = self.intrinsics.loc[self.intrinsics['name'] == self.images['name'][0]]['intrinsics'].values[0]
        self.ref_r = self.rotations.loc[self.rotations['name'] == self.images['name'][0]]['rotations'].values[0]
        # initialize random depthmap
        self.depth_lowerlim = 1000
        self.depth_upperlim = 50000
        self.depthmap = (np.random.rand(images['image'][0].shape[0], images['image'][0].shape[1]) + self.depth_lowerlim)*self.depth_upperlim
        print("TODO: how to set limits for sampling depth?")
        print("TODO: implement get_corresponding_img")
        self.state_table = ['top_to_bottom',
                            'bottom_to_top',
                            'left_to_right',
                            'right_to_left']
        self.other_img = self.images['image'][1]
        self.other_p_mat = self.p_matrices.loc[self.p_matrices['name'] == self.images['name'][1]]['pmat'].values[0]
        self.window_size = 3
    
    def get_corresponding_img(self, x,y):
        """"""
        return self.other_img, self.other_p_mat

    def project_to_3d(self, pos, z, K, R):
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        u = pos[0]
        v = pos[1]
        x = (u-cx)*z/fx
        y = (v-cy)*z/fy
        ret = np.dot(R,np.array([[x,y,z]]).T)
        return np.array([ret[0],ret[1],ret[2],1])

    def project_to_2d(self, pos, P):
        ret = np.dot(P,pos)
        return ret/ret[2]
    
    def project_to(self, loc, depth, K0, R0, P1):
        # # print("TODO: check if its correct to use K and then also f/Z")
        # """"""
        # x = np.dot(self.ref_k, loc)
        # x = x/x[2]
        # X = np.ones(4)
        # X[0] = (depth*x[0])/self.ref_k[0,0]
        # X[1] = (depth*x[1])/self.ref_k[1,1]
        # X[2] = depth
        # x_ = np.dot(pmat, X)
        # x_ = x_/x_[2]
        # print(x, X, x_)
        # return x_[:-1]

        # point_3d = self.project_to_3d([1500, 1000, 1], 15000, K0, R0)
        point_3d = self.project_to_3d(loc, depth, K0, R0)
        point_reprojected = self.project_to_2d(point_3d, P1)
        # print("{} maps to {}".format(loc, point_reprojected))
        return np.array([point_reprojected[1], point_reprojected[0]])

    def eval_photoconsistency(self, original_pos, other_img, positions):
        """"""
        # print(self.ref_image.shape)
        scores = []
        for pos in positions:
            if (int(pos[0])>int(self.window_size/2)) and (int(pos[1])>int(self.window_size/2)) and (int(pos[0])<self.ref_image.shape[0]-int(self.window_size/2)) and (int(pos[1])<self.ref_image.shape[1]-int(self.window_size/2)) and (original_pos[0]>int(self.window_size/2)) and (original_pos[1]>int(self.window_size/2)) and (original_pos[0]<self.ref_image.shape[0]-int(self.window_size/2)) and (original_pos[1]<self.ref_image.shape[1]-int(self.window_size/2)):
                # print(self.ref_image[original_pos[0]-int(self.window_size/2):original_pos[0]+1+int(self.window_size/2), original_pos[1]-int(self.window_size/2):original_pos[1]+1+int(self.window_size/2)].shape)
                l1 = int(pos[0])-int(self.window_size/2)
                h1 = int(pos[0])+int(self.window_size/2)+1
                l2 = int(pos[1])-int(self.window_size/2)
                h2 = int(pos[1])+int(self.window_size/2)+1
                # print(other_img[l1:h1,l2:h2,:].shape)
                try:
                    scores.append(SAD(self.ref_image[original_pos[0]-int(self.window_size/2):original_pos[0]+1+int(self.window_size/2), original_pos[1]-int(self.window_size/2):original_pos[1]+1+int(self.window_size/2),:], other_img[l1:h1,l2:h2,:]))
                except:
                    print(original_pos, pos)
            else:
                scores.append(np.inf)
        return np.argmin(np.array(scores))

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
                neighbor_pos = self.project_to(np.array([j,i,1]), neighbor, self.ref_k, self.ref_r, other_pmat)
                rand_pos = self.project_to(np.array([j,i,1]), rand_val, self.ref_k, self.ref_r, other_pmat)
                original_pos = self.project_to(np.array([j,i,1]), self.depthmap[i,j], self.ref_k, self.ref_r, other_pmat)
                # neighbor_pos = self.project_to(np.array([j,i,1]), other_pmat, neighbor)
                # rand_pos = self.project_to(np.array([j,i,1]), other_pmat, rand_val)
                # original_pos = self.project_to(np.array([j,i,1]), other_pmat, self.depthmap[i,j])

                # Step4
                best_depth = self.eval_photoconsistency((i,j), other_img, [neighbor_pos, rand_pos, original_pos])

                # Step5
                if best_depth == 0:
                    self.depthmap[i,j] = neighbor
                elif best_depth == 1:
                    self.depthmap[i,j] = rand_val
                # else leave depth as it is
        np.save("depth.npy", self.depthmap)

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
    f_rotations = mvs_dataset.get_rotations('fountain')

    pm_fountain = PatchMatch(f_images, f_p_mats, f_intrinsics, f_rotations)
    pm_fountain.run(1)
