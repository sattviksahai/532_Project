import os
import numpy as np
from dataset import Dataset
import cv2
from tqdm import tqdm
from photoconsistency import SAD

class PatchMatch:
    def __init__(self, images, p_matrices, intrinsics, rotations, translations):
        """"""
        # Read reference image and params
        self.ref_image = images['image'][0]
        self.ref_name = images['name'][0]
        self.ref_pmat = p_matrices['pmat'][0]
        self.ref_k = intrinsics['intrinsics'][0]
        self.ref_r = rotations['rotation'][0]
        self.ref_t = translations['translation'][0]
        # Read other image 1 and params
        self.other_image_1 = images['image'][1]
        self.other_name_1 = images['name'][1]
        self.other_pmat_1 = p_matrices['pmat'][1]
        self.other_k_1 = intrinsics['intrinsics'][1]
        self.other_r_1 = rotations['rotation'][1]
        self.other_t_1 = translations['translation'][1]
        # Read other image 2 and params
        self.other_image_2 = images['image'][2]
        self.other_name_2 = images['name'][2]
        self.other_pmat_2 = p_matrices['pmat'][2]
        self.other_k_2 = intrinsics['intrinsics'][2]
        self.other_r_2 = rotations['rotation'][2]
        self.other_t_2 = translations['translation'][2]

        ########
        # self.images = images
        # self.p_matrices = p_matrices
        # self.intrinsics = intrinsics
        # self.rotations = rotations
        # self.translations = translations
        # self.ref_image = self.images['image'][0]
        # self.ref_pmat = self.p_matrices.loc[self.p_matrices['name'] == self.images['name'][0]]['pmat'].values[0]
        # self.ref_k = self.intrinsics.loc[self.intrinsics['name'] == self.images['name'][0]]['intrinsics'].values[0]
        # self.ref_r = self.rotations.loc[self.rotations['name'] == self.images['name'][0]]['rotation'].values[0]
        # self.ref_t = self.translations.loc[self.translations['name'] == self.images['name'][0]]['translation'].values[0]
        self.depth_lowerlim = 1000
        self.depth_upperlim = 50000
        self.depthmap = (np.random.rand(images['image'][0].shape[0], images['image'][0].shape[1]) + self.depth_lowerlim)*self.depth_upperlim
        print("TODO: how to set limits for sampling depth?")
        print("TODO: implement get_corresponding_img")
        self.state_table = ['top_to_bottom',
                            'bottom_to_top',
                            'left_to_right',
                            'right_to_left']
        # self.other_img = self.images['image'][1]
        # self.other_p_mat = self.p_matrices.loc[self.p_matrices['name'] == self.images['name'][1]]['pmat'].values[0]
        self.window_size = 3

    def project_to_3d(self, pos, z, K, R, T):
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        u = pos[0]
        v = pos[1]
        x = (u-cx)*z/fx
        y = (v-cy)*z/fy
        ret = np.dot(R,np.array([[x,y,z]]).T) - np.dot(R,T)
        return np.array([ret[0],ret[1],ret[2],1])

    def project_to_2d(self, pos, P):
        ret = np.dot(P,pos)
        return ret/ret[2]
    
    def project_to(self, loc, depth, K0, R0, T0, P1):
        point_3d = self.project_to_3d(loc, depth, K0, R0, T0)
        point_reprojected = self.project_to_2d(point_3d, P1)
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

    def run_iteration(self, direction, iteration):
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
                # project into other image 1 with all depth hypotheses
                neighbor_pos = self.project_to(np.array([j,i,1]), neighbor, self.ref_k, self.ref_r, self.ref_t, self.other_pmat_1)
                rand_pos = self.project_to(np.array([j,i,1]), rand_val, self.ref_k, self.ref_r, self.ref_t, self.other_pmat_1)
                original_pos = self.project_to(np.array([j,i,1]), self.depthmap[i,j], self.ref_k, self.ref_r, self.ref_t, self.other_pmat_1)
                # print("from:", np.array([j,i]))
                # print("to:", neighbor_pos)

                # Step4
                best_depth = self.eval_photoconsistency((i,j), self.other_image_1, [neighbor_pos, rand_pos, original_pos])

                # Step5
                if best_depth == 0:
                    self.depthmap[i,j] = neighbor
                elif best_depth == 1:
                    self.depthmap[i,j] = rand_val
                # else leave depth as it is
        np.save("depthmaps/depth_{}.npy".format(iteration), self.depthmap)

    def run(self, num_iterations):
        for it in range(num_iterations):
            self.run_iteration(self.state_table[it%len(self.state_table)], it)

        
        
    
if __name__ == "__main__":
    # read dataset
    data_path = "data/"
    mvs_dataset = Dataset(data_path)

    f_images = mvs_dataset.get_images('fountain')
    f_intrinsics = mvs_dataset.get_intrinsics('fountain')
    f_rotations = mvs_dataset.get_rotation('fountain')
    f_translations = mvs_dataset.get_translation('fountain')
    f_p_mats = mvs_dataset.compute_p_matrices('fountain', f_rotations, f_translations, f_intrinsics)

    pm_fountain = PatchMatch(f_images, f_p_mats, f_intrinsics, f_rotations, f_translations)
    pm_fountain.run(1)
