import os
import pandas as pd
import numpy as np
import cv2

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scenes = os.listdir(self.data_path)
        self.img_scale_factor = 1.0
        print("Detected the following scenes: ", self.scenes)
        print("TODO: remove image resizing from Dataset")

    def get_images(self, scene):
        images_df = pd.DataFrame(columns=['name', 'image'])
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'images')):
            images_df = images_df.append({'name': image_name,
                                        'image': cv2.resize(cv2.imread(os.path.join(self.data_path, scene, 'images', image_name)), None,fx=self.img_scale_factor,fy=self.img_scale_factor)}, ignore_index=True)
        return images_df
    
    def get_p_matrices(self, scene):
        pmat_df = pd.DataFrame(columns=['name', 'pmat'])
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'p-matrices')):
            f=open(os.path.join(self.data_path, scene, 'p-matrices', image_name), "r")
            contents =f.read()
            pmat_df = pmat_df.append({'name': image_name[:-2],
                                    'pmat': np.array([float(x) for x in contents.split()]).reshape((3,4))}, ignore_index=True)
            f.close()
        return pmat_df
    
    def get_intrinsics(self, scene):
        intrinsics_df = pd.DataFrame(columns=['name', 'intrinsics']) # should i be keeping the intrinsics separate? 
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'cameras')):
            with open(os.path.join(self.data_path, scene, 'cameras', image_name), "r") as f:
                contents = f.read()
                elements = contents.split()
                intrinsics_df = intrinsics_df.append({'name': image_name[:-7],
                                                'intrinsics': np.array([float(x) for x in elements[0:9]]).reshape((3,3))}, ignore_index=True)
                f.close()
        return intrinsics_df

    def get_rotation(self, scene):
        rotation_df = pd.DataFrame(columns=['name', 'rotation'])
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'cameras')):
            with open(os.path.join(self.data_path, scene, 'cameras', image_name), "r") as f:
                contents = f.read()
                elements = contents.split()
                rotation_df = rotation_df.append({'name': image_name[:-7],
                                                'rotation': np.array([float(x) for x in elements[12:21]]).reshape((3,3))}, ignore_index=True)
                f.close()
        return rotation_df


    def get_translation(self, scene):
        translation_df = pd.DataFrame(columns=['name', 'translation'])
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'cameras')):
            with open(os.path.join(self.data_path, scene, 'cameras', image_name), "r") as f:
                contents = f.read()
                elements = contents.split()
                translation_df = translation_df.append({'name': image_name[:-7],
                                                'translation': np.array([float(x) for x in elements[21:24]]).reshape((3,1))}, ignore_index=True)
                f.close()
        return translation_df

    def compute_p_matrices(self, scene, rotation_df, translation_df, intrinsics_df):
        pmat_df = pd.DataFrame(columns=['name', 'pmat'])
        
        assert(len(intrinsics_df.index) == len(rotation_df.index) == len(translation_df.index))

        for i in range(len(rotation_df.index)):
            mat1 = np.transpose(rotation_df['rotation'][i])
            mat2 = np.dot(-np.transpose(rotation_df['rotation'][i]), translation_df['translation'][i])
            mat3 = np.concatenate((mat1, mat2), axis=1)
            resized_intrinsics = np.multiply(intrinsics_df['intrinsics'][i][0:2], self.img_scale_factor)
            resized_intrinsics = np.concatenate((resized_intrinsics, [intrinsics_df['intrinsics'][i][2]]))
            P = np.dot(resized_intrinsics, mat3)
            pmat_df = pmat_df.append({'name': rotation_df['name'][i], 
                                    'pmat': P}, ignore_index=True)
        return pmat_df

# modify get_p_matrices to compute from instrinsics and extrinsics



if __name__ == "__main__":
    data_path = "data/"
    mvs_dataset = Dataset(data_path)

    intrinsics = mvs_dataset.get_intrinsics('fountain')

    rotation = mvs_dataset.get_rotation('fountain')
    translation = mvs_dataset.get_translation('fountain')

    pmat = mvs_dataset.compute_p_matrices('fountain', rotation, translation, intrinsics)
    print(pmat)

    entry_images = mvs_dataset.get_images('entry')
    p_mats = mvs_dataset.get_p_matrices('fountain')
    print(p_mats)

