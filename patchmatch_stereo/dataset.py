import os
import pandas as pd
import numpy as np
import cv2

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scenes = os.listdir(self.data_path)
        self.img_scale_factor = 0.2
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
                intrinsics_df = intrinsics_df.append({'name': image_name[:-7],
                                                'rotation': np.array([float(x) for x in elements[12:21]]).reshape((3,3))}, ignore_index=True)
                f.close()
        return rotation_df


    def get_translation(self, scene):
        translation_df = pd.DataFrame(columns=['name', 'translation'])
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'cameras')):
            with open(os.path.join(self.data_path, scene, 'cameras', image_name), "r") as f:
                contents = f.read()
                elements = contents.split()
                intrinsics_df = intrinsics_df.append({'name': image_name[:-7],
                                                'translation': np.array([float(x) for x in elements[21:24]]).reshape((3,1))}, ignore_index=True)
                f.close()
        return translation_df

# modify get_p_matrices to compute from instrinsics and extrinsics



if __name__ == "__main__":
    data_path = "data/"
    mvs_dataset = Dataset(data_path)

    intrinsics = mvs_dataset.get_intrinsics('fountain')
    print(intrinsics)

    entry_images = mvs_dataset.get_images('entry')
    print(entry_images.head())
    p_mats = mvs_dataset.get_p_matrices('fountain')
    print(p_mats.head())

