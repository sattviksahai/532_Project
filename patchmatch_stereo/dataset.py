import os
import pandas as pd
import numpy as np
import cv2

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scenes = os.listdir(self.data_path)
        print("Detected the following scenes: ", self.scenes)

    def get_images(self, scene):
        images_df = pd.DataFrame(columns=['name', 'image'])
        for image_name in os.listdir(os.path.join(self.data_path, scene, 'images')):
            images_df = images_df.append({'name': image_name,
                                        'image': cv2.imread(os.path.join(self.data_path, scene, 'images', image_name))}, ignore_index=True)
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

if __name__ == "__main__":
    data_path = "data/"
    mvs_dataset = Dataset(data_path)
    entry_images = mvs_dataset.get_images('entry')
    print(entry_images.head())
    p_mats = mvs_dataset.get_p_matrices('fountain')
    print(p_mats.head())