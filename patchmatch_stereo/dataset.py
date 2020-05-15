import os
import pandas as pd
import numpy as np
import cv2

class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scenes = os.listdir(self.data_path)
        self.img_scale_factor = 1
        print("Detected the following scenes: ", self.scenes)
        # configurations
        self.fountain_indices = [5, 4, 6]
        self.hers_indices = [6, 7, 5]
        self.entry_indices = [4, 3, 5]
        
    def get_config(self, scene):
        if 'fountain' == scene:
            return self.fountain_indices
        elif 'entry' == scene:
            return self.entry_indices
        elif 'hers-jezu' == scene:
            return self.hers_indices

    def get_images(self, scene):
        images_df = pd.DataFrame(columns=['name', 'image'])
        for image_index in self.get_config(scene):
            image_name = "000{}.png".format(image_index)
            images_df = images_df.append({'name': image_name,
                                        'image': cv2.resize(cv2.imread(os.path.join(self.data_path, scene, 'images', image_name)), None,fx=self.img_scale_factor,fy=self.img_scale_factor)}, ignore_index=True)
        return images_df
    
    def get_intrinsics(self, scene):
        intrinsics_df = pd.DataFrame(columns=['name', 'intrinsics'])
        for index in self.get_config(scene):
            filename = "000{}.png.camera".format(index)
            with open(os.path.join(self.data_path, scene, 'cameras', filename), "r") as f:
                contents = f.read()
                elements = contents.split()
                pmat = np.array([float(x) for x in elements[0:9]]).reshape((3,3))
                pmat[0:2] = np.multiply(pmat[0:2], self.img_scale_factor)
                intrinsics_df = intrinsics_df.append({'name': filename[:-7],
                                                'intrinsics': pmat}, ignore_index=True)
                f.close()
        return intrinsics_df

    def get_rotation(self, scene):
        rotation_df = pd.DataFrame(columns=['name', 'rotation'])
        for index in self.get_config(scene):
            filename = "000{}.png.camera".format(index)
            with open(os.path.join(self.data_path, scene, 'cameras', filename), "r") as f:
                contents = f.read()
                elements = contents.split()
                rotation_df = rotation_df.append({'name': filename[:-7],
                                                'rotation': np.array([float(x) for x in elements[12:21]]).reshape((3,3))}, ignore_index=True)
                f.close()
        return rotation_df


    def get_translation(self, scene):
        translation_df = pd.DataFrame(columns=['name', 'translation'])
        for index in self.get_config(scene):
            filename = "000{}.png.camera".format(index)
            with open(os.path.join(self.data_path, scene, 'cameras', filename), "r") as f:
                contents = f.read()
                elements = contents.split()
                translation_df = translation_df.append({'name': filename[:-7],
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
            resized_intrinsics = np.multiply(intrinsics_df['intrinsics'][i][0:2], 1.0)
            resized_intrinsics = np.concatenate((resized_intrinsics, [intrinsics_df['intrinsics'][i][2]]))
            P = np.dot(intrinsics_df['intrinsics'][i], mat3)
            pmat_df = pmat_df.append({'name': rotation_df['name'][i], 
                                    'pmat': P}, ignore_index=True)
        return pmat_df


if __name__ == "__main__":
    data_path = "data/"
    mvs_dataset = Dataset(data_path)

    intrinsics = mvs_dataset.get_intrinsics('entry')
    print(intrinsics['name'][0], intrinsics['intrinsics'][0].shape)

    rotation = mvs_dataset.get_rotation('fountain')
    print(rotation['name'][0], rotation['rotation'][0])
    translation = mvs_dataset.get_translation('fountain')
    print(translation['name'][0], translation['translation'][0])

    pmat = mvs_dataset.compute_p_matrices('hers-jezu', rotation, translation, intrinsics)
    print(pmat['name'][0], pmat['pmat'][0])

    entry_images = mvs_dataset.get_images('entry')