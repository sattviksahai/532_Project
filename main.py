from patchmatch_stereo.dataset import Dataset
from patchmatch_stereo.patchmatch import PatchMatch

if __name__ == "__main__":
    # read dataset
    data_path = "data/"
    mvs_dataset = Dataset(data_path)
    
    scene = 'hers-jezu'
    f_images = mvs_dataset.get_images(scene)
    f_intrinsics = mvs_dataset.get_intrinsics(scene)
    f_rotations = mvs_dataset.get_rotation(scene)
    f_translations = mvs_dataset.get_translation(scene)
    f_p_mats = mvs_dataset.compute_p_matrices(scene, f_rotations, f_translations, f_intrinsics)

    pm_fountain = PatchMatch(f_images, f_p_mats, f_intrinsics, f_rotations, f_translations)
    pm_fountain.run(10)