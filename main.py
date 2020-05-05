from patchmatch_stereo.dataset import Dataset
from patchmatch_stereo.patchmatch import PatchMatch

if __name__ == "__main__":
    data_path = "data/"
    mvs_dataset = Dataset(data_path)

    f_images = mvs_dataset.get_images('fountain')
    f_p_mats = mvs_dataset.get_p_matrices('fountain')
    f_intrinsics = mvs_dataset.get_intrinsics('fountain')

    pm_fountain = PatchMatch(f_images, f_p_mats, f_intrinsics)

    # pm_fountain.run_iteration()