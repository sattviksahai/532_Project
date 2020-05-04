from patchmatch_stereo.dataset import Dataset

if __name__ == "__main__":
    data_path = "data/"
    mvs_dataset = Dataset(data_path)
    entry_images = mvs_dataset.get_images('entry')
    print(entry_images.head())
    p_mats = mvs_dataset.get_p_matrices('fountain')
    print(p_mats.head())