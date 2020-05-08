import numpy as np
from patchmatch_stereo.dataset import Dataset

dataset = Dataset("data/")

intrinsics = dataset.get_intrinsics('fountain')
rotation = dataset.get_rotation('fountain')
translation = dataset.get_translation('fountain')

p_mat = dataset.get_p_matrices('fountain')
