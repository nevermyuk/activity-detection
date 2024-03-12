import numpy as np

original = np.load(
    "../data/TSU/TSU_RGB_i3d_feat/RGB_i3d_16frames_64000_SSD/P02T01C06.npy"
)
original_num_feat = original.shape[0]

vashin = np.load("../data/TSU/TSU_Video_features/rgb/P02T01C06.npy")
vashin_num_feat = vashin.shape[0]
print(
    f" Original number of features:{original_num_feat} V-ashin number of features:{vashin_num_feat}"
)
