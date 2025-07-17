import tqdm
import numpy as np

import os
import numpy as np

# Path to the main directory
main_dir = "./BABEL-QA/motion_sequences"
data_list = []

# Iterate through all subfolders
for subfolder in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subfolder)
    
    if os.path.isdir(subfolder_path):
        joints_file = os.path.join(subfolder_path, "joints.npy")
        
        if os.path.isfile(joints_file):
            motion_data = np.load(joints_file)

            data = np.load(joints_file)
            if np.isnan(data).any():
                # print(file)
                continue
            if len(data.shape) != 3:
                continue
            data_list.append(data)

data = np.concatenate(data_list, axis=0)
Mean = data.mean(axis=0)
Std = data.std(axis=0)

print(Mean.shape, Std.shape)
np.save("./IPGRM_formatted_data/mean.npy", Mean)
np.save("./IPGRM_formatted_data/std.npy", Std)



