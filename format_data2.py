from itertools import product
import numpy as np
import pickle
import os
import torch

tomography_projections_gt = np.load("./data_npy/ground_truth.npy")

laminography_projections = np.load("./data_npy/lamino_chip.npy")
laminography_projections_180 = np.load("./data_npy/projections_laminography.npy")

#Correct: k3_02 (for visualization)
tomography_projections_gt = np.rot90(tomography_projections_gt, k=3, axes=(0, 2))

#TESTing

#laminography_projections = np.rot90(laminography_projections, k=2, axes=(1, 2))
#laminography_projections = torch.tensor(laminography_projections)
#laminography_projections = torch.flip(laminography_projections, dims=(1,2))

#CORRECT: k2_02
laminography_projections_180 = np.rot90(laminography_projections_180, k=2, axes=(1, 2))


# Transpose for a side view (swap axes 0 and 1)
# Calculate max values for normalization
max_lami = laminography_projections.max()
max_lami_gt = laminography_projections_180.max()
max_chest = 0.06712057  # Maximum value in CHEST data


lami_normalized = (laminography_projections / max_lami) * max_chest
lami_180_normalized = (laminography_projections_180/ max_lami_gt) * max_chest

# Define output directory for pickle files
output_directory = './picklefiles'
os.makedirs(output_directory, exist_ok=True)

# Define possible values for offOrigin
possible_values = [-64, 0, 64]

# Generate and save a pickle file for each combination of offOrigin
for offOrigin_values in product(possible_values, repeat=3):
    # Set current offOrigin in data
    data = {
        'numTrain': 50,
        'numVal': 50,
        'DSD': 1500.0,
        'DSO': 1000.0,
        'nDetector': [128, 128],
        'dDetector': [1.0, 1.0],
        'nVoxel': [128, 128, 128],
        'dVoxel': [1, 1, 1],
        'offOrigin': list(offOrigin_values),  # Set the current offset combination
        'offDetector': [0, 0],
        'accuracy': 0.5,
        'mode': 'parallel',
        'filter': None,
        'totalAngle': 180,
        'startAngle': 0,
        'randomAngle': False,
        'convert': False,
        'rescale_slope': 1.0,
        'rescale_intercept': 0.0,
        'normalize': True,
        'noise': 0,
        'tilt_angle': 0,
        'image': tomography_projections_gt,
        'train': {
            'angles': np.linspace(np.pi/2, 2 * np.pi - np.pi/2, 180, endpoint=True),  # 360 projections equally spaced
            #'angles': np.linspace(0, 180, 180, endpoint=True),  # 360 projections equally spaced
            'projections': lami_180_normalized,  # projections from npy file
    },
        'val': {
            'angles':  np.linspace(np.pi/2, 2 * np.pi - np.pi/2, 360, endpoint=True),  # 360 projections equally spaced
            'projections': lami_normalized
        }
    }
    
    # Create a filename based on the current offOrigin values
    filename = f"tomography_offset_{offOrigin_values[0]}_{offOrigin_values[1]}_{offOrigin_values[2]}.pickle"
    filepath = os.path.join(output_directory, filename)
    
    # Save to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved: {filepath}")
