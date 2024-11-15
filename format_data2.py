import numpy as np
import pickle
import os
from itertools import product

# Load data files
tomography_new = np.load("./data_npy/projections_real.npy")
tomography_new = np.transpose(tomography_new, axes=(0, 2, 1))

angles = np.load("./data_npy/angles_real.npy")
angles_radians = np.deg2rad(angles)
angles_radians_inverted = -angles_radians

tomography_projections = np.load("./data_npy/projections.npy")
tomography_projections_gt = np.load("./data_npy/ground_truth.npy")
tomography_projections_gt = np.rot90(tomography_projections_gt, k=3, axes=(0, 2))

# Normalize data
max_tomo_new = tomography_new.max()
max_tomo = tomography_projections.max()
max_tomo_gt = tomography_projections_gt.max()
max_chest = 0.06712057

tomo_normalized = (tomography_projections / max_tomo) * max_chest
tomo_gt_normalized = (tomography_projections_gt / max_tomo_gt) * max_chest
tomo_new_normalized = (tomography_new / max_tomo_new) * max_chest

# Define output directory for pickle files
output_directory = './picklefiles'
os.makedirs(output_directory, exist_ok=True)

# Define possible values for offOrigin
possible_values = [-128, 0, 128]

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
        'totalAngle': np.pi,
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
            'angles': np.linspace(0, np.pi, 180, endpoint=False),  # 360 projections equally spaced
            'projections': tomo_normalized,
        },
        'val': {
            'angles': angles_radians_inverted,
            'projections': tomo_new_normalized
        }
    }
    
    # Create a filename based on the current offOrigin values
    filename = f"tomography_offset_{offOrigin_values[0]}_{offOrigin_values[1]}_{offOrigin_values[2]}.pickle"
    filepath = os.path.join(output_directory, filename)
    
    # Save to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved: {filepath}")
