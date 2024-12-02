from itertools import product
import numpy as np
import pickle
import os
import torch

ground_truth_2 = np.load("./data_npy/ground_truth_2.npy")
projections_real_2 = np.load("./data_npy/projections_real_2.npy")
print(ground_truth_2.shape)
print(projections_real_2.shape)
#CORRECT: k2_02

# Define output directory for pickle files
output_directory = './picklefiles'
os.makedirs(output_directory, exist_ok=True)

# Define possible values for offOrigin
possible_values = [-64, 0, 64]

# Generate and save a pickle file for each combination of offOrigin
for offOrigin_values in product(possible_values, repeat=3):
    data = {
        'numTrain': 50,
        'numVal': 50,
        'DSD': 1500.0,  # Distance Source to Detector
        'DSO': 1000.0,  # Distance Source to Object
        'nDetector': [128, 128],  # Number of detector elements (will be transposed later)
        'dDetector': [1.0, 1.0],  # Size of detector elements
        'nVoxel': [128, 128, 128],  # Number of voxels
        'dVoxel': [1, 1, 1],  # Voxel size
        'offOrigin': list(offOrigin_values),
        'offDetector': [0, 0],  # Offset of the detectorW
        'accuracy': 0.5,
        'mode': 'parallel',  # Scan mode
        'filter': None,  # Some filter applied to the data
        'totalAngle': 180,   # Total scan angle 
        'startAngle': 0,  # Start angle of the scan
        'randomAngle': False,  # If the scan angles are randomized
        'convert': False,  # Conversion flag
        'rescale_slope': 1.0,  # Rescale slope
        'rescale_intercept': 0.0,  # Rescale intercept
        'normalize': True,  # If the data is normalized
        'noise': 0,  # Noise level
        'tilt_angle': 0,
        'image': ground_truth_2,  # Placeholder 3D image
        'train': {
            'angles': np.linspace(0,   np.pi , 180, endpoint=False),  # 360 projections equally spaced
            'projections': projections_real_2,  # projections from npy file
        },
        'val': {
            #'angles': np.linspace(0, 2 * np.pi, 180, endpoint=False),  # 360 projections equally spaced
            'angles': np.linspace(0,   np.pi , 180, endpoint=False),  # 360 projections equally spaced
            'projections': projections_real_2,  # projections from npy file
        }
    }
# Create a filename based on the current offOrigin values
    filename = f"tomography_offset_{offOrigin_values[0]}_{offOrigin_values[1]}_{offOrigin_values[2]}.pickle"
    filepath = os.path.join(output_directory, filename)
    
    # Save to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved: {filepath}")




          
