import numpy as np
import pickle
import torch


ground_truth_2 = np.load("./data_npy/ground_truth_2.npy")

laminography_360 = np.load("./data_npy/projections_lamino.npy")
laminography_180 = laminography_360[::2]
laminography_90 = laminography_180[::2]
laminography_45 = laminography_90[::2]
#laminography_projections_torch = torch.from_numpy(laminography_360)
#laminography_projections_torch = torch.flip(laminography_projections_torch, dims=(0,2))
#laminography_360 = laminography_projections_torch.numpy()

print(laminography_360.shape)

data = {
    'numTrain': 360,
    'numVal': 360,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [128, 175],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [128, 128, 128],  # Number of voxels
    'dVoxel': [1, 1, 1],  # Voxel size
    'offOrigin': [-128, -128, -128],  # Offset of the origin
    'offDetector': [0, 0],  # Offset of the detector
    'accuracy': 0.5,
    'mode': 'parallel',  # Scan mode
    'filter': None,  # Some filter applied to the data
    'totalAngle': 360.0,  # Total scan angle in degrees
    'startAngle': 0.0,  # Start angle of the scan
    'randomAngle': False,  # If the scan angles are randomized
    'convert': False,  # Conversion flag
    'rescale_slope': 1.0,  # Rescale slope
    'rescale_intercept': 0.0,  # Rescale intercept
    'normalize': True,  # If the data is normalized
    'noise': 0,  # Noise level
    'tilt_angle': 29,
    'image': ground_truth_2,  # Placeholder 3D image
    'train': {
        'angles': np.linspace(0, 2 * np.pi, 360, endpoint=True),  # 360 projections equally spaced
        #'angles': np.linspace(0, 180, 180, endpoint=True),  # 360 projections equally spaced
        'projections': laminography_360,  # projections from npy file
    },
    'val': {
        'angles': np.linspace(0, 2 * np.pi, 360, endpoint=True),  # 360 projections equally spaced
        'projections': laminography_360,  # projections from npy file
    }
}


with open('./data/laminography.pickle', 'wb') as f:
    pickle.dump(data, f)