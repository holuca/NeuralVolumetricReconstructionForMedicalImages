import numpy as np
import pickle

projections_chip = np.load("./data/lamino_chip.npy")


# Calculate max values for normalization
max_lamino = projections_chip.max()
max_chest = 0.06712057  # Maximum value in CHEST data

# Normalize lamino to the CHEST range
lamino_normalized = (projections_chip / max_lamino) * max_chest



data = {
    'numTrain': 50,
    'numVal': 50,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [128, 175],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [175, 128, 128],  # Number of voxels
    'dVoxel': [1.0, 1.0, 1.0],  # Voxel size
    'offOrigin': [0, 0, 0],  # Offset of the origin
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
    'tilt_angle': 0,
    'image': np.zeros((175, 128, 128), dtype=np.float32),  # Placeholder 3D image
    'train': {
        'angles': np.linspace(0, 2 * np.pi, 360, endpoint=True),  # 360 projections equally spaced
        #'angles': np.linspace(0, 180, 180, endpoint=True),  # 360 projections equally spaced
        'projections': lamino_normalized,  # projections from npy file
    },
    'val': {
        #'angles': np.linspace(0, 2 * np.pi, 180, endpoint=False),  # 360 projections equally spaced
        'angles': np.linspace(0, 2 * np.pi, 360, endpoint=True),  # 360 projections equally spaced
        'projections': lamino_normalized,  # projections from npy file
    }
}


with open('./data/lamino_chip.pickle', 'wb') as f:
    pickle.dump(data, f)