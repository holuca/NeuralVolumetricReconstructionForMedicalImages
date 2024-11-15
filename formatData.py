import numpy as np
import pickle

tomography_projections_gt = np.load("./data/ground_truth.npy")

projections_chip = np.load("./data/projections.npy")
projections_chip = np.rot90(projections_chip, k=2, axes=(1,2))

tomography_new = np.load("./data/projections_real.npy")
tomography_new = np.transpose(tomography_new, axes=(0, 2, 1))
tomography_new = np.rot90(tomography_new, k=2, axes=(1,2))
angles = np.load("./data/angles_real.npy")
angles_radians = np.deg2rad(angles)
angles_radians_inverted = - angles_radians

# Calculate max values for normalization
max_lamino = projections_chip.max()

max_tomo_new = tomography_new.max()
max_chest = 0.06712057  # Maximum value in CHEST data

# Normalize lamino to the CHEST range
lamino_normalized = (projections_chip / max_lamino) * max_chest
tomo_new_normalized = (tomography_new/max_tomo_new)*max_chest

'''
# Using NumPy's np.rot90
rotated_np = np.rot90(projections, k=2, axes=(0, 2))

# Using PyTorch's torch.flip
flipped_torch = torch.flip(torch.from_numpy(projections), dims=(0, 2)).numpy()

# Check equivalence
assert np.array_equal(rotated_np, flipped_torch), "The results do not match!"
'''

data = {
    'numTrain': 50,
    'numVal': 50,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [128, 128],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [128, 128, 128],  # Number of voxels
    'dVoxel': [1.0, 1.0, 1.0],  # Voxel size
    'offOrigin': [-128, -128, -128],  # Offset of the origin
    'offDetector': [0, 0],  # Offset of the detector
    'accuracy': 0.5,
    'mode': 'parallel',  # Scan mode
    'filter': None,  # Some filter applied to the data
    'totalAngle': 180.0,  # Total scan angle in degrees
    'startAngle': 0.0,  # Start angle of the scan
    'randomAngle': False,  # If the scan angles are randomized
    'convert': False,  # Conversion flag
    'rescale_slope': 1.0,  # Rescale slope
    'rescale_intercept': 0.0,  # Rescale intercept
    'normalize': True,  # If the data is normalized
    'noise': 0,  # Noise level
    'tilt_angle': 0,
    'image': tomography_projections_gt,
    'train': {
            'angles': np.linspace(0, np.pi, 180, endpoint=False),
            'projections': lamino_normalized    ,
        },
    'val': {
        'angles': np.linspace(0, 2 * np.pi, 360, endpoint=False),  # 360 projections equally spaced
        'projections': lamino_normalized,  # projections from npy file
    }
}


with open('./data/tomo_new.pickle', 'wb') as f:
    pickle.dump(data, f)