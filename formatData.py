import numpy as np
import pickle


tomography_new = np.load("./data_npy/projections_real.npy")
angles = np.load("./data_npy/angles_real.npy")
angles_radians = np.deg2rad(angles)
angles_radians_inverted = - angles_radians
tomography_new = np.transpose(tomography_new, axes=(0, 2, 1))

tomography_projections = np.load("./data_npy/projections.npy")

tomography_projections_gt = np.load("./data_npy/ground_truth.npy")
tomography_projections_gt = np.rot90(tomography_projections_gt, k=3, axes=(0, 2))
#tomography_projections_gt = np.rot90(tomography_projections_gt, k=1, axes=(1, 2))
tomography_projections = np.rot90(tomography_projections, k=2, axes=(0, 2))


max_tomo_new = tomography_new.max()

# Calculate max values for normalization
max_tomo = tomography_projections.max()
max_tomo_gt = tomography_projections_gt.max()
max_chest = 0.06712057  # Maximum value in CHEST data


# Normalize lamino to the CHEST range
tomo_normalized = (tomography_projections / max_tomo) * max_chest
tomo_gt_normalized = (tomography_projections_gt / max_tomo_gt) * max_chest
tomo_new_normalized = (tomography_new/max_tomo_new)*max_chest
data = {
    'numTrain': 50,
    'numVal': 50,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [128, 128],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [128, 128, 128],  # Number of voxels
    'dVoxel': [1, 1, 1],  # Voxel size
    'offOrigin': [-128, -128, -128],  # Offset of the origin
    'offDetector': [0, 0],  # Offset of the detectorW
    'accuracy': 0.5,
    'mode': 'parallel',  # Scan mode
    'filter': None,  # Some filter applied to the data
    'totalAngle': np.pi,  # Total scan angle in radian
    'startAngle': 0,  # Start angle of the scan
    'randomAngle': False,  # If the scan angles are randomized
    'convert': False,  # Conversion flag
    'rescale_slope': 1.0,  # Rescale slope
    'rescale_intercept': 0.0,  # Rescale intercept
    'normalize': True,  # If the data is normalized
    'noise': 0,  # Noise level
    'tilt_angle': 0,
    'image': tomography_projections_gt,  # Placeholder 3D image
    'train': {
        'angles': np.linspace(0, np.pi, 180, endpoint=False),  # 360 projections equally spaced
        'projections': tomo_normalized,  # projections from npy file
    },
    'val': {
        #'angles': np.linspace(0, 2 * np.pi, 180, endpoint=False),  # 360 projections equally spaced
        'angles': angles_radians,
        'projections': tomo_new_normalized  # projections from npy file
    }
}


with open('./data/tomography.pickle', 'wb') as f:
    pickle.dump(data, f)