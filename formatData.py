import numpy as np
import pickle

tomography_projections = np.load("./data_npy/projections.npy")
tomography_projections_gt = np.load("./data_npy/ground_truth.npy")
tomography_projections_gt = np.rot90(tomography_projections_gt, k=3, axes=(0, 2))
#tomography_projections_gt = np.rot90(tomography_projections_gt, k=1, axes=(1, 2))
tomography_projections = np.rot90(tomography_projections, k=2, axes=(0, 2))
#Correct: k3_02 (for visualization)
non_zero_indices = np.argwhere(tomography_projections > 0)

# Calculate the bounding box
min_coords = non_zero_indices.min(axis=0)  # Minimum coordinates (x_min, y_min, z_min)
max_coords = non_zero_indices.max(axis=0)  # Maximum coordinates (x_max, y_max, z_max)
center_coords = (min_coords + max_coords) // 2  # Center of the object in (x, y, z)

print("Bounding Box Min:", min_coords)
print("Bounding Box Max:", max_coords)
print("Center Coordinates:", center_coords)




print(tomography_projections.shape)
print(tomography_projections_gt.shape)



# Transpose for a side view (swap axes 0 and 1)
# Calculate max values for normalization
max_tomo = tomography_projections.max()
max_tomo_gt = tomography_projections_gt.max()
max_chest = 0.06712057  # Maximum value in CHEST data


print(max_tomo)
print(max_tomo_gt)
# Normalize lamino to the CHEST range
tomo_normalized = (tomography_projections / max_tomo) * max_chest
tomo_gt_normalized = (tomography_projections_gt / max_tomo_gt) * max_chest

print(tomo_normalized.max())
print(tomo_gt_normalized.max())

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
    'totalAngle': 180.0,  # Total scan angle in degrees
    'startAngle': 0.0,  # Start angle of the scan
    'randomAngle': False,  # If the scan angles are randomized
    'convert': False,  # Conversion flag
    'rescale_slope': 1.0,  # Rescale slope
    'rescale_intercept': 0.0,  # Rescale intercept
    'normalize': True,  # If the data is normalized
    'noise': 0,  # Noise level
    'tilt_angle': 0,
    'image': tomography_projections_gt,  # Placeholder 3D image
    'train': {
        'angles': np.linspace(0 - np.pi, np.pi - np.pi, 180, endpoint=False),  # 360 projections equally spaced
        #'angles': np.linspace(0, 180, 180, endpoint=True),  # 360 projections equally spaced
        'projections': tomo_normalized,  # projections from npy file
    },
    'val': {
        #'angles': np.linspace(0, 2 * np.pi, 180, endpoint=False),  # 360 projections equally spaced
        'angles': np.linspace(0, np.pi, 180, endpoint=False),  # 360 projections equally spaced
        'projections': tomo_normalized,  # projections from npy file
    }
}


with open('./data/tomography.pickle', 'wb') as f:
    pickle.dump(data, f)