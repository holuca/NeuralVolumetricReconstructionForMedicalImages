import numpy as np
import pickle

tomography_projections = np.load("./data_npy/projections.npy")
tomography_projections_gt = np.load("./data_npy/ground_truth.npy")

laminography_projections = np.load("./data_npy/lamino_chip.npy")
laminography_projections_gt = np.load("./data_npy/projections_laminography.npy")




laminography_projections = np.rot90(laminography_projections, k=3, axes=(1, 2))



print(laminography_projections.shape)
print(laminography_projections_gt.shape)
print(tomography_projections.shape)
print(tomography_projections_gt.shape)

# Transpose for a side view (swap axes 0 and 1)
# Calculate max values for normalization
max_tomo = tomography_projections.max()
max_tomo_gt = tomography_projections_gt.max()
max_lami = laminography_projections.max()
max_lami_gt = laminography_projections_gt.max()
max_chest = 0.06712057  # Maximum value in CHEST data


print(max_tomo)
print(max_tomo_gt)
print(max_lami)
print(max_lami_gt)
# Normalize lamino to the CHEST range
tomo_normalized = (tomography_projections / max_tomo) * max_chest
tomo_gt_normalized = (tomography_projections_gt / max_tomo_gt) * max_chest

lami_normalized = (laminography_projections / max_lami) * max_chest
lami_gt_normalized = (laminography_projections_gt / max_lami_gt) * max_chest

print(tomo_normalized.max())
print(tomo_gt_normalized.max())
print(lami_normalized.max())
print(lami_gt_normalized.max())

data = {
    'numTrain': 50,
    'numVal': 50,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [128, 175],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [175, 128, 128],  # Number of voxels
    'dVoxel': [1, 1, 1],  # Voxel size
    'offOrigin': [-175, -128, -128],  # Offset of the origin
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
    'image': np.zeros((175, 128, 128), dtype=np.float32),  # Placeholder 3D image
    'train': {
        'angles': np.linspace(0, 2 * np.pi, 360, endpoint=True),  # 360 projections equally spaced
        #'angles': np.linspace(0, 180, 180, endpoint=True),  # 360 projections equally spaced
        'projections': lami_normalized,  # projections from npy file
    },
    'val': {
        #'angles': np.linspace(0, 2 * np.pi, 180, endpoint=False),  # 360 projections equally spaced
        'angles': np.linspace(0, 2 * np.pi, 180, endpoint=True),  # 360 projections equally spaced
        'projections': lami_gt_normalized,  # projections from npy file
    }
}


with open('./data/laminography.pickle', 'wb') as f:
    pickle.dump(data, f)