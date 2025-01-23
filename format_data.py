import numpy as np
import pickle
import torch


brain_orig = np.load("./data_npy/low_res_projections_brain.npy")
brain_orig = np.rot90(brain_orig, k=1, axes=(1, 2))
brain_orig_phase = np.angle(brain_orig)



# Convert normalized phase t
angles = np.load("./data_npy/low_res_angles_brain.npy")
angles_rad = np.deg2rad(angles)

numAngles = angles.numel()
#H = brain_projections.shape[1]
#W = brain_projections.shape[2]

H = brain_orig_phase.shape[1]
W = brain_orig_phase.shape[2]



data = {
    'numTrain': numAngles,
    'numVal': numAngles,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [W, H],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [W, W, 70],  # Number of voxels
    'dVoxel': [1, 1, 1],  # Voxel size
    'offOrigin': [-W, -W, -70],  # Offset of the origin
    'offDetector': [0, 0],  # Offset of the detector
    'accuracy': 0.5,
    'mode': 'parallel',  # Scan mode
    'filter': None,  # Some filter applied to the data
    'totalAngle': 360,  #
    'startAngle': 0,  # Start angle of the scan
    'randomAngle': False,  # If the scan angles are randomized
    'convert': False,  # Conversion flag
    'rescale_slope': 1.0,  # Rescale slope
    'rescale_intercept': 0.0,  # Rescale intercept
    'normalize': True,  # If the data is normalized
    'noise': 0,  # Noise level
    'tilt_angle': 29,
    'image': np.zeros((W, W, 70)),  # Placeholder 3D image, #np.zeros((70, 356, 356)), # Placeholder 3D image
    'full_proj': brain_orig,
    'train': {
        'angles':angles_rad,  # 
        'projections': brain_orig_phase,  # projections from npy file
    },
    'val': {
        'angles':  angles_rad,  #
        'projections': brain_orig_phase,  # projections from npy file
    }
}


with open('./data/brain.pickle', 'wb') as f:
    pickle.dump(data, f)


print(data['train']['projections'].dtype)  # Should match (num_angles, height, width)