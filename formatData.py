
import numpy as np
import pickle

import torch


max_chest = 0.06712057  # Maximum value in CHEST data

lung_projections = np.load("./data_npy/lung_projections.npy")

ground_truth_2 = np.load("./data_npy/ground_truth_2.npy")
projections_real_2 = np.load("./data_npy/projections_real_2.npy")
print(projections_real_2.shape)
projections_real_50 = np.load("./data_npy/projections_real_50.npy")
angles_50 = [0. , 0.06283185, 0.12566371, 0.18849556, 0.25132741,
0.31415927, 0.37699112, 0.43982297, 0.50265482, 0.56548668,
0.62831853, 0.69115038, 0.75398224, 0.81681409, 0.87964594,
0.9424778 , 1.00530965, 1.0681415 , 1.13097336, 1.19380521,
1.25663706, 1.31946891, 1.38230077, 1.44513262, 1.50796447,
1.57079633, 1.63362818, 1.69646003, 1.75929189, 1.82212374,
1.88495559, 1.94778745, 2.0106193 , 2.07345115, 2.136283 ,
2.19911486, 2.26194671, 2.32477856, 2.38761042, 2.45044227,
2.51327412, 2.57610598, 2.63893783, 2.70176968, 2.76460154,
2.82743339, 2.89026524, 2.95309709, 3.01592895, 3.0787608 ]


data = {
    'numTrain': 180,
    'numVal': 180,
    'DSD': 1500.0,  # Distance Source to Detector
    'DSO': 1000.0,  # Distance Source to Object
    'nDetector': [128, 128],  # Number of detector elements (will be transposed later)
    'dDetector': [1.0, 1.0],  # Size of detector elements
    'nVoxel': [128, 128, 128],  # Number of voxels
    'dVoxel': [1, 1, 1],  # Voxel size
    'offOrigin': [0, 0, 0],  # Offset of the origin
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
        #'angles': angles_radians, 
        'angles': angles_rad_fromBeginning,
        'projections': projections_real_2,  # projections from npy file
    },
    'val': {
        'angles': angles_rad_fromBeginning,
        'projections': projections_real_2  # projections from npy file
    }
}


with open('./data/tomography.pickle', 'wb') as f:
    pickle.dump(data, f)