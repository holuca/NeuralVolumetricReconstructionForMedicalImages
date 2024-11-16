
import pickle
import numpy as np

# Load the pickle file
with open("./data/chest_50.pickle", "rb") as f:
    data = pickle.load(f)
print(data)
# Access the projections_chip data from the loaded dictionary
projections_chip = data['train']['projections']  # Adjust if it's under a different key



non_zero_indices = np.argwhere(projections_chip > 0)

# Calculate the bounding box
min_coords = non_zero_indices.min(axis=0)  # Minimum coordinates (x_min, y_min, z_min)
max_coords = non_zero_indices.max(axis=0)  # Maximum coordinates (x_max, y_max, z_max)
center_coords = (min_coords + max_coords) // 2  # Center of the object in (x, y, z)

print("Bounding Box Min:", min_coords)
print("Bounding Box Max:", max_coords)
print("Center Coordinates:", center_coords)
print("Shape of projections: ", projections_chip.shape)

# Find the 10 highest values
top_10_values = np.sort(projections_chip, axis=None)[-10:][::-1]

print("Top 10 highest values in CHEST:")
print(top_10_values)
print("############")

from scipy.ndimage import center_of_mass

# Load the .npy files
file1 = projections_chip
file2 = np.load("./data_npy/lamino_chip.npy")

# Compute center of mass of non-zero values
center1 = center_of_mass(file1)
center2 = center_of_mass(file2)

print("Center of mass for FOOT:", center1)
print("Center of mass for lami:", center2)

import numpy as np

# Calculate the bounding box for the non-zero elements
def calculate_off_origin(image_data, nVoxel):
    # Find non-zero positions
    non_zero_positions = np.array(np.nonzero(image_data))
    min_coords = non_zero_positions.min(axis=1)
    max_coords = non_zero_positions.max(axis=1)
    
    # Center of the bounding box
    bbox_center = (min_coords + max_coords) / 2
    
    # Calculate `offOrigin` to place the bounding box center in the middle of the voxel grid
    nVoxel_center = np.array(nVoxel) / 2
    offOrigin = nVoxel_center - bbox_center
    return offOrigin

# Example usage with the image data from your pickle file
image_data = np.zeros((175, 128, 128), dtype=np.float32)  # Replace with actual image data if needed
offOrigin_calculated = calculate_off_origin(projections_chip, [128, 128, 128])

print("Calculated offOrigin:", offOrigin_calculated)

#print(" @@@@@@@@@@@@@@@@@@")
#
#with open('./data/created_pickle_file.pickle', 'rb') as file:
#    data = pickle.load(file)
#    print("Projections example:", data['train']['projections'][50][50][50])  # Print a sample projection
##print(data)
#print(data["train"]["projections"].shape)
#print("image shape lami-chip", data["image"].shape)
#print("image shape chest", data["image"][50][50][50])

#import pickle
#import numpy as np
#
## Step 1: Load the existing pickle file
#with open('./data/chest_50.pickle', 'rb') as f:
#    data = pickle.load(f)
#
## Step 2: Load the new .npy arraySW
#new_val = data["train"]["projections"]
## Step 3: Update the training data (projections and adjust angles if needed)
#data['val']["projections"] = new_val
#
## Step 3.1: If the angles need adjustment for 360 degrees (depending on your case)
#
## Step 4: Save the updated data back to the pickle file (or create a new one)
#with open('chest_valEqualsTrain.pickle', 'wb') as f:
#    pickle.dump(data, f)
#
#print("Pickle file updated with new projections.")