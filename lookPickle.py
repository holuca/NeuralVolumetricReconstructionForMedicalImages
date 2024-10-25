
import pickle
import numpy as np

# Load the pickle file
with open("./data/chest_50.pickle", "rb") as f:
    data = pickle.load(f)

# Access the projections_chip data from the loaded dictionary
projections_chip = data['train']['projections']  # Adjust if it's under a different key

# Find the 10 highest values
top_10_values = np.sort(projections_chip, axis=None)[-10:][::-1]

print("Top 10 highest values in CHEST:")
print(top_10_values)


with open('./data/lamino_chip.pickle', 'rb') as file:
    data = pickle.load(file)
    #print("Projections example:", data['train']['projections'][50][50][50])  # Print a sample projection
#print(data)

# Access the projections_chip data from the loaded dictionary
projections_chip = data['train']['projections']  # Adjust if it's under a different key

# Find the 10 highest values
top_10_values = np.sort(projections_chip, axis=None)[-10:][::-1]

print("Top 10 highest values in lamino:")
print(top_10_values)

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