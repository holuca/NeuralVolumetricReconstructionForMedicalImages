import pickle
import numpy as np

with open('./data/chest_50.pickle', 'rb') as file:
    data = pickle.load(file)

print(data)
print(data["train"]["projections"].shape)

with open('./data/created_pickle_file.pickle', 'rb') as file:
    data = pickle.load(file)
#print(data)
print(data["train"]["projections"].shape)
print(data["image"].shape)