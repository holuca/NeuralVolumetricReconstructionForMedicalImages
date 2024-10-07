import pickle

# Open the pickle file in read-binary mode
with open('./data/chest_50.pickle', 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the deserialized Python object
print(data)