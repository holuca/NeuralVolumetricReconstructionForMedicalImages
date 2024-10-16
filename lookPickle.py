import pickle

with open('./data/chest_50.pickle', 'rb') as file:
    data = pickle.load(file)

print(data)