import pickle

from sklearn import metrics

models = []
with open(input('Path to the models\' file : '), "rb") as file:
    models = pickle.load(file)

