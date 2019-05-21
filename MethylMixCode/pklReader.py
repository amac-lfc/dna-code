import numpy as np

with open('MatrixX.pkl', 'rb') as file:
    X = np.load('MatrixX.pkl')

print(X)


with open('MatrixY.pkl', 'rb') as file:
    Y = np.load("MatrixY.pkl")

print(Y)
