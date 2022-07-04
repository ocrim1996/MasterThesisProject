import random
import numpy as np
import perturbation as pb

vectorX = np.random.random_sample(5)
vectorX = vectorX / np.sum(vectorX)

print("VectorX di partenza:", vectorX)
print("Somma elementi di VectorX:", np.sum(vectorX))

vectorX = pb.perturbation1(vectorX)
print("VectorX aggiornato:", vectorX)
print("Somma elementi di VectorX:", np.sum(vectorX))