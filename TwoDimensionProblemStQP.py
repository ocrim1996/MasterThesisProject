import numpy as np


class TwoDimensionProblem:
    def __init__(self, x, matrixQ, G, b, c):
        self.matrixQ = matrixQ
        self.vectorC = c
        self.bounds = b
        #self.vectorA = np.ones(2, dtype=float)
        self.bestD = np.array([1, -1], dtype=float)
        self.vectorG = G
        self.bestX = x

    # def objective_function(self, x):
    #     return (np.dot(np.dot(x, self.matrixQ), x)) + np.dot(self.vectorC, x)
    #
    # def constraint(self, x):
    #     return np.dot(self.vectorA.T, x) - 1

    def solver(self):
        G = self.vectorG
        d = self.bestD
        x = self.bestX

        # Per costruzione (?)
        B = x[1]

        # (Metodo del Gradiente) Passo di discesa B, direzione di discesa d
        if np.dot(np.dot(d, self.matrixQ), d) > 0:
            # Calcolo il passo di discesa (passo ottimale)
            nvB = - np.dot(G, d) / (np.dot(np.dot(d, self.matrixQ), d))
            B = np.minimum(B, nvB)

        x = x + np.dot(B, d)

        return x