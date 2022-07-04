import numpy as np
import TwoDimensionProblemStQP as tdp


class SMOAlgorithm:
    def __init__(self, n, Q, c, x0):
        self.vectorX = x0
        assert(np.all(self.vectorX >= 0))
        assert(np.all(self.vectorX <= 1))
        self.vectorG = np.dot(Q, self.vectorX) + c
        self.Q = np.array(Q, dtype=float)
        self.n = n
        self.vectorA = np.ones(n, dtype=float)
        self.bounds = np.array([0, np.Inf])
        self.mx = 1.
        self.MX = -1.
        self.tau = 1e-8
        self.vectorC = c
        self.k = 0
        self.objValue = self.objective_function()

    # Funzione obiettivo dei problemi StQp.
    def objective_function(self):
        x = self.vectorX
        Q = self.Q
        c = self.vectorC
        return (np.dot(np.dot(x, Q), x)) + np.dot(c, x)

    # Si seleziona la coppia di indici {i, j} che violano maggiormente le condizioni di ottimalità (MVP).
    def select_index(self):
        x = self.vectorX
        G = self.vectorG
        index = np.empty(2, dtype=int)
        Gmax = np.NINF
        Gmin = np.Inf

        for i in range(0, self.n):
            if -G[i] >= Gmax:
                Gmax = -G[i]
                index[0] = i

        for j in range(0, self.n):
            if x[j] > 0:
                if -G[j] <= Gmin:
                    Gmin = -G[j]
                    index[1] = j

        self.mx = Gmax
        self.MX = Gmin

        self.k += 1

        return index

    # Serve per restituire una matrice di dimensione 2x2 per il sotto-problema in 2 variabili.
    def getQD(self, index):
        QD = np.empty((2, 2), dtype=float)

        for i in range(0, 2):
            for j in range(0, 2):
                QD[i][j] = self.Q[index[i]][index[j]]

        return QD

    # Risolve i vari passi dell'algoritmo SMO applicato a problemi StQP.
    def solve_problem(self):
        while self.mx - self.MX > self.tau:
            index = self.select_index()
            print("index (Working Set):", index)
            print("vectorX:", self.vectorX)
            print("vectorC:", self.vectorC)
            print("vectorG:", self.vectorG)

            if self.mx - self.MX < self.tau:
                break

            # Si riduce tutti gli array o matrice a due variabili
            QD = self.getQD(index)
            x = np.array([self.vectorX[index[0]], self.vectorX[index[1]]])
            print("x(2D):", x)
            c = np.array([self.vectorC[index[0]], self.vectorC[index[1]]])
            print("c(2D):", c)
            G = np.array([self.vectorG[index[0]], self.vectorG[index[1]]])
            print("G(2D):", G)

            # Calcola problema due variabili
            problem = tdp.TwoDimensionProblem(x, QD, G, self.bounds, c)

            bestX = problem.solver()
            print("bestX:", bestX)

            # Aggiorna soluzione corrente
            self.vectorX[index[0]] = bestX[0]
            self.vectorX[index[1]] = bestX[1]
            print("vectorX Aggiornato:", self.vectorX)

            # Aggiorna vettore gradiente
            deltaX = bestX - x
            self.vectorG += np.dot(self.Q[index[0]], deltaX[0]) + np.dot(self.Q[:, index[1]], deltaX[1])
            print("vectorG Aggiornato:", self.vectorG)
            print()

            self.objValue = self.objective_function()

        return self.objective_function()
