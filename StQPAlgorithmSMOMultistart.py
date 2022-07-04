import numpy as np
import TwoDimensionProblemStQP as tdp
import perturbation as pb
import copy

class SMOAlgorithm:
    def __init__(self, n, Q, c):
        self.vectorX = np.full(n, (1.0 / n), dtype=float)
        # self.vectorX = np.zeros(n, dtype=float)
        # self.vectorX[0] = 1.0
        assert (np.all(self.vectorX >= 0))
        assert (np.all(self.vectorX <= 1))
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
        return (np.dot(np.dot(x.T, Q), x)) + np.dot(c.T, x)

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

            if self.mx - self.MX < self.tau:
                break

            QD = self.getQD(index)
            x = np.array([self.vectorX[index[0]], self.vectorX[index[1]]])
            c = np.array([self.vectorC[index[0]], self.vectorC[index[1]]])
            G = np.array([self.vectorG[index[0]], self.vectorG[index[1]]])

            # Calcola problema due variabili
            problem = tdp.TwoDimensionProblem(x, QD, G, self.bounds, c)

            bestX = problem.solver()

            # Aggiorna soluzione corrente
            self.vectorX[index[0]] = bestX[0]
            self.vectorX[index[1]] = bestX[1]

            # Aggiorna vettore gradiente
            deltaX = bestX - x

            self.vectorG += np.dot(self.Q[index[0]], deltaX[0]) + np.dot(self.Q[:, index[1]], deltaX[1])

            self.objValue = self.objective_function()

        return self.objective_function()

    # Risolve i vari passi dell'algoritmo SMO Multistart applicato a problemi StQP.
    def solve_problem_multistart_ones(self):
        solSmo = self.solve_problem()
        # bestX = self.vectorX

        # Genera tutte le possibili combinazioni di punti iniziali (1,0, . . .,0), (0,1,0, . . .,0), ..., (0,0, . . .,1)
        for i in range(0, self.n):
            self.vectorX = np.zeros(self.n, dtype=float)
            self.vectorX[i] = 1.0
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            tmp = self.solve_problem()
            if tmp < solSmo:
                solSmo = tmp
                # bestX = self.vectorX

        return solSmo

    # Risolve i vari passi dell'algoritmo SMO Multistart applicato a problemi StQP.
    def solve_problem_multistart_random_points(self):
        solSmo = self.solve_problem()
        # bestX = self.vectorX

        # Genera ultieriori punti presi casualmente e normalizzati in maniera tale che la somma delle componenti dia 1
        for i in range(0, 1000):
            self.vectorX = np.random.random_sample(self.n)
            self.vectorX = self.vectorX / np.sum(self.vectorX)
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            tmp = self.solve_problem()
            if tmp < solSmo:
                solSmo = tmp
                # bestX = problemSMO.vectorX

        return solSmo

    # Risolve i vari passi dell'algoritmo SMO Multistart applicato a problemi StQP.
    def solve_problem_multistart_mix(self):
        solSmo = self.solve_problem()
        # bestX = self.vectorX

        # Genera tutte le possibili combinazioni di punti iniziali (1,0, . . .,0), (0,1,0, . . .,0), ..., (0,0, . . .,1)
        for i in range(0, self.n):
            self.vectorX = np.zeros(self.n, dtype=float)
            self.vectorX[i] = 1.0
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            tmp = self.solve_problem()
            if tmp < solSmo:
                solSmo = tmp
                # bestX = self.vectorX

        # Genera ultieriori punti presi casualmente e normalizzati in maniera tale che la somma delle componenti dia 1
        for i in range(0, 1000):
            self.vectorX = np.random.random_sample(self.n)
            self.vectorX = self.vectorX / np.sum(self.vectorX)
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            tmp = self.solve_problem()
            if tmp < solSmo:
                solSmo = tmp
                # bestX = problemSMO.vectorX

        return solSmo

    # Multistart con Perturbazioni
    def solve_problem_multistart_perturbation(self):
        # N: numero punti di partenza, M: numero massimo di perturbazioni prime di fare restart
        N = 100
        M = 5
        counts = []
        solSmo = None

        for i in range(N):
            print("*"*100)
            self.vectorX = np.random.random_sample(self.n)
            self.vectorX = self.vectorX / np.sum(self.vectorX)
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            solSmo = self.solve_problem()
            print("solSmo:", solSmo)
            count = 0

            while count < M:
                self.vectorX = pb.perturbation1(self.vectorX)
                tmp = self.solve_problem()
                print("tmp:", tmp)
                if tmp < solSmo:
                    print("aaaa")
                    solSmo = tmp
                    counts.append(count)
                    count = 0
                else:
                    count = count + 1

        return solSmo, counts
