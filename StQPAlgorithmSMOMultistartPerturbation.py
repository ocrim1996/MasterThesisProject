import copy

import numpy as np
import TwoDimensionProblemStQP as tdp
import perturbation as pb

class SMOAlgorithm:
    def __init__(self, n, Q, c):
        self.vectorX = np.full(n, (1.0 / n), dtype=float)
        # self.vectorX = np.zeros(n, dtype=float)
        # self.vectorX[0] = 1.0
        assert (np.all(self.vectorX >= 0))
        assert (np.all(self.vectorX <= 1))
        #self.vectorG = np.dot(Q, self.vectorX) + c
        self.Q = np.array(Q, dtype=float)
        self.vectorC = c
        self.vectorG = self.gradient_function(self.vectorX)
        self.n = n
        self.vectorA = np.ones(n, dtype=float)
        self.bounds = np.array([0, np.Inf])
        self.mx = 1.
        self.MX = -1.
        self.tau = 1e-8
        self.k = 0
        self.objValue = self.objective_function(self.vectorX)

    # Funzione obiettivo dei problemi StQp.
    def objective_function(self, vectX):
        x = vectX
        Q = self.Q
        c = self.vectorC
        return (np.dot(np.dot(x.T, Q), x)) + np.dot(c.T, x)

    # Funzione grdiente dei problemi StQp.
    def gradient_function(self, vectX):
        x = vectX
        Q = self.Q
        c = self.vectorC
        return np.dot(Q, x) + c

    # Si seleziona la coppia di indici {i, j} che violano maggiormente le condizioni di ottimalità (MVP).
    def select_index(self, vectX):
        x = vectX
        #G = self.vectorG
        G = self.gradient_function(x)
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
    def solve_problem(self, vectX):
        self.vectorG = self.gradient_function(vectX)
        self.mx = 1.
        self.MX = -1.

        while self.mx - self.MX > self.tau:
            index = self.select_index(vectX)
            gradient = self.gradient_function(vectX)

            if self.mx - self.MX < self.tau:
                break

            QD = self.getQD(index)
            x = np.array([vectX[index[0]], vectX[index[1]]])
            c = np.array([self.vectorC[index[0]], self.vectorC[index[1]]])
            G = np.array([gradient[index[0]], gradient[index[1]]])

            # Calcola problema due variabili
            problem = tdp.TwoDimensionProblem(x, QD, G, self.bounds, c)

            bestX = problem.solver()

            # Aggiorna soluzione corrente
            vectX[index[0]] = bestX[0]
            vectX[index[1]] = bestX[1]

            # Aggiorna vettore gradiente
            deltaX = bestX - x

            gradient += np.dot(self.Q[index[0]], deltaX[0]) + np.dot(self.Q[:, index[1]], deltaX[1])

            self.objValue = self.objective_function(vectX)
        #print("vectX:", vectX)
        return self.objective_function(vectX), vectX

    # Risolve i vari passi dell'algoritmo SMO Multistart applicato a problemi StQP.
    def solve_problem_multistart_ones(self):
        solSmo, _ = self.solve_problem(self.vectorX)
        # bestX = self.vectorX

        # Genera tutte le possibili combinazioni di punti iniziali (1,0, . . .,0), (0,1,0, . . .,0), ..., (0,0, . . .,1)
        for i in range(0, self.n):
            self.vectorX = np.zeros(self.n, dtype=float)
            self.vectorX[i] = 1.0
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            tmp, _ = self.solve_problem(self.vectorX)
            if tmp < solSmo:
                solSmo = tmp
                # bestX = self.vectorX

        return solSmo

    # Risolve i vari passi dell'algoritmo SMO Multistart applicato a problemi StQP.
    def solve_problem_multistart_random_points(self, n_max):
        N_max = n_max
        f_star = np.Inf
        N_lists = []

        N = 0
        while N <= N_max:
            self.vectorX = np.random.random_sample(self.n)
            self.vectorX = self.vectorX / np.sum(self.vectorX)
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            tmp, _ = self.solve_problem(self.vectorX)

            if tmp < f_star:
                f_star = tmp
                N_lists.append(N)
                N = 0
            else:
                N = N + 1
                if N == N_max:
                    N_lists.append(N)

        return f_star, N_lists

    # Risolve i vari passi dell'algoritmo SMO Multistart con Perturbazione (ILS) applicato a problemi StQP.
    def solve_problem_multistart_random_points_perturbation(self, n_max, m_max, eps):
        N_max = n_max
        M_max = m_max
        epsilon = eps

        f_star = np.Inf
        N = 0
        M = 0

        while N < N_max:
            self.vectorX = np.random.random_sample(self.n)
            self.vectorX = self.vectorX / np.sum(self.vectorX)
            self.vectorG = np.dot(self.Q, self.vectorX) + self.vectorC
            self.mx = 1.
            self.MX = -1.

            solSmo, vectX = self.solve_problem(self.vectorX)

            while M < M_max:
                vectZ = pb.perturbation2(vectX, epsilon, self.n)
                tmp, vectZ = self.solve_problem(vectZ)
                if tmp < solSmo:
                    solSmo = tmp
                    vectX = vectZ
                    M = 0
                else:
                    M = M + 1

            if solSmo < f_star:
                f_star = solSmo
                N = 0
            else:
                N = N + 1

        return f_star

    # Multistart con Perturbazioni
    def solve_problem_multistart_perturbation(self):
        # N: numero punti di partenza, M: numero massimo di perturbazioni prime di fare restart
        N = 10
        M = 10
        counts = []
        solSmo = None
        solutions_per_iter = []

        for i in range(N):
            print(str(i) + " " +"*"*100)
            self.vectorX = np.random.random_sample(self.n)
            self.vectorX = self.vectorX / np.sum(self.vectorX)
            print("vectorX:", self.vectorX)

            solSmo, _ = self.solve_problem(self.vectorX)
            print("solSmo:", solSmo)
            count = 0

            while count <= M:
                vectX = copy.deepcopy(self.vectorX)
                vectorZ = pb.perturbation1(vectX)
                print("vectorZ:", vectorZ)
                tmp, _ = self.solve_problem(vectorZ)
                print("tmp:", tmp)

                if tmp < solSmo:
                    print("Siamo dentro l'if")
                    solSmo = tmp
                    counts.append(count)
                    count = 0
                else:
                    count = count + 1
                    if count == M:
                        counts.append(count)
            solutions_per_iter.append(solSmo)

        solSmo = min(solutions_per_iter)
        return solSmo, counts