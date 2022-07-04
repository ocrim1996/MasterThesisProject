import numpy as np
import timeit as tm
import StQPAlgorithmSMO as stqpSMO

prob_file_name = "Problems/Dataset_Generic/Problem_200x200(0.75)_1.txt"

Q = np.loadtxt(prob_file_name)
n = Q.shape[0]
c = np.zeros(n, dtype=float)

# Punto di partenza x0 = (1, 0, . . ., 0)
#x0 = np.zeros(n, dtype=float)
#x0[0] = 1.0

# Punto di partenza x0 = (1/n, . . ., 1/n)
x0 = np.full(n, (1.0/n), dtype=float)

problemSMO = stqpSMO.SMOAlgorithm(n, Q, c, x0)

start_time = tm.default_timer()
solution = problemSMO.solve_problem()
stop_time = tm.default_timer()

print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
print("Objective function minimum:", solution)
print("Minimization time:", stop_time - start_time)
