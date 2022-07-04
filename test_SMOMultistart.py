import numpy as np
import timeit as tm
import StQPAlgorithmSMOMultistart as stqpSMOMultistart

prob_file_name = "Problems/Dataset_Generic/Problem_100x100(0.5)_1.txt"

Q = np.loadtxt(prob_file_name)
n = Q.shape[0]
c = np.zeros(n, dtype=float)

problemSMO = stqpSMOMultistart.SMOAlgorithm(n, Q, c)

start_time = tm.default_timer()
solution = problemSMO.solve_problem_multistart_ones()
stop_time = tm.default_timer()

print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
print("Objective function minimum:", solution)
print("Minimization time:", stop_time - start_time)