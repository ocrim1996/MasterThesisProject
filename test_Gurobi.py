import numpy as np
import Gurobi as grb

#prob_file_name = "Problems/Dataset_Generic/Problem_50x50(0.75).txt"
prob_file_name = "Problems/Dataset_BSU/BSU_8x8.txt"

Q = np.loadtxt(prob_file_name)
n = Q.shape[0]
c = np.zeros(n, dtype=float)

problemGurobi = grb.StQPSolverEfficientGurobi(Q, c)
x_star, f_star = problemGurobi.solve()

print("PROBLEM:", prob_file_name.split("/")[2][:-4])
print("Objective function minimum:", f_star)
print("Minimization time:", grb.time_to_opt)
print("Total time:", grb.tot_time)

