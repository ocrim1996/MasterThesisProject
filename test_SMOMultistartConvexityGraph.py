import os
import numpy as np
import timeit as tm
import utility_functions as uf
import statistics
#import best_indices as bidx
import StQPAlgorithmSMOMultistartPerturbation as stqpSMOMultistartPerturbation

"""
folder_path = "Problems/Dataset_Generic/"
prob_file_names = ["Problem_500x500(0.25)_1.txt", "Problem_500x500(0.25)_2.txt", "Problem_500x500(0.25)_3.txt",
                   "Problem_500x500(0.5)_1.txt", "Problem_500x500(0.5)_2.txt", "Problem_500x500(0.5)_3.txt",
                   "Problem_1000x1000(0.25).txt"]
"""
"""
for prob_file_name in prob_file_names:
    i_best_prob = bidx.i_bests[prob_file_name]
    prob_file_name = folder_path + prob_file_name
    Q = np.loadtxt(prob_file_name)
    n = Q.shape[0]
    c = np.zeros(n, dtype=float)

    starting_points, mean_zeros, mean_ones = uf.create_convexity_graph_vector_and_bitwise(Q, i_best_prob)

    problemSMO = stqpSMOMultistartPerturbation.SMOAlgorithm(n, Q, c)

    start_time = tm.default_timer()
    solution, nonzero_indices_bestX = problemSMO.solve_problem_multistart_convexity_graph(starting_points)
    #solution, i_best, bestX = problemSMO.solve_problem_multistart_ones()
    #solution, nonzero_indices_bestX= problemSMO.solve_problem_multistart_ones_i_best(i_best_prob)
    stop_time = tm.default_timer()

    print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
    print("Objective function minimum:", solution)
    print("Minimization time:", stop_time - start_time)
    print("NonZero_indices_bestX:", nonzero_indices_bestX)
    #print("i_best_prob:", i_best_prob)
    #print("Average of zeros number:", mean_zeros)
    #print("Average of ones number:", mean_ones)
    #print("i_best:", i_best)
    #print("bestX:", bestX)
"""

folder_path = "Problems/Dataset_BHOSLIB/"
prob_file_names = sorted(os.listdir(folder_path))[:1]

for prob_file_name in prob_file_names:
    prob_file_name = folder_path + prob_file_name
    Q = np.loadtxt(prob_file_name)
    n = Q.shape[0]
    c = np.zeros(n, dtype=float)

    solutions = []
    minimization_times = []
    for iter in range(4000):
        starting_point = uf.create_vector_two_indices(Q)

        problemSMO = stqpSMOMultistartPerturbation.SMOAlgorithm(n, Q, c)

        start_time = tm.default_timer()
        solution = problemSMO.solve_problem_multistart_two_indices(starting_point)
        stop_time = tm.default_timer()

        minimization_time = stop_time - start_time

        solution = uf.truncate(solution, 6)
        solutions.append(solution)
        minimization_times.append(minimization_time)

    min_solution_value = min(solutions)
    min_solution_indices = [i for i, j in enumerate(solutions) if j == min_solution_value]
    num_min = solutions.count(min_solution_value)

    min_minimization_times = [minimization_times[i] for i in min_solution_indices]

    interval_indices = [i for i in range(min_solution_indices[0] + 1)]
    time_to_min = sum(minimization_times[i] for i in interval_indices)
    total_time = sum(minimization_times)

    min_solution_value_final = 1 / (1 + min_solution_value)

    print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
    print("Min Solution Value:", min_solution_value)
    print("Min solution Value Final:", min_solution_value_final)
    print("First Min Solution Index:", min_solution_indices[0])
    print("Num of Min Solution:", num_min)
    # print("Mean Min Minimization Times:", statistics.mean(min_minimization_times))
    print("Time to Find Min Solution:", time_to_min)
    print("Total time (2000 iterations):", total_time)



