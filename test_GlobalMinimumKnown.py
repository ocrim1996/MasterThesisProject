import numpy as np
import timeit as tm
import StQPAlgorithmSMOMultistartPerturbation as stqpSMOMultistartPerturbation
import statistics


folder_path = "Problems/Dataset_Generic/"
prob_file_names = ["Problem_500x500(0.25)_1.txt", "Problem_500x500(0.25)_2.txt", "Problem_500x500(0.25)_3.txt",
                   "Problem_500x500(0.5)_1.txt", "Problem_500x500(0.5)_2.txt", "Problem_500x500(0.5)_3.txt",
                   "Problem_1000x1000(0.25).txt"]

"""
folder_path = "Problems/Dataset_BSU/"
prob_file_names = ["BSU_5x5.txt", "BSU_6x6.txt", "BSU_7x7.txt", "BSU_8x8.txt", "BSU_9x9.txt"]
"""

for prob_file_name in prob_file_names:
    problem_name = prob_file_name
    prob_file_name = folder_path + prob_file_name
    Q = np.loadtxt(prob_file_name)
    n = Q.shape[0]
    c = np.zeros(n, dtype=float)

    problemSMO = stqpSMOMultistartPerturbation.SMOAlgorithm(n, Q, c)

    min_times = []
    smo_counts_list = []
    solutions = []

    for i in range(10):
        start_time = tm.default_timer()
        solution, smo_counts = problemSMO.solve_problem_multistart_random_points(problem_name)
        #solution, smo_counts = problemSMO.solve_problem_multistart_random_points_perturbation(problem_name, 5)
        stop_time = tm.default_timer()
        time_exe = stop_time - start_time

        min_times.append(time_exe)
        smo_counts_list.append(smo_counts)
        solutions.append(solution)

    avg_time = statistics.mean(min_times)
    avg_smo_counts = statistics.mean(smo_counts_list)


    print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
    #print("Parametri --> Nmax: {}, Mmax: {}".format(param[0], param[1]))
    #print("Parametri --> Nmax: {}".format(param))
    print("Objective function minimum:", min(solutions))
    print("Avg minimization time:", avg_time)
    print("Numero (medio) chiamate a SMO:", avg_smo_counts)
