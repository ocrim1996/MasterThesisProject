import math
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

    prob_file_name = folder_path + prob_file_name
    Q = np.loadtxt(prob_file_name)
    n = Q.shape[0]
    c = np.zeros(n, dtype=float)

    problemSMO = stqpSMOMultistartPerturbation.SMOAlgorithm(n, Q, c)

    #params = [10, 15, 20]
    #params = [[10, 10], [10, 15], [10, 20], [15, 10], [15, 15], [15, 20], [20, 10], [20, 15], [20, 20]]
    params = [20]
    #params = [[10, 10]]

    for param in params:
        obj_func_minimums = []
        min_times = []
        smo_counts_list = []

        for i in range(100):
            #print("Iterazione:", i)
            start_time = tm.default_timer()
            solution, smo_counts = problemSMO.solve_problem_multistart_random_points(param)
            #solution, smo_counts = problemSMO.solve_problem_multistart_random_points_perturbation(param[0], param[1])
            stop_time = tm.default_timer()
            time_exe = stop_time - start_time

            obj_func_minimums.append(solution)
            min_times.append(time_exe)
            smo_counts_list.append(smo_counts)

        obj_func_minimum = min(obj_func_minimums)

        #index = obj_func_minimums.index(obj_func_minimum)
        #min_time = min_times[index]

        #indexes = [i for i, x in enumerate(obj_func_minimums) if format(x, '.6f') == format(obj_func_minimum, '.6f')]
        indexes = [i for i, x in enumerate(obj_func_minimums) if math.isclose(x, obj_func_minimum, rel_tol=1e-6)]
        minimums_times = []

        for i in indexes:
            minimums_times.append(min_times[i])

        #min_time = min(minimums_times)
        avg_time = statistics.mean(minimums_times)
        avg_smo_counts = statistics.mean(smo_counts_list)

        print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
        #print("Parametri --> Nmax: {}, Mmax: {}".format(param[0], param[1]))
        print("Parametri --> Nmax: {}".format(param))
        print("Objective function minimum:", obj_func_minimum)
        print("Avg minimization time:", avg_time)
        print("Numero di volte corretto:", len(minimums_times))
        print("Numero (medio) chiamate a SMO:", avg_smo_counts)
