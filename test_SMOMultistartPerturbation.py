import numpy as np
import timeit as tm
import StQPAlgorithmSMOMultistartPerturbation as stqpSMOMultistartPerturbation
import statistics

#prob_file_name = "Problems/Dataset_Generic/Problem_3x3(0.75).txt"
prob_file_name = "Problems/Dataset_Generic/Problem_200x200(0.75)_2.txt"
#prob_file_name = "Problems/Dataset_BSU/BSU_8x8.txt"

Q = np.loadtxt(prob_file_name)
n = Q.shape[0]
c = np.zeros(n, dtype=float)

problemSMO = stqpSMOMultistartPerturbation.SMOAlgorithm(n, Q, c)

"""
start_time = tm.default_timer()
#solution, counts = problemSMO.solve_problem_multistart_perturbation()
#solution = problemSMO.solve_problem_multistart_ones()
#solution, counts = problemSMO.solve_problem_multistart_random_points(20)
solution = problemSMO.solve_problem_multistart_random_points_perturbation(10, 10, 0.2)
stop_time = tm.default_timer()

print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
print("Objective function minimum:", solution)
print("Minimization time:", stop_time - start_time)
#print(counts)
"""

"""
obj_func_minimums = []
min_times = []

for i in range(10):
    print("Iterazione:", i)
    start_time = tm.default_timer()
    #solution, counts = problemSMO.solve_problem_multistart_random_points(10)
    solution = problemSMO.solve_problem_multistart_random_points_perturbation(10, 10, 0.2)
    stop_time = tm.default_timer()

    obj_func_minimums.append(solution)
    time_exe = stop_time - start_time
    min_times.append(time_exe)

obj_func_minimum = min(obj_func_minimums)
#index = obj_func_minimums.index(obj_func_minimum)
#min_time = min_times[index]

indexes = [i for i, x in enumerate(obj_func_minimums) if x == obj_func_minimum]
minimums_times = []

for i in indexes:
    minimums_times.append(min_times[i])

min_time = min(minimums_times)

print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
print("Objective function minimum:", obj_func_minimum)
print("Minimization time:", min_time)
print("Numero di volte corretto:", len(minimums_times))
"""

"""
params = [[10, 10, 0.1], [10, 10, 0.15], [10, 10, 0.2], [15, 15, 0.1], [15, 15, 0.15], [15, 15, 0.2], [20, 10, 0.1],
          [10, 20, 0.1], [20, 10, 0.2], [10, 20, 0.2]]

for param in params:
    obj_func_minimums = []
    min_times = []

    for i in range(250):
        print("Iterazione:", i)
        start_time = tm.default_timer()
        #solution, counts = problemSMO.solve_problem_multistart_random_points(10)
        solution = problemSMO.solve_problem_multistart_random_points_perturbation(param[0], param[1], param[2])
        stop_time = tm.default_timer()

        obj_func_minimums.append(solution)
        time_exe = stop_time - start_time
        min_times.append(time_exe)

    obj_func_minimum = min(obj_func_minimums)
    #index = obj_func_minimums.index(obj_func_minimum)
    #min_time = min_times[index]

    indexes = [i for i, x in enumerate(obj_func_minimums) if x == obj_func_minimum]
    minimums_times = []

    for i in indexes:
        minimums_times.append(min_times[i])

    min_time = min(minimums_times)

    print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
    print("Parametri --> Nmax: {}, Mmax: {}, eps: {}".format(param[0], param[1], param[2]))
    print("Objective function minimum:", obj_func_minimum)
    print("Minimization time:", min_time)
    print("Numero di volte corretto:", len(minimums_times))
"""

#params = [10, 15, 20]
params = [[10, 10], [10, 15], [10, 20], [15, 10], [15, 15], [15, 20], [20, 10], [20, 15], [20, 20]]

for param in params:
    obj_func_minimums = []
    min_times = []

    for i in range(100):
        #print("Iterazione:", i)
        start_time = tm.default_timer()
        #solution = problemSMO.solve_problem_multistart_random_points(param)
        solution = problemSMO.solve_problem_multistart_random_points_perturbation(param[0], param[1])
        stop_time = tm.default_timer()
        time_exe = stop_time - start_time

        obj_func_minimums.append(solution)
        min_times.append(time_exe)

    obj_func_minimum = min(obj_func_minimums)

    #index = obj_func_minimums.index(obj_func_minimum)
    #min_time = min_times[index]

    indexes = [i for i, x in enumerate(obj_func_minimums) if format(x, '.6f') == format(obj_func_minimum, '.6f')]
    minimums_times = []

    for i in indexes:
        minimums_times.append(min_times[i])

    #min_time = min(minimums_times)
    avg_time = statistics.mean(minimums_times)

    print("\nPROBLEM:", prob_file_name.split("/")[2][:-4])
    print("Parametri --> Nmax: {}, Mmax: {}".format(param[0], param[1]))
    #print("Parametri --> Nmax: {}".format(param))
    print("Objective function minimum:", obj_func_minimum)
    print("Avg minimization time:", avg_time)
    print("Numero di volte corretto:", len(minimums_times))