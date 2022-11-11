from statistics import mean
import numpy as np
import random
import copy
import math


# Create vector 1/m, 0, ...., 1/m, 0, ...
def create_vector_uniform(dim):
    m = int(dim/10)
    #m = math.floor(dim/2)
    vectX = np.zeros(dim, dtype=float)
    indexes = random.sample(range(dim), m)
    vectX[indexes] = 1/m
    return vectX


# Create random vector 0, x, 0, ..., y, 0, z, 0, ...
def create_random_vector(dim):
    m = int(dim / 5)
    vectX = np.zeros(dim, dtype=float)
    indexes = random.sample(range(dim), m)

    vector = sorted(np.random.uniform(0, 1, m))
    sum = 0
    for k, v in enumerate(indexes[:-1]):
        value = vector[k + 1] - vector[k]
        sum = sum + value
        vectX[v] = value
    vectX[indexes[len(indexes) - 1]] = 1 - sum
    return vectX


# Create a vector starting from the convexity graph of the matrix.
def create_convexity_graph_vector(matrix):
    Q = copy.deepcopy(matrix)
    n = Q.shape[0]

    H = np.zeros([n, n])

    # Si legge per righe la matrice e si applica la formula e si mette a zero gli elementi Hij>0 e
    # a uno gli elementi Hij<0
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i][j] = (2 * Q[i][j] - Q[i][i] - Q[j][j]) / 2
                if H[i][j] > 0:
                    H[i][j] = 0
                elif H[i][j] < 0:
                    H[i][j] = 1

    # Si mette la diagonale della matrice pari a uno.
    np.fill_diagonal(H, 1)

    # Si conta per ogni colonna della matrice il numero (m) di 1 e la loro posizione e si crea un vettore di partenza
    # che abbia in corrispondenza degli uno, un valore pari a 1/m
    starting_points = []
    for i in range(n):
        m = np.count_nonzero(H[:][i] == 1)
        indexes = []
        for index in range(len(H[:][i])):
            if H[index][i] == 1:
                indexes.append(index)
        starting_point = np.zeros(len(H[:][i]))
        starting_point[indexes] = 1 / m
        starting_points.append(starting_point)

    starting_points = np.array(starting_points)

    # Si conta il numero di zeri e di uni (gli uni sono 1/m) per tutti gli starting points
    # (ovvero le colonne della matrice)
    zeros = []
    ones = []
    for starting_point in starting_points:
        zeros_count = np.count_nonzero(starting_point == 0)
        ones_count = np.count_nonzero(starting_point != 0)
        zeros.append(zeros_count)
        ones.append(ones_count)

    return starting_points, mean(zeros), mean(ones)


# Create a vector starting from the convexity graph of the matrix and with AND Bitwise operator
def create_convexity_graph_vector_and_bitwise(matrix, i_best_prob):
    Q = copy.deepcopy(matrix)
    n = Q.shape[0]

    H = np.zeros([n, n])

    # Si legge per righe la matrice si applica la formula e si mette a zero gli elementi Hij>0 e
    # a uno gli elementi Hij<0
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i][j] = (2 * Q[i][j] - Q[i][i] - Q[j][j]) / 2
                if H[i][j] > 0:
                    H[i][j] = 0
                elif H[i][j] < 0:
                    H[i][j] = 1

    # Si mette la diagonale della matrice pari a uno.
    np.fill_diagonal(H, 1)
    H = H.astype(int)

    """
    # AND Bitwise algoritmo pensato con Matteo.
    modified_columns = []
    for i_col in range(n):
        column = H[:][i_col]
        for j in range(n):
            if column[j] == 1:
                column = column & H[:][j]
        modified_columns.append(column)

        if i_col == i_best_prob:
            ones_indices_column = [i for i in range(len(column)) if column[i] == 1]
            print(ones_indices_column)

    modified_columns = np.array(modified_columns)
    """

    # Si calcola la colonna C relativa a H_i_best
    modified_columns = []
    column = H[:][i_best_prob]
    for j in range(n):
        if column[j] == 1:
            column = column & H[:][j]
    modified_columns.append(column)
    modified_columns = np.array(modified_columns)

    """
    # Si modifica la matrice binaria prendendo ogni colonna e iterativamente eseguire un'operazione di AND Logico con
    # un'altra i-esima colonna corrispondente all'i-esimo 1 presente nella colonna di partenza presa in considerazione.
    modified_columns = []
    for i_col in range(n):
        column = H[:][i_col]
        ones_count = np.count_nonzero(column == 1)
        ones_indices = [i for i in range(len(column)) if column[i] == 1]
        if i_col in ones_indices:
            ones_indices.remove(i_col)

        iterations = 0
        while True:
            if ones_count <= (n * 5) / 100 or iterations > n:
                break
            column = column & H[:][ones_indices[0]]
            ones_count = np.count_nonzero(column == 1)
            ones_indices = [i for i in range(len(column)) if column[i] == 1]
            if i_col in ones_indices:
                ones_indices.remove(i_col)

            iterations = iterations + 1

        modified_columns.append(column)

    modified_columns = np.array(modified_columns)
    """

    # Si conta per ogni colonna della matrice il numero (m) di 1 e la loro posizione e si crea un vettore di partenza
    # che abbia in corrispondenza degli uno, un valore pari a 1/m
    starting_points = []
    for i in range(len(modified_columns)):
        m = np.count_nonzero(modified_columns[i] == 1)
        indices = [idx for idx in range(len(modified_columns[i])) if modified_columns[i][idx] == 1]
        starting_point = np.zeros(len(modified_columns[i]))
        starting_point[indices] = 1 / m
        starting_points.append(starting_point)

    starting_points = np.array(starting_points)

    # Si conta il numero di zeri e di uni (gli uni sono 1/m) per tutti gli starting points
    # (ovvero le colonne della matrice)
    zeros = []
    ones = []
    for starting_point in starting_points:
        zeros_count = np.count_nonzero(starting_point == 0)
        ones_count = np.count_nonzero(starting_point != 0)
        zeros.append(zeros_count)
        ones.append(ones_count)

    return starting_points, mean(zeros), mean(ones)


# Si crea un vettore con solo due indici diversi da zero e questi indici si prendono uno random e uno in corrispondenza
# di un 1 (preso random) sulla colonna relativa al primo indice scelto a random
def create_vector_two_indices(matrix):
    Q = copy.deepcopy(matrix)
    n = Q.shape[0]

    H = np.zeros([n, n])

    # Si legge per righe la matrice si applica la formula e si mette a zero gli elementi Hij>0 e
    # a uno gli elementi Hij<0
    for i in range(n):
        for j in range(n):
            if i != j:
                H[i][j] = (2 * Q[i][j] - Q[i][i] - Q[j][j]) / 2
                if H[i][j] > 0:
                    H[i][j] = 0
                elif H[i][j] < 0:
                    H[i][j] = 1

    # Si mette la diagonale della matrice pari a uno.
    np.fill_diagonal(H, 1)
    H = H.astype(int)

    idx_i = random.randint(0, n-1)
    column = H[:][idx_i]
    indices_J = [idx for idx in range(len(column)) if column[idx] == 1]
    size = len(indices_J)
    k = random.randint(0, size - 1)
    idx_j_hat = indices_J[k]

    starting_point = np.zeros(n)
    starting_point[idx_i] = 0.5
    starting_point[idx_j_hat] = 0.5

    return starting_point


# Difference of two list.
def list_difference(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


# Swap function.
def swap_positions(li, pos1, pos2):
    li[pos1], li[pos2] = li[pos2], li[pos1]
    return li


# Non zero indexes and complementary indexes.
def indexes_list(vector):
    all_indexes = np.arange(len(vector))
    non_zero_indexes = np.nonzero(vector)[0]
    complementary_indexes = np.asarray(list_difference(all_indexes, non_zero_indexes))

    np.random.shuffle(complementary_indexes)
    np.random.shuffle(non_zero_indexes)
    non_zero_indexes = np.delete(non_zero_indexes, 0)

    return non_zero_indexes, complementary_indexes


# To truncate float numbers.
def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

