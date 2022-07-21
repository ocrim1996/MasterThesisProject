import random
import numpy as np
import utility_functions as uf

# Prima regola di perturbazione del vettore vectorX.
def perturbation1(vectorX):
    same_index = True
    while same_index:
        index_i = random.randint(0, len(vectorX) - 1)
        index_j = random.randint(0, len(vectorX) - 1)
        if index_i != index_j and vectorX[index_i] > 1e-5:
            same_index = False

    rho = random.uniform(0, 1)

    # for h, value in enumerate(vectorX):
    #     if h == index_i:
    #         value = value - (rho * value)
    #         vectorX[h] = value
    #     elif h == index_j:
    #         value = value + (rho * vectorX[index_i])
    #         vectorX[h] = value

    value = vectorX[index_i]
    diff = rho * value
    vectorX[index_i] -= diff
    vectorX[index_j] += diff


    return vectorX


def perturbation2(vectorX, epsilon, dim):
    x_tilde = np.random.random_sample(dim)
    x_tilde = x_tilde / np.sum(x_tilde)

    rho = random.uniform(0, epsilon)

    x_perturb = (1 - rho) * vectorX + (rho * x_tilde)
    return x_perturb


# Exchange all variables non zero except one.
def perturbation3(vectorX, non_zero_indexes, complementary_indexes):
    for index in range(len(non_zero_indexes)):
        uf.swap_positions(vectorX, non_zero_indexes[index], complementary_indexes[index])
    return vectorX

