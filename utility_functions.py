import numpy as np
import random


# Create vector 1/m, 0, ...., 1/m, 0, ...
def create_vector_uniform(dim):
    m = int(dim/10)
    vectX = np.zeros(dim, dtype=float)
    indexes = random.sample(range(dim), m)
    vectX[indexes] = 1/m
    return vectX


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

