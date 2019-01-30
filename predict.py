import algo.jaro_wrinkler as jaro_wrinkler
import algo.custom_algorithm as custom_algorithm

# jaro-wrinkler
def jaroWrinkler(x, y):
    return jaro_wrinkler.get(x, y)

# 커스텀 알고리즘
def customAlgorithm(x, y):
    return custom_algorithm.get(x, y)
