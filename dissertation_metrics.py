import numpy as np
from fastdtw import fastdtw
from similaritymeasures import frechet_dist

def L0(original, adversary):
    return np.sum((np.abs(original - adversary) > 0.05)) / len(original)

def L1(original, adversary):
    return np.sum(np.abs(original - adversary))

def L2(original, adversary):
    return np.linalg.norm(original - adversary)

def Linf(original, adversary):
    return np.max(np.abs(original - adversary))

def cosine(original, adversary):
    norm_1 = np.linalg.norm(original)
    norm_2 = np.linalg.norm(adversary)
    return 1 - abs(np.dot(original, adversary)/(norm_1 * norm_2))

def pearson(original, adversary):
    corr_matrix = np.corrcoef(original, adversary)
    return 1 - abs(corr_matrix[0][1])

def DTW(original, adversary):
    distance, _ = fastdtw(original, adversary, radius = len(original), dist = 2)
    return distance

def frechet(original, adversary):
    original_3d = []
    for element in original:
        original_3d.append([element, 0])
    original_3d = np.array(original_3d)
    adversary_3d = []
    for element in adversary:
        adversary_3d.append([element, 0])
    adversary_3d = np.array(adversary_3d)
    return frechet_dist(original_3d, adversary_3d)
