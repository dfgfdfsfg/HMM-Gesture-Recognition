import numpy as np
from Kmeans import *

def gcff(path, delta=2):

    cluster_centers = Kmeans_(path)
    data = cluster_centers.T
    ans = [data]
    diff_1 = []
    diff_2 = []
    if delta >= 1:
        for line in data:
            diff = np.diff(line, prepend=0)
            diff_1.append(diff)
        ans.append(diff_1)
    if delta >= 2:
        for line in diff_1:
            diff = np.diff(line, prepend=0)
            diff_2.append(diff)
        ans.append(diff_2)
    # print(ans)
    return np.transpose(np.concatenate(ans, axis=0), [1, 0])