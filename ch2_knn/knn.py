from collections import Counter
import numpy as np

def classify0(input_, dataset, labels, k):
    size0 = dataset.shape[0]
    diff_mat = np.tile(input_, (size0, 1)) - dataset;
    distances = ((diff_mat ** 2).sum(axis=1)) ** 0.5;
    sorted_dist_indices = distances.argsort()
    vote_labels = [labels[sorted_dist_indices[i]] for i in range(k)]
    cnt = Counter(vote_labels)
    sorted_cnt = sorted(cnt.items(), key=lambda t: t[1], reverse=True)
    return sorted_cnt[0][0]