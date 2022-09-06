import numpy as np
import torch
import faiss
import pandas as pd


def to_numpy(x):
    return x.cpu().data.numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j]
            is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def perform_faiss(query, index, K, d=60, gpu=False, nogpu=0, dist='euc'):
    '''kNN search with faiss
    query: query set
    index: gallery set
    K: number of nearest neighbors
    d: dimension of the embedding space
    gpu: gpu number used by faiss
    '''
    if dist == 'euc':
        index_flat = faiss.IndexFlatL2(d)
    elif dist == 'cos':
        index_flat = faiss.IndexFlatIP(d)

    if gpu:
        res = faiss.StandardGpuResources()
        # put the gallery on gpu:0
        gpu_index_flat = faiss.index_cpu_to_gpu(res, nogpu, index_flat)
        gpu_index_flat.add(index)
        D, I = gpu_index_flat.search(query, K)
    else:
        index_flat.add(index)
        D, I = index_flat.search(query, K)
    return I


def get_recall(close_idx, labels, K):
    k_close_idx = close_idx[:, :K]
    true_pos = np.equal(labels[k_close_idx], labels[:, None]).sum(axis=1)
    recall_at_k = true_pos > 0
    return np.mean(recall_at_k) * 100
