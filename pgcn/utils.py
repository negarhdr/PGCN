import numpy as np
import scipy.sparse as sp
import torch


def load_data(path="../data/", dataset="cora"):
    """Load citation network dataset """
    print('Loading {} dataset...'.format(dataset))

    data_path = path + dataset

    features = sp.load_npz(data_path + '/normfeatures.npz')
    adj = sp.load_npz(data_path + '/normadj.npz')
    labels = np.load(data_path + '/labels.npy')
    train_mask = np.array(np.load(data_path + '/train_mask.npy'), dtype=int)
    val_mask = np.array(np.load(data_path + '/val_mask.npy'), dtype=int)
    test_mask = np.array(np.load(data_path + '/test_mask.npy'), dtype=int)
    idx_train = np.squeeze(np.argwhere(train_mask))
    idx_val = np.squeeze(np.argwhere(val_mask))
    idx_test = np.squeeze(np.argwhere(test_mask))
    ############ Laplacian matrix ########
    N = labels.shape[0]
    I_N = np.eye(N, N)
    Lplcn = I_N - np.array(adj.todense())
    labels_onehot = labels
    ######################################
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return adj, features, labels, idx_train, idx_val, idx_test, train_mask, Lplcn, labels_onehot


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
