"""Data set class and corresponding PyTorch loaders."""
from glob import glob

import numpy as np
import torch
from scipy.sparse import load_npz, tril
from torch.utils.data import DataLoader, Dataset

from config import CFG


class _FvmOpenFOAM(Dataset):
    """Sparse finite volume matrices from an OpenFOAM simulation.

    :param data_root: A string, the folder containing the matrices.
    """

    def __init__(self, data_root):
        self.files = glob(data_root+'L*.npz')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        l_matrix = load_npz(self.files[idx])
        row = l_matrix.row
        col = l_matrix.col
        val = l_matrix.data
        n_rows = l_matrix.shape[0]

        features = tril(l_matrix).data.astype(np.float32)
        coors = np.stack((tril(l_matrix).nnz*[0], tril(l_matrix).row,
                          tril(l_matrix).col), axis=1)

        i = torch.LongTensor(np.vstack((row, col)))
        val = torch.FloatTensor(val)
        return (features, coors, l_matrix.shape,
                torch.sparse.FloatTensor(i, val, torch.Size([n_rows, n_rows])))


def init_loaders():
    """Initialize loders for train/validate/test data set."""
    data = _FvmOpenFOAM(CFG['DATA_ROOT'])
    # Check if all system matrices are SPD.
    n_train = int(CFG['PC_TRAIN']*len(data))
    n_val = int(CFG['PC_VAL']*len(data))
    n_test = len(data)-n_train-n_val

    train_data, val_data, test_data = torch.utils.data.random_split(
        data, (n_train, n_val, n_test))
    # https://stackoverflow.com/questions/55820303
    torch.manual_seed(torch.initial_seed())
    return (DataLoader(train_data), DataLoader(val_data),
            DataLoader(test_data))
