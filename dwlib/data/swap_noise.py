import numpy as np
from numpy.random import choice


class swap_noise():
    def __init__(self, data):
        self.df = data.copy()
        self.columns = data.columns

    def _int_to_idx(self, idxs):
        nrow = self.df.shape[0]
        cols = idxs//nrow
        rows = idxs%nrow
        idxs_rc = [[r, self.columns[c]] for r, c in zip(rows, cols)]
        return np.array(idxs_rc)

    def generate(self, noise):
        if noise<0:
            noise=0
        elif noise<1:
            noise = int(noise * self.df.shape[0] * self.df.shape[1])
        
        idxs_int = choice(self.df.shape[0]*self.df.shape[1], noise, replace=False)
        idxs = self._int_to_idx(idxs_int)
        for col in self.columns:
            fr_idx = np.where(idxs[:, 1]==col)[0]
            if len(fr_idx)==0:
                continue
            to = self.df.loc[:, col].to_numpy()
            for el in idxs[:,0][fr_idx]:
                cho_to = choice(to)
                self.df.loc[int(el), col] = cho_to
        return self.df