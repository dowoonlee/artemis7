import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from artemis.stats.binning import *

@staticmethod
def isdiscrete(x):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    elif isinstance(x, list):
        x = np.array(x)
    
    nuniq = len(np.unique(x))
    nbin = sturges(x)
    if nuniq < nbin:
        return True
    else:
        False

class MVSampler():
    def __init__(self, reference, inference, target_cols):
        """
        reference : pd.DataFrame.
        inference : pd.DataFrame.
        target_cols : array-like.
        """
        self._pdf = {}
        self._cnt = np.zeros(reference.shape[0])
        self.reference = reference.copy()
        del reference
        self.inference = inference.copy()
        del inference
        self.target_cols = target_cols

        bins = sturges(self.inference.shape[0])
        for col in self.target_cols:
            inf = self.inference[col].to_numpy()
            if isdiscrete(inf): #해당 column이 discrete 하다면
                b, p = np.unique(inf, return_counts=True)
                p, b = p[np.argsort(b)], np.sort(b)

            else: # continuous하다면
                p, bin_edge = np.histogram(self.inference[col].values,
                                            bins=np.min([bins, len(self.inference[col].unique())]),
                                            # bins = bins,
                                            density=True)
                b, bin_width = (bin_edge[1:] + bin_edge[:-1])/2, (bin_edge[1]-bin_edge[0])/2
                
                b, p = np.insert(b, 0, self.inference[col].min()-bin_width), np.insert(p, 0, 0)
                b, p = np.append(b, self.inference[col].max()+bin_width), np.append(p, 0)
            
            f = interp1d(b, p, kind="linear")
            self._pdf[col] = (p, b, f)
            lcond = self.reference[col].values>=b.min()
            rcond = self.reference[col].values<=b.max()
            self._cnt += (lcond*rcond*1)

    def build(self, row_min = -1, outlier = 0):
        """
        row_min : int. sampling에 사용될 reference의 최소 row 갯수
        outlier : int. ref/inf 의 target_col 분포를 비교할 때, 고려하지 않을 column의 갯수
        모든 target_col이 ref/inf 분포가 겹치지 않을 수 있기 때문에 일정 row 갯수가 확보될 때 까지 증가
        """
        if row_min  < 0:
            row_min = int(self.inference.shape[0] * 0.1)

        nsize = len(np.where(self._cnt>=len(self.target_cols)-outlier)[0])
        target_cols = self.target_cols
        while (nsize <= row_min) and (outlier < len(self.target_cols)):
            outlier += 1
            target_cols = self.target_cols[:len(self.target_cols)-outlier]
            self._cnt = np.zeros(self.reference.shape[0])
            for col in target_cols:
                b = self._pdf[col][1]
                lcond = self.reference[col].values>=b.min()
                rcond = self.reference[col].values<=b.max()
                self._cnt += (lcond*rcond*1)
            nsize = len(np.where(self._cnt>=len(target_cols))[0])
        self._outlier = outlier

        self.log_likelihood = np.zeros(self.reference.shape[0])
        for i, idx in enumerate(self.reference.index.to_numpy()):
            if self._cnt[i]>=len(target_cols):
                p = []
                for col in target_cols:
                    _pdf = self._pdf[col]
                    if self.reference.at[idx, col]<=_pdf[1].min() or self.reference.at[idx, col]>=_pdf[1].max():
                        p.append(0)
                    else:
                        p.append(self._pdf[col][2](self.reference.at[idx, col]))
                p = np.array(p)
                self.log_likelihood[i] = np.sum(np.log10(p[np.where(p!=0)[0]]))
                # np.power(10, np.sum(np.log10(p[np.where(p!=0)[0]])))

    def sample(self, n=-1, seed=None, return_index=False):
        """
        n : int. sample의 크기
        return_index : bool. True일 경우 sample의 index를 array로 return, False일 경우 sample을 pd.DataFrame으로 return. 기본은 False
        return : pd.DataFrame / array-like
        """
        n = n if n>0 else self.inference.shape[0]//10
        prob = np.power(10, self.log_likelihood)
        zeros = np.where(self.log_likelihood==0)[0]
        prob[zeros] = 0
        prob /= np.sum(prob)

        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(0, self.reference.shape[0]), replace=True, p=prob, size=n)
        if return_index:
            return idx
        return self.reference.iloc[idx, :]