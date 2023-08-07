import numpy as np

class PRS():
    def __init__(self, X, bins, seed=False):
        if seed:
            np.random.seed(seed)
        self.X = X
        self._xrange = (np.min(X), np.max(X))
        self._bins = bins
        self._wi, self._x_edge = np.histogram(self.X, bins = self._bins, range=self._xrange, density=True)
        self._wi_csum = np.cumsum(self._wi)
    
    def sample(self, data, n=100, resampling=False, index=False):
        """
        data : Target population
        n : The number of samples. Default is 100
        resampling : resample or not. Default is False
        index : return index rather value. Default is False
        """

        y, _ = np.histogram(data, bins=self._bins, range=self._xrange, density=True)
        X_max_bin = np.where(self._wi == np.max(self._wi))[0]
        n_limit = int(len(self.X) * y[X_max_bin] / self._wi[X_max_bin]*0.9)
        n = np.min([n, n_limit])

        result, cnt = np.zeros(n), 0
        while cnt<n:
            Y = np.random.random() * self._wi_csum[-1]
            idx = 0

            while self._wi_csum[idx]<Y:
                idx += 1
            sample_area = np.where((data >= self._x_edge[idx]) & (data<self._x_edge[idx + 1]))[0]

            if len(sample_area):
                sampled_idx = np.random.choice(sample_area)
                if index:
                    result[cnt] = sampled_idx.astype(np.int64)
                else:
                    result[cnt] = data[sampled_idx]
                cnt += 1

                if not resampling:
                    data = np.delete(data, sampled_idx)
        return result
    
    def generate(self, n=100, mode="uniform"):
        """
        Generate the random numbers following the given distribution
        n : The number of samples. Default is 100
        mode : The distribution for the sampling
        """
        result, cnt = np.zeros(n), 0
        bin_width = self._x_edge[1] - self._x_edge[0]

        while cnt < n:
            Y = np.random.random() * self._wi_csum[-1]
            idx = 0

            while self._wi_csum[idx] < Y:
                idx += 1
            
            if mode == "uniform":
                result[cnt] = np.random.rand() * (bin_width / 2) + (self._x_edge[idx -1] + self._x_edge[idx]) / 2
            elif mode == "gaussian":
                result[cnt] = np.random.randn() * (bin_width / 2) + (self._x_edge[idx -1] + self._x_edge[idx]) / 2
            cnt += 1
        return result
