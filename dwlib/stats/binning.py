import numpy as np

class binning():
    def __init__(self, X):
        if isinstance(X, list):
            X = np.array(X)
        self.X = X
        if self.X.ndim>2:
            raise np.AxisError("Dimension of array is too deep")
        self.methods_ = ["Sturges", "Doane", "Scott", "Freedman-Diaconis"]

    def bins(self, method="Sturges", axis = 0):

        if axis >= self.X.ndim:
            raise np.AxisError("axis %d is out of bounds for array of dimension %d"%(axis, self.X.ndim))
        
        if method == "Sturges":
            def bfunc(X):
                n = len(X)
                return 1+3.322*np.log10(n)
        elif method == "Doane":
            def bfunc(X):
                n = len(X)
                g1 = np.sum( (X - np.mean(X))**3 )
                g1 /= (np.std(X)**2)**(3/2)
                sig_g1 = np.sqrt(6*(n - 2)/(n+1)/(n+3))
                return  1 + np.log2(n) + np.log2(1+ abs(g1)/sig_g1)
        elif method == "Scott":
            def bfunc(X):
                n = len(X)
                return 3.49 * np.std(X) / (n**(1/3))
        elif method == "Freedman-Diaconis":
            def bfunc(X):
                n = len(X)
                IQR = np.percentile(X, 75) - np.percentile(X, 25)
                return 2* IQR / (n**(1/3))
        

        if self.X.ndim == 1:

            return int(bfunc(self.X))
        elif axis == 0:
            _ = self.X.shape[axis]
            return np.array([int(bfunc(self.X[i, :])) for i in range(_)])
        elif axis == 1:
            _ = self.X.shape[axis]
            return np.array([int(bfunc(self.X[:, i])) for i in range(_)])


        
