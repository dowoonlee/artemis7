import numpy as np

@staticmethod
def __getBin(rule, x, axis):
    result = None
    if axis==None:
        result = rule(x)
    elif axis==0:
        r = np.array([rule(x[i, :]) for i in range(x.shape[0])])
        result = r.reshape(-1, 1)
    elif axis==1:
        r = np.array([rule(x[:, i]) for i in range(x.shape[1])])
        result = r.reshape(1, -1)
    else:
        raise np.AxisError("axis %s is out of bounds for array of dimension %d"%(axis, x.ndim))
    return result

def sturges(x, axis=None):
    def __sturges(x):
        if np.ndim(x)==0:
            n = x
        else:
            n = len(x)
        return int(1 + np.ceil(np.log2(n)))
    return __getBin(__sturges, x, axis)

def doane(x, axis=None):
    def __doane(x):
        n = len(x)
        g1 = np.sum( (x - np.mean(x))**3 )
        g1 /= (np.std(x)**2)**(3/2)
        sig_g1 = np.sqrt(6*(n - 2)/(n+1)/(n+3))
        return  int(1 + np.log2(n) + np.log2(1+ abs(g1)/sig_g1))
    return __getBin(__doane, x, axis)

def scott(x, axis=None):
    def __scott(x):
        n = len(x)
        h = 3.49 * np.std(x) / (n**(1/3))
        return int(max(x)-min(x)/h)
    return __getBin(__scott, x, axis)

def freedman_diaconis(x, axis=None):
    def __freedman_diaconis(x):
        n = len(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        h = 2*iqr/(n**(1/3))
        return int(max(x)-min(x)/h)
    return __getBin(__freedman_diaconis, x, axis)
