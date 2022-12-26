from dwlib.stats.binning import binning
from dwlib.stats.piecewise_rejection_sampling import PRS
from dwlib.stats.frechet_inception_distance import FID

import numpy as np


x = np.random.rand(100, 100)
y = np.random.rand(100, 100)
def actuator(dset1, dset2):
    norm=True
    b1 = binning(dset1[:,0])
    b2 = binning(dset2[:,0])
    bins = np.max([b1.bins(), b2.bins()])

    xr = (np.min([dset1[:, 0].min(), dset2[:, 0].min()]), np.max([dset1[:, 0].max(), dset2[:, 0].max()]))
    h1, _ = np.histogram(dset1[:, 0], bins=bins, range=xr, density=norm)
    h2, _ = np.histogram(dset2[:, 0], bins=bins, range=xr, density=norm)

    act1, act2 = h1, h2

    for i in range(1, len(dset1[0, :])):
        xr = (np.min([dset1[:, i].min(), dset2[:, i].min()]), np.max([dset1[:, i].max(), dset2[:, i].max()]))
        h1, _ = np.histogram(dset1[:, i], bins=bins, range=xr, density=norm)
        h2, _ = np.histogram(dset2[:, i], bins=bins, range=xr, density=norm)
        act1 = np.concatenate((act1, h1), axis=0)
        act2 = np.concatenate((act2, h2), axis=0)
    act1 = act1.reshape((bins, len(dset1[0, :])))
    act2 = act2.reshape((bins, len(dset2[0, :])))
    return FID(act1, act2)

print(actuator(x, y))
    