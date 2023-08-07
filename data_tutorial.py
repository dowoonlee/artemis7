import numpy as np
from dwlib.util.make_movie import make_movie
from dwlib.datagenerator.virtual_drift_generator import concept_drift_generator
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning
import matplotlib.pyplot as plt
from astropy.time import Time
import pandas as pd


time_range = ["2012-01-01", "2020-01-01"]
t0 = Time("2014-01-01").mjd
#drift_time_seq = ["2014-01-01", "2014-04-01"]
drift_time_seq = [str(Time(t0+90*i, format="mjd").datetime64) for i in range(20)]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 20, 20
cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
n_drift_type=1, n_cont=n_cont, n_disc=n_disc,
dt = 1/24)
df = cdg.generate(drift= "virtual", drift_type="incremental", strength=20, noise=0.02)
cdg.save_to_csv("./dwlib/data/generated_table/")
# df.to_csv("./dwlib/data/generated_table/vi_c%02d_d%02d.csv"%(n_cont, n_disc))
# df = pd.read_csv("./dwlib/data/generated_table/vi_c%02d_d%02d.csv"%(n_cont, n_disc), index_col=0)


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
        act1 = np.vstack((act1, h1))
        act2 = np.vstack((act2, h2))
    return FID(act1, act2)
    

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class random_forest():
    def __init__(self, data_set, time, label):
        data_set = data_set.drop(columns=time)
        self.X = data_set.loc[:, data_set.columns != label].values
        self.y = data_set[label].values
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(max_depth=6)
        model.fit(X_train,y_train)

        self.time = time
        self.label = label
        self.model = model

    def validate(self, data_set):
        df = data_set.drop(columns=[self.time])
        X = df.loc[:, df.columns != self.label].values
        y = df[self.label].values

        model = self.model
        pred_lr = model.predict(X)
        accuracy = accuracy_score(pred_lr,y)*100
        return accuracy

d0 = df[(df.time>=Time("2012-01-01").mjd) & (df.time<Time("2013-01-01").mjd)]
tr = Time(time_range).mjd
rf = random_forest(d0,"time", "y")
t, fids, acc = [], [], []
dt = 7
for tt in np.arange(int(tr[0]), int(tr[1]), dt):
    dd = df[(df.time>tt) & (df.time<=tt+dt)]
    t.append(tt+dt/2)
    fids.append(actuator(d0.drop(columns=["time", "y"]).to_numpy(), dd.drop(columns=["time", "y"]).to_numpy()))
    acc.append(rf.validate(dd))

plt.figure(figsize =(12, 5))
ax = plt.subplot()
ax.plot(Time(t, format="mjd").datetime64, fids, c='k')
#ax.set_xlim(t[0], t[-1])

ax.grid(True)
# for dts in drift_time_seq:
#     ax.vlines(Time(dts).mjd, np.min(fids), np.max(fids))

axt = plt.twinx(ax)
axt.plot(Time(t, format="mjd").datetime64, acc, c='r', label="acc")
axt.legend(loc='best')
plt.show()


