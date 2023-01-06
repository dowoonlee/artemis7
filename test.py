from dwlib.util.concept_drift_generator import concept_drift_generator
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np

time_range = ["2012-01-01", "2018-01-01"]
drift_time_seq = ["2014-01-01"]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 6, 0
cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
n_drift_type=1, n_cont=n_cont, n_disc=n_disc,
dt = 1/24)
base = cdg.generate(drift= "real", drift_type="sudden", strength=5, noise=0.1)

# befs = base[base.time<Time(drift_time_seq[0]).mjd]
# afts = base[base.time>=Time(drift_time_seq[0]).mjd]

# b0, b1 = (befs.y==0).to_numpy(), (befs.y==1).to_numpy()
# a0, a1 = (afts.y==0).to_numpy(), (afts.y==1).to_numpy()
# c1, c2 = cdg.att_cols[0], cdg.att_cols[1]
# plt.scatter(befs["C%02d"%c1][b0], befs["C%02d"%c2][b0], s=1, alpha=0.1)
# plt.scatter(befs["C%02d"%c1][b1], befs["C%02d"%c2][b1], s=1, alpha=0.1)
# plt.scatter(afts["C%02d"%c1][a0], afts["C%02d"%c2][a0], s=1, alpha=0.1)
# plt.scatter(afts["C%02d"%c1][a1], afts["C%02d"%c2][a1], s=1, alpha=0.1)

# plt.legend(loc='best')
# plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class random_forest():
    def __init__(self, data_set, target):
        self.target = target
        data_set = data_set.drop(columns="time")
        self.X = data_set.loc[:, data_set.columns != self.target].values
        self.y = data_set[self.target].values
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(max_depth=6)
        model.fit(X_train,y_train)
        print("ACC_PREV: ", accuracy_score(model.predict(X_test), y_test)*100)
        self.model = model

    def validate(self, data_set):
        data_set = data_set.drop(columns="time")
        X = data_set.loc[:, data_set.columns != self.target].values
        y = data_set[self.target].values

        model = self.model
        pred_lr = model.predict(X)
        accuracy = accuracy_score(pred_lr,y)*100
        return accuracy

tset = base[base.time<Time("2013-01-01").mjd]
vset = base[base.time>Time("2013-01-01").mjd]


dt = 30
rf = random_forest(tset, "y")
time, acc = [], []
for t in range(int(vset.time.min()), int(vset.time.max()), dt):
    dvs = vset[(vset.time>=t) & (vset.time<t+dt)]
    if len(dvs)<10:
        break
    time.append(t+dt/2)
    acc.append(rf.validate(dvs))

plt.plot(time, acc)
for dts in drift_time_seq:
    plt.vlines(Time(dts).mjd, min(acc), max(acc), color='k')
#plt.ylim(min(acc), max(acc))
plt.show()



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

dt = 30
time, fids = [], []
ori = tset.drop(columns=["time", "y"]).to_numpy()
for t in range(int(vset.time.min()), int(vset.time.max()), dt):
    dvs = vset[(vset.time>=t) & (vset.time<t+dt)]
    if len(dvs)<10:
        break
    time.append(t+dt/2)
    fids.append(actuator(ori, dvs.drop(columns=["time", "y"]).to_numpy()))

plt.plot(time, fids)
for dts in drift_time_seq:
    plt.vlines(Time(dts).mjd, min(fids), max(fids), color='k')
plt.show()