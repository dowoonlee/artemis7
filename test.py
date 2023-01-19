from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint, choice, gamma
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from dwlib.data.concept_drift_generator import concept_drift_generator
from dwlib.data.experiment import experiment
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning
from dwlib.data.swap_noise import swap_noise
from dwlib.util.progressbar import progressbar
from dwlib.stats.population_stability_index import population_stability_index

prob_type = "classification"
time_range = ["2012-01-01", "2020-01-01"]
drift_time_seq = ["2016-01-01"]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 5, 5
strength = 50
noise = 0.1
path = "./dwlib/data/generated_table/"
drift = "virtual"
drift_type = "incremental"

file_name = "test"#(prob_type[0]+drift[0]+drift_type[0]).upper()+"%d%d"%(n_cont, n_disc)

cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
prob_type=prob_type,
n_drift_type=1, n_cont = n_cont, n_disc=n_disc,
dt = 1/24)
dts = Time(drift_time_seq[0]).mjd
base = cdg.generate(drift= drift, drift_type=drift_type, strength=strength, noise=noise)
cdg.save_to_csv(path = path, file_name=file_name)


df = pd.read_csv("./dwlib/data/generated_table/test.csv", index_col=0)
df = df.drop(columns=["time", "y", "drift"]).iloc[:1000]


def labeling(df):
    arr = df.to_numpy()
    s = np.sum(arr, axis=1)
    m = np.mean(s)
    y = (s>m)*1
    return y


class testrun():
    def __init__(self, df, noises):
        sn = swap_noise(df)
        total_size = df.shape[0]*df.shape[1]
        weight = noises/total_size
        acc = np.zeros(len(noises))
        fid = np.zeros(len(noises))
        d, p = np.zeros(len(noises)), np.zeros(len(noises))
        psi = np.zeros(len(noises))


        X0 = df.values
        print(X0[:,0].shape)
        y0 = labeling(df)
        X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.2, random_state=42)
        model = RandomForestClassifier(max_depth=6)
        model.fit(X_train,y_train)

        pb = progressbar(len(noises))
        pb.start()
        for i, noise in enumerate(noises):
            pb.update(i)
            df_noise = sn.generate(noise)
            X = df_noise.values
            pred_lr = model.predict(X)
            acc[i] = accuracy_score(pred_lr, y0)*100
            fid[i] = FID(X, X0)
            for j in range(len(X0[0,:])):
                dsub, psub = ks_2samp(X[:,j], X0[:, j])
                psi_sub = population_stability_index(X[:,j], X0[:, j])
                d[i] += dsub
                p[i] += psub
                psi[i] += psi_sub
        
        self.weight = weight
        self.acc = acc
        self.fid = fid
        self.d = d/len(X0[0, :])
        self.p = p/len(X0[0, :])
        self.psi = psi/len(X0[0, :])

ratio = 0.15
ratio_arr = np.arange(1, int(ratio*df.shape[0]*df.shape[1]))
tr = testrun(df, ratio_arr)

plt.figure(figsize=(12, 8))
nr = 4
gs = GridSpec(nrows=nr, ncols=1)
gs.update(hspace=0, wspace=0)
axes = [plt.subplot(gs[i]) for i in range(nr)]


w = np.where(tr.p<=0.05)[0][0]

axes[0].plot(tr.weight, tr.fid, c='k',label="FID")
axes[1].plot(tr.weight, tr.d, c='r', label="D")
axes[2].plot(tr.weight, tr.p, c='b', label="p")
axes[3].plot(tr.weight, tr.psi, c='g', label="PSI")



for ax in axes:
    ax.set_xlim(0, ratio)
    ax.legend(loc='best')
    ax.set_ylim(0, ax.set_ylim()[1])
    ax.vlines(tr.weight[w], 0, ax.set_ylim()[1], linestyles=":", colors="k")
plt.show()