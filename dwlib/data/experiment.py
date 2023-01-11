import numpy as np
import pandas as pd
import json
from scipy.stats import ks_2samp
from astropy.time import Time
import sys
sys.path.append('C:\\dev\\dwlib')
from dwlib.util.progressbar import progressbar
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning
from dwlib.stats.population_stability_index import population_stability_index as psi


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class experiment():
    def __init__(self, path, file_name, dt):
        with open(path+"%s.json"%file_name, "r") as f:
            file_info = json.load(f)
        self._file_info = file_info
        self.df = pd.read_csv(path+"%s.csv"%file_info["file"], index_col=0)
        self.dt = dt

        r_init = 0.7
        t_init = r_init*self._file_info["drift_time"][0] + (1-r_init) * self._file_info["time_range"][0]
        self.df_init = self.df[self.df.time<t_init]
        self.trange = np.arange(int(t_init), int(self.df.time.max()-dt))
        self.time = np.array([self.df[(self.df.time>=t) & (self.df.time<t+self.dt)].time.mean() for t in self.trange])          

    def rf_init(self):
        df_init = self.df_init.drop(columns="time")
        X = df_init.loc[:, df_init.columns != "y"].values
        y = df_init.y.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(max_depth=6)
        model.fit(X_train,y_train)
        accruacy = np.zeros(len(self.time))
        pb = progressbar(self.trange[-1]-self.trange[0])
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            X, y = df_comp.drop(columns=["time", "y"]).values, df_comp.y.values
            pred_lr = model.predict(X)
            accruacy[idx] = accuracy_score(pred_lr, y)*100
        pb.finish()
        return accruacy

    def fid(self):

        def actuator(set1, set2):
            b1 = binning(set1[:, 0])
            b2 = binning(set2[:, 0])
            bins = np.max([b1.bins(), b2.bins()])

            xr = (np.min([set1[:, 0].min(), set2[:, 0].min()]), np.max([set1[:, 0].max(), set2[:, 0].max()]))
            h1, _ = np.histogram(set1[:, 0], bins=bins, range=xr, density=True)
            h2, _ = np.histogram(set2[:, 0], bins=bins, range=xr, density=True)

            act1, act2 = h1, h2
            for i in range(1, len(set1[0, :])):
                xr = (np.min([set1[:, i].min(), set2[:, i].min()]), np.max([set1[:, i].max(), set2[:, i].max()]))
                h1, _ = np.histogram(set1[:, i], bins=bins, range=xr, density=True)
                h2, _ = np.histogram(set2[:, i], bins=bins, range=xr, density=True)
                act1 = np.vstack((act1, h1))
                act2 = np.vstack((act2, h2))
            return FID(act1, act2)

        fids = np.zeros(len(self.time))
        X0 = self.df_init.drop(columns=["time", "y"]).values
        pb = progressbar(self.trange[-1]-self.trange[0])
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            X = df_comp.drop(columns=["time", "y"]).values
            fids[idx] = actuator(X0, X)
        pb.finish()
        return fids

    def ks_test(self, average=False):
        columns = self.df.drop(columns=["time", "y"]).columns
        dvalue = {"%s_D"%col : np.zeros(len(self.time)) for col in columns}
        pvalue = {"%s_p"%col : np.zeros(len(self.time)) for col in columns}
        pb = progressbar(self.trange[-1]-self.trange[0])
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            for col in columns:
                dvalue["%s_D"%col][idx], pvalue["%s_p"%col][idx] = ks_2samp(self.df_init[col].values, df_comp[col].values)
        pb.finish()
        if average:
            d, p = np.zeros(len(self.time)), np.zeros(len(self.time))
            for _, v in dvalue.items():
                d += v
            for _, v in pvalue.items():
                p += v
            deno = self._file_info["n_cont"]+ self._file_info["n_disc"]
            return d/deno, p/deno
        else:
            return dvalue, pvalue
    
    def psi(self, average=False):
        columns = self.df.drop(columns=["time", "y"]).columns
        psis = {"%s"%col : np.zeros(len(self.time)) for col in columns}
        X0 = self.df_init.drop(columns=["time", "y"])
        pb = progressbar(self.trange[-1]-self.trange[0])
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            X = df_comp.drop(columns=["time", "y"])
            for col in columns:
                psis["%s"%col][idx] = psi(X0[col].values, X[col].values)
        pb.finish()
        if average:
            psi_v = np.zeros(len(self.time))
            for _, v in psis.items():
                psi_v += v
            psi_v /= self._file_info["n_cont"]+ self._file_info["n_disc"]
            return psi_v
        return psis



path = "./dwlib/data/generated_table/"
file_name = "RI55"
ex = experiment(path, file_name, 30)
acc = ex.rf_init()
d, p = ex.ks_test(average=True)
fid = ex.fid()
psi = ex.psi(average=True)
time = Time(ex.time, format="mjd").datetime64

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def nrow_plot(time, acc, data_dict):
    nr = len(data_dict.keys())

    plt.figure(figsize=(15, nr*2+1))
    gs = GridSpec(nrows=nr, ncols=1)
    gs.update(hspace=0, wspace=0)
    axes= [plt.subplot(gs[i]) for i in range(nr)]
    for idx, (k, v) in enumerate(data_dict.items()):
        axes[idx].plot(time, v, label=k)
    drift = ex._file_info["drift_time"]
    for ax in axes:
        axt = plt.twinx(ax)
        axt.plot(time, acc, c='gray', alpha=0.3)
        ax.grid(True)
        ax.set_xlim(time[0], time[-1])
        ax.legend(loc=1)
        ax.vlines(Time(np.array(drift), format="mjd").datetime64, ax.set_ylim()[0], ax.set_ylim()[1], colors='r', linestyles=":")
    plt.show()

nrow_plot(time, acc, {"fid": fid,
                      "D-statistics" : d,
                      "p-value": p,
                      "PSI": psi})