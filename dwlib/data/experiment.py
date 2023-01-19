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
from dwlib.stats.population_stability_index import population_stability_index as PSI
from scipy.stats import wasserstein_distance


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

class experiment():
    def __init__(self, path, file_name, dt):
        with open(path+"%s.json"%file_name, "r") as f:
            file_info = json.load(f)
        self._file_info = file_info
        self.df = pd.read_csv(path+"%s.csv"%file_info["file"], index_col=0)
        self.dt = dt
        self.drop_cols = ["time", "y", "drift"]

        r_init = 0.7
        t_init = r_init*self._file_info["drift_time"][0] + (1-r_init) * self._file_info["time_range"][0]
        self.df_init = self.df[self.df.time<t_init]
        self.trange = np.arange(int(t_init), int(self.df.time.max()-dt))
        self.time = np.array([self.df[(self.df.time>=t) & (self.df.time<t+self.dt)].time.mean() for t in self.trange])          
        
        self.drift = np.array([self.df[(self.df.time>=t) & (self.df.time<t+self.dt)].drift.mean() for t in self.trange]) 
        self.Tdrift = np.zeros(len(self._file_info["drift_time"]))
        for idx, Tdrift in enumerate(self._file_info["drift_time"]):
            absdt = abs(self.time-Tdrift)
            self.Tdrift[idx] = np.where(absdt==np.min(absdt))[0]
        self.Tdrift = self.Tdrift.astype(int)


    def evaluate(self):
        df_init = self.df_init.drop(columns=["time", "drift"])
        X0 = df_init.loc[:, df_init.columns != "y"].values
        y0 = df_init.y.values
        X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.2, random_state=42)
        pb = progressbar(self.trange[-1]-self.trange[0])
        pb.start()

        if self._file_info["prob_type"] == "classification":
            model = RandomForestClassifier(max_depth=6)
            model.fit(X_train,y_train)
            accuracy = np.zeros(len(self.time))
            for idx, t in enumerate(self.trange):
                pb.update(idx)
                df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
                X, y = df_comp.drop(columns=self.drop_cols).values, df_comp.y.values
                pred_lr = model.predict(X)
                accuracy[idx] = accuracy_score(pred_lr, y)*100
            pb.finish()
            self.accuracy = accuracy
        
        else:
            lr = LinearRegression().fit(X_train, y_train)
            score = np.zeros(len(self.time))
            for idx, t in enumerate(self.trange):
                pb.update(idx)
                df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
                X, y = df_comp.drop(columns=self.drop_cols).values, df_comp.y.values
                score[idx] = lr.score(X, y)
            pb.finish()
            self.score = score


    @staticmethod
    def _actuator(set1, set2):
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

    def fid(self):

        fids = np.zeros(len(self.time))
        X0 = self.df_init.drop(columns=self.drop_cols).values
        pb = progressbar(self.trange[-1]-self.trange[0], "FID")
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            X = df_comp.drop(columns=self.drop_cols).values
            fids[idx] = self._actuator(X0, X)
        pb.finish()

        return fids
    
    def wd(self):
        columns = self.df.drop(columns=self.drop_cols).columns
        wds = {col : np.zeros(len(self.time)) for col in columns}
        m, s = {}, {}
        for col in columns:
            m[col], s[col] = self.df[col].mean(), self.df[col].std()
        pb = progressbar(self.trange[-1]-self.trange[0], "WD")
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            for col in columns:
                di = (self.df_init[col].values-m[col])/s[col]
                dc = (df_comp[col].values - m[col])/s[col]
                wds[col][idx] = wasserstein_distance(di, dc)
        pb.finish()
        return wds

    def ks_test(self, average=False):
        columns = self.df.drop(columns=self.drop_cols).columns
        dvalue = {col : np.zeros(len(self.time)) for col in columns}
        pvalue = {col : np.zeros(len(self.time)) for col in columns}
        pb = progressbar(self.trange[-1]-self.trange[0], "KS")
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            for col in columns:
                dvalue[col][idx], pvalue[col][idx] = ks_2samp(self.df_init[col].values, df_comp[col].values)
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
        columns = self.df.drop(columns=self.drop_cols).columns
        psis = {"%s"%col : np.zeros(len(self.time)) for col in columns}
        X0 = self.df_init.drop(columns=self.drop_cols)
        pb = progressbar(self.trange[-1]-self.trange[0], "PSI")
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = self.df[(self.df.time>=t) & (self.df.time<t+self.dt)]
            X = df_comp.drop(columns=self.drop_cols)
            for col in columns:
                psis["%s"%col][idx] = PSI(X0[col].values, X[col].values)        
        pb.finish()
        if average:
            psi_v = np.zeros(len(self.time))
            for _, v in psis.items():
                psi_v += v
            psi_v /= self._file_info["n_cont"]+ self._file_info["n_disc"]
            return psi_v
        else:
            return psis

# import matplotlib.pyplot as plt
# path = "./dwlib/data/generated_table/"
# file_name = "RI55"
# ex = experiment(path=path, file_name=file_name, dt=30)
# d, p = ex.ks_test(average=True)
# ds, ps = ex.ks_test()
# fid = ex.fid()
# psi = ex.psi(average=True)
# psis = ex.psi()
# wds = ex.wd()
# time = Time(ex.time, format="mjd").datetime64


