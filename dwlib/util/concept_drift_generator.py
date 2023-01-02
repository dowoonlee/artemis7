import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy.random import gamma, randint
from scipy.signal import savgol_filter
from astropy.time import Time


class concept_drift_generator():
    def __init__(self, time_range, drift_time_sequence, size, n_cont, n_disc):
        

        self.time = np.linspace(Time(time_range[0]).mjd, Time(time_range[1]).mjd, size)
        self._size = size
        self._ncol = (n_cont, n_disc)
        dt = 1e-6
        self._drift_sequence = np.append(Time(drift_time_sequence).mjd, self.time[-1]+dt)
        self._drift_sequence = np.insert(self._drift_sequence, 0, Time(time_range[0]).mjd-dt)

    @staticmethod
    def _generate_cdf(size):
        pdf = np.random.random(size)
        pdf /= np.sum(pdf)
        cdf = np.cumsum(pdf)
        cdf = np.insert(cdf, 0, 0)
        return cdf
    
    def _generate_continuous(self, shape, scale, size=None):
        if size is None:
            size = self._size
        data = gamma(shape, scale, size=size)
        data = savgol_filter(data, int(len(data)/20//2*2)+1, 3)
        return data

    def _generate_continuous_drift(self, shape, scale, size=None):
        if size is None:
            size = self._size

        std = shape*(scale**2)
        amplitude = np.random.normal()*std

        data = np.zeros(size)

        on_drift = False
        for ts, te in zip(self._drift_sequence[:-1], self._drift_sequence[1:]):
            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            if on_drift:
                data[cur_interval] = gamma(shape, scale, size=len(cur_interval)) + amplitude
            else:
                data[cur_interval] = gamma(shape, scale, size=len(cur_interval))
            on_drift = not on_drift
        data = savgol_filter(data, int(len(data)/20//2*2)+1, 3)
        
        return data

    def _generate_discrete(self, sample, size=None):
        if size is None:
            size = self._size

        cdf = self._generate_cdf(len(sample))
        data = np.zeros(size)
        dums = np.random.random(size)

        for idx in range(size):
            pidx = 0
            while cdf[pidx]<dums[idx]:
                pidx += 1
            pidx -= 1
            data[idx] = sample[pidx]
        return data
    
    def _generate_dicrete_drift(self, sample, size=None):
        if size is None:
            size = self._size

        cdf_base = self._generate_cdf(len(sample))
        cdf_drift = self._generate_cdf(len(sample))

        data = np.zeros(size)
        dums = np.random.random(size)

        on_drift = False
        for ts, te in zip(self._drift_sequence[:-1], self._drift_sequence[1:]):
            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            incre_pivot = cur_interval[int(itv_len*0.3)]
            decre_pivot = cur_interval[int(itv_len*0.7)]

            cdf_bef = cdf_base if on_drift else cdf_drift
            cdf_now = cdf_drift if on_drift else cdf_base

            for c, idx in enumerate(cur_interval):
                pidx = 0
                if idx<incre_pivot:
                    bef = (l-c)/(2*l)
                elif idx>decre_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                cdf = cdf_bef*bef + cdf_now*(1-bef)
                while cdf[pidx]<dums[idx]:
                    pidx += 1
                pidx -= 1
                data[idx] = sample[pidx]
            on_drift = not on_drift
        return data.astype(int)
    
    def _generate_real(self, n_cont, n_disc, n_label=2):

        base = {"time" : self.time}
        drift = {}

        for conti in range(n_cont):
            base["C%02d"%conti] = self._generate_continuous(2, 2)
            drift["C%02d"%conti] = self._generate_continuous(4, 4, size=1000)
        for discr in range(n_disc):
            disc_sample_limit = np.max([4, int(gamma(shape=3, scale=3))])
            disc_sample = randint(1, disc_sample_limit, int(disc_sample_limit/2))
            base["D%02d"%discr] = self._generate_discrete(disc_sample)
            drift["D%02d"%discr] = self._generate_discrete(disc_sample, size=1000)
            
        
        base = pd.DataFrame(base)
        drift = pd.DataFrame(drift)

        on_drift = False
        y = np.array([])
        
        X_base = base.drop(columns="time").to_numpy()
        X_drift = drift.to_numpy()
        kms_base = KMeans(n_clusters=n_label, random_state=42).fit(X_base)
        kms_drift = KMeans(n_clusters=n_label, random_state=42).fit(X_drift)
        for ts, te in zip(self._drift_sequence[:-1], self._drift_sequence[1:]):
            X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()
            if on_drift:
                yadd = kms_drift.predict(X)
            else:
                yadd = kms_base.predict(X)
            y = np.concatenate((y, yadd), axis=0)
        base["y"] = y
        return base
    
    def _generate_virtual(self, n_cont, n_disc, n_label=2):
        d = {"time" : self.time}
        for conti in range(n_cont):
            d["C%02d"%conti] = self._generate_continuous_drift(2, 2)
        for discr in range(n_disc):
            disc_sample_limit = np.max([4, int(gamma(shape=3, scale=3))])
            disc_sample = randint(1, disc_sample_limit, int(disc_sample_limit/2))
            d["D%02d"%discr] = self._generate_dicrete_drift(disc_sample)

        d = pd.DataFrame(d)
        X_base = d[d.time<self._drift_sequence[1]].drop(columns="time").to_numpy()
        kms = KMeans(n_clusters=n_label, random_state=42).fit(X_base)
        y = kms.predict(d.drop(columns="time").to_numpy())
        d["y"] = y
        return d

    def generate(self, drift="real", n_label=2, time="mjd"):
        n_cont, n_disc = self._ncol
        
        if drift == "real":
            d = self._generate_real(n_cont, n_disc, n_label)

        elif drift == "virtual":
            d = self._generate_virtual(n_cont, n_disc, n_label)
        
        if time=="datetime64":
            d.time = Time(d.time, format="mjd").datetime64

        return d
