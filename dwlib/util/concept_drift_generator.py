import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from numpy.random import gamma, randint
from astropy.time import Time


class concept_drift_generator():
    def __init__(self, time_range, drift_time_sequence, size, n_cont, n_disc):
        """
        time_range : array-like String. 데이터셋의 시작시간과 끝시간. (start, end)형태로 입력.
        drift_time_sequence : array-like String. drift가 작동하는 시간.
        size : Integer. data point 갯수
        n_cont : 연속데이터 컬럼의 갯수
        n_disc : 불연속데이터 컬럼의 갯수
        """
        self.time = np.linspace(Time(time_range[0]).mjd, Time(time_range[1]).mjd, size)
        self._size = size
        self._ncol = (n_cont, n_disc)
        dt = 1e-6
        self._drift_sequence = np.append(Time(drift_time_sequence).mjd, self.time[-1]+dt)
        self._drift_sequence = np.insert(self._drift_sequence, 0, Time(time_range[0]).mjd-dt)

    @staticmethod
    def _generate_cdf(size):
        """
        주어진 크기의 임의의 CDF (Cumulative Dist Func)을 만드는 함수
        size : CDF의 data point 갯수
        """
        pdf = np.random.random(size)
        pdf /= np.sum(pdf)
        cdf = np.cumsum(pdf)
        cdf = np.insert(cdf, 0, 0)
        return cdf
    
    def _generate_continuous(self, shape, scale, size=None):
        """
        gamma distribution에 따른 난수 생성
        shape : Float. shape parameter k
        scale : Float. scale parameter theta
        size : Integer. data point의 갯수. 미입력 시 self._size로 대체
        """
        if size is None:
            size = self._size
        data = gamma(shape, scale, size=size)
        return data

    def _generate_continuous_drift(self, base_coeff, drift_coeff, size=None):
        """
        Virtual Drift가 존재하는 연속 데이터 생성
        base_coeff : array-like. base-data를 생성하기위한 gamma dist의 (shape, scale)
        dirft_coeff : array-like. drift-data를 생성하기위한 gamma dist의 (shape, scale)
        size : Integer. data point의 갯수. 미입력 시 self._size로 대체
        """
        if size is None:
            size = self._size

        data = np.zeros(size)

        on_drift = False
        for ts, te in zip(self._drift_sequence[:-1], self._drift_sequence[1:]):
            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            if on_drift:
                data[cur_interval] = gamma(drift_coeff[0], drift_coeff[1], size=len(cur_interval))
            else:
                data[cur_interval] = gamma(base_coeff[0], base_coeff[1], size=len(cur_interval))
            on_drift = not on_drift
        return data

    def _generate_discrete(self, sample, size=None):
        """
        불연속 데이터 생성
        sample : array-like. 불연속 데이터의 unique array.
        size : Integer. data point의 갯수. 미입력 시 self._size로 대체
        """
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
        """
        불연속 데이터 생성
        sample : array-like. 불연속 데이터의 unique array.
        size : Integer. data point의 갯수. 미입력 시 self._size로 대체
        """
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
        """
        Real Drift가 존재하는 column 생성
        n_cont : 연속데이터 컬럼의 갯수
        n_disc : 불연속데이터 컬럼의 갯수
        n_label : classification/regression 을 위한 label의 갯수
        """

        base = {"time" : self.time}
        drift = {}
        for conti in range(n_cont):
            base_coeff = np.random.rand(2) * 5 + 1
            drift_coeff = np.random.rand(2)* 3 + 6
            base["C%02d"%conti] = self._generate_continuous(base_coeff[0], base_coeff[1])
            drift["C%02d"%conti] = self._generate_continuous(drift_coeff[0], drift_coeff[1], size=1000)
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
        """
        Virtual Drift가 존재하는 column 생성
        n_cont : 연속데이터 컬럼의 갯수
        n_disc : 불연속데이터 컬럼의 갯수
        n_label : classification/regression 을 위한 label의 갯수
        """
        d = {"time" : self.time}
        for conti in range(n_cont):
            base_coeff = np.random.rand(2) * 5 + 1
            drift_coeff = np.random.rand(2)* 3 + 6
            d["C%02d"%conti] = self._generate_continuous_drift(base_coeff, drift_coeff)
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
        """
        Drift Generator
        drift : ["real", "virtual"]. drift의 종류 설정
        n_label : classification/regression 을 위한 label의 갯수
        time : ["mjd", "datetime64"]. output의 time형식
        """
        n_cont, n_disc = self._ncol
        
        if drift == "real":
            d = self._generate_real(n_cont, n_disc, n_label)

        elif drift == "virtual":
            d = self._generate_virtual(n_cont, n_disc, n_label)
        
        if time=="datetime64":
            d.time = Time(d.time, format="mjd").datetime64

        return d
