import numpy as np
import pandas as pd
from numpy.random import choice, rand, gamma
from astropy.time import Time


"""
Concept drift generator (CDG)


1. Initialization

Between (time_range[0]~time_range[1]), Drifts take place at 'drfit_time_sequence'.
'time_range' and 'drift_time_sequence' support float (mjd) and string (datetime64)

Currently (2023-01-06) CDG only support continuous columns and 2type drift mode (base-drift).
Discrete columns and many-type drift mode are under developement.


2. Generation
The X (Data set) is randomly sampled with Gamma distribution (https://en.wikipedia.org/wiki/Gamma_distribution).
Gamma distribution has two parameters (k, theta). The parameters follow normal distribution.

generate() methods supports 2 types of drift modes [real/virtual] and 3types of drift patterns [sudden/incremental/gradual].
In case of drift mode, "real" is the drift in p(y|X) which is the case that X doesn't change but label y change only.
For "virtual", p(y|X) doesn't change but the distribution of X changes.
you can control the strength of the drift with parameter strength. 'strength' works linearly in "real" and logarithmic in "virtual".
Drift pattern depends on the time interval of [drift_start, drift_end]. "sudden" make drift with time interval=0. "Incremental" and "gradual" have same time interval of [0.3*dT_bef, 0.3*dT_aft]. The difference between them is that "incremental" has gentle drift in time interval and "gradual" has abrupting dirft.



"""

class concept_drift_generator():
    def __init__(self, time_range, drift_time_sequence, n_drift_type, n_cont, n_disc, **kwargs):
        """
        time_range : array-like. 데이터셋의 시작시간과 끝시간. (start, end)형태로 입력.
        drift_time_sequence : array-like. drift가 작동하는 시간.
        n_drift_type : Integer. drift 종류의 갯수(현재 2 (base/drift)까지만 지원)
        size : Integer. data point 갯수
        dt : Float. data point의 간격
        n_cont : 연속데이터 컬럼의 갯수
        n_disc : 불연속데이터 컬럼의 갯수 (under developement)
        """

        if not time_range[0].isdigit():
            time_range = np.sort(Time(time_range).mjd)
        if not drift_time_sequence[0].isdigit():
            drift_time_sequence = np.sort(Time(drift_time_sequence).mjd)
        keys = kwargs.keys()

        if "size" in keys:
            self._size = kwargs["size"]
            self.time = np.linspace(time_range[0], time_range[1], self._size)
        elif "dt" in keys:
            self.time = np.arange(time_range[0], time_range[1], kwargs["dt"])
            self._size = len(self.time)

        if n_cont + n_disc<=0:
            raise ValueError("At least 1 column should be exist")

        self._ncol = (n_cont, n_disc)
        dt = 1e-6
        self._drift_sequence = np.concatenate((drift_time_sequence, [time_range[1]+dt , time_range[0]-dt]), axis=0)
        self._drift_sequence = np.sort(self._drift_sequence)
        self.n_drift_type = np.min([n_drift_type, len(drift_time_sequence), 2])
        self.noise = 0

    @staticmethod
    def _classfication(X, att_cols, threshold):
        X = X[:, att_cols]
        sumX = np.sum(X, axis=1)
        return (sumX>threshold).astype(np.int32)

    @staticmethod
    def _discrete_sampling(sample, pdf, size):
        if len(sample) != len(pdf):
            raise ValueError("The sample and PDF have different size")
        pdf /= np.sum(pdf)
        cdf = np.cumsum(pdf)
        cdf = np.insert(cdf, 0, 0)

        data = np.zeros(size)
        dums = np.random.random(size)

        for idx in range(size):
            pidx = 0
            while cdf[pidx]<dums[idx]:
                pidx += 1
            pidx -= 1
            data[idx] = sample[pidx]
        return data
    
    def _add_noise_cont(self, base_col):
        pos = np.random.random(self._size)
        pos = np.where(pos<=self.noise)[0]
        base_col[pos] = base_col[pos]*np.random.normal(loc=1, size=len(pos))
        return base_col

    def _real_sudden(self, base, thresholds, att_cols):
        y = np.array([])
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
                X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()
                threshold = thresholds[idx%2]
                yadd = self._classfication(X = X,
                                        att_cols = att_cols,
                                        threshold = threshold)
                y = np.concatenate((y,yadd),axis=0)
        base["y"] = y
        return base
    
    def _real_incremental(self, base, thresholds, att_cols):
        y = np.array([])
        on_drift = False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()

            threshold_bef = thresholds[0] if on_drift else thresholds[1]
            threshold_now = thresholds[1] if on_drift else thresholds[0]

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, idx in enumerate(cur_interval):
                if idx<pre_pivot:
                    bef = (l-c)/(2*l)
                elif idx>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                X_sub = X[c, :].reshape(1, -1)
                threshold = threshold_bef*bef + threshold_now*(1-bef)
                yadd = self._classfication(X = X_sub,
                                    att_cols = att_cols,
                                    threshold = threshold)
                y = np.concatenate((y,yadd),axis=0)
            on_drift = not on_drift
        base["y"] = y
        return base
    
    def _real_gradual(self, base, thresholds, att_cols):
        y = np.array([])
        on_drift = False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()

            threshold_bef = thresholds[0] if on_drift else thresholds[1]
            threshold_now = thresholds[1] if on_drift else thresholds[0]

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, idx in enumerate(cur_interval):
                if idx<pre_pivot:
                    bef = (l-c)/(2*l)
                elif idx>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                X_sub = X[c, :].reshape(1, -1)
                threshold = threshold_bef if np.random.random()<=bef else threshold_now
                yadd = self._classfication(X = X_sub,
                                    att_cols = att_cols,
                                    threshold = threshold)
                y = np.concatenate((y,yadd),axis=0)
            on_drift = not on_drift
        base["y"] = y
        return base
    
    def _generate_real(self, n_cont, n_disc, strength, drift_type="sudden"):
        base_coeff = rand(n_cont, 2)*10 #G[[k, theta]*n_cont]

        base = {"time" : self.time}
        attribute_cols = choice(np.arange(n_cont), n_cont//2, replace=False)
        self.att_cols = attribute_cols
        dumx = np.zeros(self._size) #rename
        for idx, (k, theta) in enumerate(base_coeff):
            gamma_dist = gamma(k, theta, self._size)
            base["C%02d"%idx] = gamma_dist
            if idx in attribute_cols:
                dumx += gamma_dist
        gamma_dist /= n_cont
        att_mean, att_std = np.mean(dumx), np.std(dumx)
        thresholds = [att_mean+att_std*strength/20, att_mean-att_std*strength/20]
        base = pd.DataFrame(base)

        if drift_type == "sudden":
            df = self._real_sudden(base = base,
            thresholds=thresholds,
            att_cols=attribute_cols)
            
        elif drift_type == "incremental":
            df = self._real_incremental(base = base,
            thresholds=thresholds,
            att_cols=attribute_cols)
        
        elif drift_type == "gradual":
            df = self._real_gradual(base = base,
            thresholds=thresholds,
            att_cols=attribute_cols)
        else:
            raise KeyError("There is no such type of drift [%s]"%drift_type)
        return df
    
    def _virtual_sudden(self, base, drift_coeff):
        on_drift=False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            cond = (base.time>ts) & (base.time<=te)
            if on_drift:
                for idx, (k, theta) in enumerate(drift_coeff):
                    temp_g = gamma(k, theta, np.sum(cond.to_numpy()))
                    base["C%02d"%idx][cond] = temp_g
            on_drift = not on_drift
        return base
    
    def _virtual_incremental(self, base, base_coeff, drift_coeff):
        on_drift=False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff_bef = base_coeff if on_drift else drift_coeff
            coeff_now = drift_coeff if on_drift else base_coeff

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, idx in enumerate(cur_interval):
                if idx<pre_pivot:
                    bef = (l-c)/(2*l)
                elif idx>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0                    
                coeffs = coeff_bef*bef + coeff_now*(1-bef)
                X_sub = np.array([gamma(k, theta) for k, theta in coeffs])
                
                for i, x in enumerate(X_sub):
                    base["C%02d"%i][idx] = x
            on_drift = not on_drift
        return base
    
    def _virtual_gradual(self, base, base_coeff, drift_coeff):
        on_drift=False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff_bef = base_coeff if on_drift else drift_coeff
            coeff_now = drift_coeff if on_drift else base_coeff

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, idx in enumerate(cur_interval):
                if idx<pre_pivot:
                    bef = (l-c)/(2*l)
                elif idx>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                coeffs = coeff_bef if np.random.random()<=bef else coeff_now
                X_sub = np.array([gamma(k, theta) for k, theta in coeffs])
                
                for i, x in enumerate(X_sub):
                    base["C%02d"%i][idx] = x
            on_drift = not on_drift
        return base
    
    def _generate_virtual(self, n_cont, n_disc, strength, drift_type="sudden"):
        base_coeff = rand(n_cont, 2)*10 #G[[k, theta]*n_cont]
        a = np.log2(strength)
        drift_coeff = np.array([[k -(k*a)/(t+a), t+a] for (k, t) in base_coeff])

        base = {"time" : self.time}
        attribute_cols = choice(np.arange(n_cont), n_cont//2, replace=False)
        dumx = np.zeros(self._size) #rename

        for idx, (k, theta) in enumerate(base_coeff):
            gamma_dist = gamma(k, theta, self._size)
            base["C%02d"%idx] = gamma_dist
            if idx in attribute_cols:
                dumx += gamma_dist
        gamma_dist /= n_cont
        att_mean, att_std = np.mean(dumx), np.std(dumx)
        base = pd.DataFrame(base)

        threshold = att_mean + att_std/2

        if drift_type == "sudden":
            df = self._virtual_sudden(base=base, drift_coeff=drift_coeff)

        elif drift_type == "incremental":
            df = self._virtual_incremental(base=base, base_coeff=base_coeff, drift_coeff=drift_coeff)

        elif drift_type == "gradual":
            df = self._virtual_gradual(base=base, base_coeff=base_coeff, drift_coeff=drift_coeff)
        else:
            raise KeyError("There is no such type of drift [%s]"%drift_type)
        
        y = self._classfication(X = df.drop(columns="time").to_numpy(),
                                att_cols = attribute_cols,
                                threshold = threshold)
        df["y"] = y

        return df

    def generate(self, drift="real", time="mjd", drift_type="sudden", strength=10, noise=0.1):
        """
        Drift Generator
        drift : ["real", "virtual"]. drift의 종류 설정
        time : ["mjd", "datetime64"]. output의 time형식
        drift_Type : ["sudden", "incremental", "gradual"]. drift 발생 유형 설정
        strength : drift의 세기. default=10
        noise : data의 noise 비율
        """
        n_cont, n_disc = self._ncol
        
        if drift == "real":
            d = self._generate_real(n_cont, n_disc, strength, drift_type)

        elif drift == "virtual":
            if drift_type == "incremental":
                print("Generating incremtanl virtual drift data may take a long time")
            d = self._generate_virtual(n_cont, n_disc, strength, drift_type)
        
        if time=="datetime64":
            d.time = Time(np.round(d.time.to_numpy(), 3), format="mjd").datetime64
        
        self.noise = noise
        if self.noise>0:
            for col in range(self._ncol[0]):
                d["C%02d"%col] = self._add_noise_cont(d["C%02d"%col].to_numpy())
        return d
