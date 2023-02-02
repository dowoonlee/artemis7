import numpy as np
import pandas as pd
from numpy.random import choice, rand, gamma, randint
from astropy.time import Time
from datetime import datetime
import json
import sys, os
sys.path.append("/".join(os.path.abspath(__file__).split("\\")[:-3]))
from datagenerator.util.progressbar import progressbar

"""
Concept drift generator (CDG)


1. Initialization

Between (time_range[0]~time_range[1]), Drifts take place at 'drfit_time_sequence'.
'time_range' and 'drift_time_sequence' support float (mjd) and string (datetime64, yyyy-mm-dd)

CDG only support continuous columns and 2type drift mode (base<->drift) (2023-01-06).
CDG can support continous and discrete columns (2023-01-12)
CDG supports regression prob type (2023-01-19)

2. Generation
X (Data set) is randomly sampled with Gamma distribution (https://en.wikipedia.org/wiki/Gamma_distribution).
Gamma distribution has two parameters (k, theta). The parameters follow normal distribution in this class.

generate() methods supports 2 types of drift modes [real/virtual] and 3types of drift patterns [sudden/incremental/gradual].
In case of drift mode, "real" is the drift in p(y|X) which is the case that X doesn't change but label y change only.
For "virtual", p(y|X) doesn't change but the distribution of X changes.
you can control the strength of the drift with parameter strength. 'strength' works linearly in "real" and logarithmic in "virtual".
Drift pattern depends on the time interval of [drift_start, drift_end].
"sudden" make drift with time interval=0.
"Incremental" and "gradual" have same time interval of [0.3*dT_bef, 0.3*dT_aft].
The difference between them is that "incremental" has gentle drift in time interval and "gradual" has abrupting dirft.



"""

class concept_drift_generator():
    def __init__(self, time_range, drift_time_sequence, prob_type, n_cont, n_disc, **kwargs):
        """
        time_range : array-like. 데이터셋의 시작시간과 끝시간. (start, end)형태로 입력.
        drift_time_sequence : array-like. drift가 작동하는 시간.
        prob_type : [classification / regression] y의 type
        n_cont : 연속데이터 컬럼의 갯수
        n_disc : 불연속데이터 컬럼의 갯수
        size : Integer. data point 갯수
        dt : Float. data point의 간격
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
        else:
            raise AssertionError("Absence in time binning")
        
        self._drift_weight = np.zeros(len(self.time))
        
        self._ncol = (n_cont, n_disc)
        self.n_att = kwargs["n_att"] if "n_att" in keys else sum(self._ncol)//2

        if sum(self._ncol)<=0:
            raise ValueError("At least 1 column should be exist")

        tol = 1e-6
        self._drift_sequence = np.sort(np.concatenate((drift_time_sequence, [time_range[1]+tol , time_range[0]-tol]), axis=0))
        self.prob_type = prob_type
        yfunction = {
            "classification" : self._classfication,
            "regression" : self._regression}
        self._label_function = yfunction[self.prob_type]
        self.noise = 0
        self.attribute_columns = []
        self.df = None
        self.df_info = {
            "file" : None,
            "prob_type" : self.prob_type,
            "date_generated" : None,
            "time_range" : time_range,
            "drift_time" : drift_time_sequence,
            "n_cont" : n_cont,
            "n_disc" : n_disc,
            "drift": None,
            "drift_type" : None,
            "strength" : 0,
            "noise" : 0,
            "att_cols": self.attribute_columns
        }

    @staticmethod
    def _classfication(X, att_cols, threshold):
        X = X[:, att_cols]
        sumX = np.sum(X, axis=1)
        return (sumX>threshold).astype(np.int32)

    @staticmethod
    def _regression(X, att_cols, threshold):
        X = X[:, att_cols]
        sumX = np.sum(X, axis=1)
        return sumX+threshold
    
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
        data = data.astype(int)
        return data[0] if size==1 else data

    @staticmethod
    def hist2d(df0, df1, dropcol):
        v0 = df0.drop(columns=dropcol).to_numpy()
        v1 = df1.drop(columns=dropcol).to_numpy()
        
        b0 = int(1+3.322*np.log2(len(v0[:, 0])))
        b1 = int(1+3.322*np.log2(len(v1[:, 0])))
        bins = np.max([b0, b1])

        xr = (np.min([v0[:, 0].min(), v1[:, 0].min()]), np.max([v0[:, 0].max(), v1[:, 0].max()]))
        a0, _ = np.histogram(v0[:, 0], bins=bins, range=xr, density=True)
        a1, _ = np.histogram(v1[:, 0], bins=bins, range=xr, density=True)
        for i in range(1, len(v0[0, :])):
            xr = (np.min([v0[:, i].min(), v1[:, i].min()]), np.max([v0[:, i].max(), v1[:, i].max()]))
            h0, _ = np.histogram(v0[:, i], bins=bins, range=xr, density=True)
            h1, _ = np.histogram(v1[:, i], bins=bins, range=xr, density=True)
            a0 = np.vstack((a0, h0))
            a1 = np.vstack((a1, h1))
        return np.sum(np.abs(a0-a1))/bins/len(a0)
    
    @staticmethod
    def read_info(path, file_name):
        with open(path+"%s.json"%file_name, "r") as f:
            file_info = json.load(f)
        return file_info
    
    def _sanity_check(self, df):
        dropcol = ["y", "time"]
        not_ok = False
        d0 = df[df.time<=(self._drift_sequence[0]+ self._drift_sequence[1])/2]
        d1 = df[(df.time>(self._drift_sequence[0]+self._drift_sequence[1])/2) &(df.time<=self._drift_sequence[1])]
        score0 = self.hist2d(d0, d1, dropcol)
        on_drift = False
        for ts, te in zip(self._drift_sequence[:-1], self._drift_sequence[1:]):
            d_now = df[(df.time>ts) & (df.time<te)]
            score = self.hist2d(d0, d_now, dropcol)

            not_ok = on_drift and (score0>score)
            if not_ok:
                return True        
        return not_ok

    def _add_noise_cont(self, base_col):
        pos = np.random.random(self._size)
        pos = np.where(pos<=self.noise)[0]
        base_col[pos] = base_col[pos]*np.random.normal(loc=1, size=len(pos))
        return base_col
    
    def _add_noise_disc(self, base_col):
        pos = np.random.random(self._size)
        pos = np.where(pos<=self.noise)[0]
        uniq = np.unique(base_col)
        for p in pos:
            base_col[p] = np.random.choice(uniq)
        return base_col


    def _real_sudden(self, base, thresholds, att_cols):
        y = np.array([])
        pb = progressbar(len(self._drift_sequence)-1, "CDG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            pb.update(idx)
            X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()
            threshold = thresholds[idx%2]
            yadd = self._label_function(X = X,
                                    att_cols = att_cols,
                                    threshold = threshold)
            y = np.concatenate((y,yadd),axis=0)
            self._drift_weight[np.where((self.time>ts)&(self.time<=te))[0]] = idx%2
        pb.finish()
        base["drift"] = self._drift_weight
        base["y"] = y
        return base
    
    def _real_incremental(self, base, thresholds, att_cols):
        y = np.array([])
        on_drift = False
        pb = progressbar(len(base.time.values), "CDG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()

            threshold_bef = thresholds[0] if on_drift else thresholds[1]
            threshold_now = thresholds[1] if on_drift else thresholds[0]

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, i in enumerate(cur_interval):
                pb.update(i)
                if i<pre_pivot and ts != self._drift_sequence[0]:
                    bef = (l-c)/(2*l)
                elif i>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                X_sub = X[c, :].reshape(1, -1)
                threshold = threshold_bef*bef + threshold_now*(1-bef)
                yadd = self._label_function(X = X_sub,
                                    att_cols = att_cols,
                                    threshold = threshold)
                y = np.concatenate((y,yadd),axis=0)
                self._drift_weight[i] = 1-bef if on_drift else bef
            on_drift = not on_drift
        pb.finish()
        base["drift"] = self._drift_weight
        base["y"] = y
        return base
    
    def _real_gradual(self, base, thresholds, att_cols):
        y = np.array([])
        on_drift = False
        pb = progressbar(len(base.time.values), "CDG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            X = base[(base.time>ts) & (base.time<=te)].drop(columns="time").to_numpy()

            threshold_bef = thresholds[0] if on_drift else thresholds[1]
            threshold_now = thresholds[1] if on_drift else thresholds[0]

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, i in enumerate(cur_interval):
                pb.update(i)
                if i<pre_pivot and ts != self._drift_sequence[0]:
                    bef = (l-c)/(2*l)
                elif i>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                X_sub = X[c, :].reshape(1, -1)
                cho = np.random.random()<=bef
                threshold = threshold_bef if cho else threshold_now
                yadd = self._label_function(X = X_sub,
                                    att_cols = att_cols,
                                    threshold = threshold)
                y = np.concatenate((y,yadd),axis=0)
                self._drift_weight[i] = 1-int(cho) if on_drift else int(cho)
            on_drift = not on_drift
        pb.finish()
        base["drift"] = self._drift_weight
        base["y"] = y
        return base
    
    def _generate_real(self, ncol, strength, drift_type="sudden"):
        n_cont, n_disc = map(int, ncol)
        base_coeff = rand(n_cont, 2)*10 #G[[k, theta]*n_cont]

        nsamples = randint(6, 30, n_disc)
        samples = [choice(np.arange(nsamples[i]), nsamples[i]//2, replace=False) for i in range(n_disc)]
        base_pdf = [rand(len(samples[i])) for i in range(n_disc)]

        base = {"time" : self.time}
        attribute_cols = choice(np.arange(np.sum(ncol)), self.n_att, replace=False)

        self.attribute_columns = np.sort(attribute_cols)
        dumx = np.zeros(self._size) #rename
        for idx, (k, theta) in enumerate(base_coeff):
            gamma_dist = gamma(k, theta, self._size)
            base["C%02d"%idx] = gamma_dist
            if idx in attribute_cols:
                dumx += gamma_dist

        for idx in range(n_disc):
            sample = self._discrete_sampling(samples[idx], base_pdf[idx], self._size)
            base["D%02d"%idx] = sample
            if int(idx+n_cont) in attribute_cols:
                dumx += sample

        gamma_dist /= len(attribute_cols)
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
    
    def _virtual_sudden(self, base, drift_coeff, samples, base_pdf, drift_pdf):
        on_drift=False
        pb = progressbar(len(self._drift_sequence)-1, "CDG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            pb.update(idx)
            cond = (base.time>ts) & (base.time<=te)
            sub_size = np.sum(cond.to_numpy())
            if on_drift:
                for i, (k, theta) in enumerate(drift_coeff):
                    temp_g = gamma(k, theta, sub_size)
                    base.loc[cond, "C%02d"%i] = temp_g
                
                for i in range(len(samples)):
                    sample, pdf = samples[i], drift_pdf[i]
                    temp_d = self._discrete_sampling(sample, pdf, sub_size)
                    base.loc[cond, "D%02d"%i] = temp_d

            self._drift_weight[np.where((self.time>ts)&(self.time<=te))[0]] = idx%2
            on_drift = not on_drift
        pb.finish()
        return base
    
    def _virtual_incremental(self, base, base_coeff, drift_coeff, samples, base_pdf, drift_pdf):
        on_drift=False
        pb = progressbar(len(base.time.values), "CDG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff_now = base_coeff if on_drift else drift_coeff
            coeff_bef = drift_coeff if on_drift else base_coeff

            pdf_now = base_pdf if on_drift else drift_pdf
            pdf_bef = drift_pdf if on_drift else base_pdf

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, i in enumerate(cur_interval):
                pb.update(i)
                if i<pre_pivot and ts != self._drift_sequence[0]:
                    bef = (l-c)/(2*l)
                elif i>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0                    
                coeffs = coeff_bef*bef + coeff_now*(1-bef)
                X_sub = np.array([gamma(k, theta) for k, theta in coeffs])
                for j, x in enumerate(X_sub):
                    base.loc[i, "C%02d"%j] = x

                pdfs_ = [pdf_bef[i] * bef + pdf_now[i] * (1-bef) for i in range(len(pdf_bef))]
                X_sub = np.array([
                    self._discrete_sampling(sample, pdf, 1) for sample, pdf in zip(samples, pdfs_)
                    ])
                for j, x in enumerate(X_sub):
                    base.loc[i, "D%02d"%j] = x

                self._drift_weight[i] = 1-bef if on_drift else bef
                
            on_drift = not on_drift
        pb.finish()
        return base
    
    def _virtual_gradual(self, base, base_coeff, drift_coeff, samples, base_pdf, drift_pdf):
        on_drift=False
        pb = progressbar(len(base.time.values), "CDG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff_now = base_coeff if on_drift else drift_coeff
            coeff_bef = drift_coeff if on_drift else base_coeff

            pdf_now = base_pdf if on_drift else drift_pdf
            pdf_bef = drift_pdf if on_drift else base_pdf

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(0.3*itv_len)
            pre_pivot = cur_interval[int(itv_len*0.3)]
            suf_pivot = cur_interval[int(itv_len*0.7)]

            for c, i in enumerate(cur_interval):
                pb.update(i)
                if i<pre_pivot and ts != self._drift_sequence[0]:
                    bef = (l-c)/(2*l)
                elif i>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                
                cho = np.random.random()<=bef
                coeffs = coeff_bef if cho else coeff_now
                X_sub = np.array([gamma(k, theta) for k, theta in coeffs])
                for j, x in enumerate(X_sub):
                    base.loc[i, "C%02d"%j] = x

                pdfs_ = [pdf_bef[i]*bef + pdf_now[i]*(1-bef) for i in range(len(pdf_bef))]
                X_sub = np.array([
                    self._discrete_sampling(sample, pdf, 1) for sample, pdf in zip(samples, pdfs_)
                    ])
                for j, x in enumerate(X_sub):
                    base.loc[i, "D%02d"%j] = x

                self._drift_weight[i] = 1-int(cho) if on_drift else int(cho)

            on_drift = not on_drift
        pb.finish()
        return base
    
    def _generate_virtual(self, ncol, strength, drift_type="sudden"):
        n_cont, n_disc = map(int, ncol)
        base_coeff = rand(n_cont, 2)*10 #G[[k, theta]*n_cont]
        a = np.log2(strength)
        drift_coeff = np.array([[k, t+a] for (k, t) in base_coeff])

        base = {"time" : self.time}
        dumx = np.zeros(self._size) #rename
        attribute_cols = choice(np.arange(sum(ncol)), self.n_att, replace=False)
        self.attribute_columns = np.sort(attribute_cols)

        ### generate base continous columns
        for idx, (k, theta) in enumerate(base_coeff):
            gamma_dist = gamma(k, theta, self._size)
            base["C%02d"%idx] = gamma_dist
            if idx in attribute_cols:
                dumx += gamma_dist
        
        ### generate base discrete columns
        nsamples = randint(6, 30, n_disc)
        samples = [choice(np.arange(nsamples[i]), nsamples[i]//2, replace=False) for i in range(n_disc)]
        base_pdf = [rand(len(samples[i])) for i in range(n_disc)]
        drift_pdf = [rand(len(samples[i])) for i in range(n_disc)]
        for idx in range(n_disc):
            sample = self._discrete_sampling(samples[idx], base_pdf[idx], self._size)
            base["D%02d"%idx] = sample
            if int(idx+n_cont) in attribute_cols:
                dumx += sample
        gamma_dist /= len(attribute_cols)

        att_mean, att_std = np.mean(dumx), np.std(dumx)
        base = pd.DataFrame(base)

        threshold = att_mean + att_std/2

        if drift_type == "sudden":
            df = self._virtual_sudden(base=base, drift_coeff=drift_coeff,
            samples=samples, base_pdf = base_pdf, drift_pdf = drift_pdf)

        elif drift_type == "incremental":
            df = self._virtual_incremental(base=base, base_coeff=base_coeff, drift_coeff=drift_coeff,
            samples=samples, base_pdf = base_pdf, drift_pdf = drift_pdf)

        elif drift_type == "gradual":
            df = self._virtual_gradual(base=base, base_coeff=base_coeff, drift_coeff=drift_coeff,
            samples=samples, base_pdf = base_pdf, drift_pdf = drift_pdf)
        else:
            raise KeyError("There is no such type of drift [%s]"%drift_type)
        
        y = self._label_function(X = df.drop(columns="time").to_numpy(),
                                att_cols = attribute_cols,
                                threshold = threshold)
        df["y"] = y
        df["drift"] = self._drift_weight

        return df

    def generate(self, drift="real", time_format="mjd", drift_type="sudden", strength=10, noise=0.1):
        """
        Drift Generator
        drift : ["real", "virtual"]. drift의 종류 설정
        time : ["mjd", "datetime64"]. output의 time형식
        drift_Type : ["sudden", "incremental", "gradual"]. drift 발생 유형 설정
        strength : drift의 세기. default=10
        noise : data의 noise 비율
        """
        sanity = True
        while sanity:
            if drift == "real":
                df = self._generate_real(self._ncol, strength, drift_type)

            elif drift == "virtual":
                if drift_type != "sudden":
                    print("Generating virtual drift data may take a long time")
                df = self._generate_virtual(self._ncol, strength, drift_type)

            sanity = self._sanity_check(df)
            
        if time_format=="datetime64":
            df.time = Time(np.round(df.time.to_numpy(), 3), format="mjd").datetime64
        
        self.noise = noise
        if self.noise>0:
            for col in range(self._ncol[0]):
                df["C%02d"%col] = self._add_noise_cont(df["C%02d"%col].to_numpy())
            for col in range(self._ncol[1]):
                df["D%02d"%col] = self._add_noise_disc(df["D%02d"%col].to_numpy())
        
        self.df_info["strength"] = strength
        self.df_info["noise"] = noise
        self.df_info["drift"] = drift
        self.df_info["drift_type"] = drift_type
        self.df_info["att_cols"] = self.attribute_columns

        self.df = df.astype({"y":int})
        return self.df
    
    def save_to_csv(self, path, file_name = None, time_format="mjd", file_info=True):
        """
        path : csv를 저장할 디렉토리
        file_name : file의 이름. 미지정시 작성일자로 자동지정
        time_format : [mjd, datetime64, iso]
        file_info : df의 상세내용을 json으로 저장
        """
        print("Now saving...")
        if self.df is None:
            raise FileNotFoundError("Data yet been generated")
        if file_name is None:
            tp = self.df_info["prob_type"][0]+self.df_info["drift"][0] + self.df_info["drift_type"][0]
            file_name = "%s_%s"%(tp.upper(), datetime.now().strftime("%Y%m%d%H%M%S"))

        self.df.to_csv(path+"/%s.csv"%file_name)

        if file_info:
            self.df_info["prob_type"] = self.prob_type
            if time_format == "datetime64":
                self.df_info["time_range"] = str(Time(self.df_info["time_range"]).datetime64).tolist()
                self.df_info["drift_time"] = str(Time(self.df_info["drift_time"]).datetime64).tolist()
            elif time_format == "iso":
                self.df_info["time_range"] = str(Time(self.df_info["time_range"]).iso).tolist()
                self.df_info["drift_time"] = str(Time(self.df_info["drift_time"]).iso).tolist()
            else:
                self.df_info["time_range"] = self.df_info["time_range"].tolist()
                self.df_info["drift_time"] = self.df_info["drift_time"].tolist()
            self.df_info["att_cols"] = self.df_info["att_cols"].tolist()
            self.df_info["file"] = file_name
            self.df_info["date_generated"] = str(datetime.now())
            with open(path+"%s.json"%file_name, "w") as f:
                json.dump(self.df_info, f)
        print("Done")
        return 
    