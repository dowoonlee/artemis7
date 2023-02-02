import numpy as np
import pandas as pd
from astropy.time import Time
import json
from numpy.random import uniform, normal, gamma, randn
from scipy.special import gamma as gamma_func
import sys, os
sys.path.append("/".join(os.path.abspath(__file__).split("\\")[:-3]))
from datagenerator.util.progressbar import progressbar
from scipy.signal import savgol_filter
from datetime import datetime

"""
Time Series Generator

1. Initialization

Between (time_range[0]~time_range[1]), Drifts take place at 'drfit_time_sequence'.
'time_range' and 'drift_time_sequence' support float (mjd) and string (datetime64, yyyy-mm-dd)

TSG only support univariate, periodic case (2023-01-30).
TSG can support both periodic/non-periodic cases (2023-01-31).

2. Generation
    1) periodic case : based on sinusodial function (combination of various period of sine functions)
    2) non-poeriodic case : based on MCMC, gamma distribution is the target function.

generate() methods supports 2types of drift modes periodic(period>0)/non-periodic(period<=0) and 2types of drift pattern [sudden/incremental].
In case of drift mode, setting period as positive one leads to periodic univariate time series data.
"""

class time_series_generator():
    def __init__(self, time_range, drift_time_sequence, prob_type, **kwargs):
        """
        time_range : array-like. 데이터셋의 시작시간과 끝시간. (start, end)형태로 입력.
        drift_time_sequence : array-like. drift가 작동하는 시간.
        prob_type : [forecasting]
        n_cont : 연속데이터 컬럼의 갯수
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
        
        # self.n_att = kwargs["n_att"] if "n_att" in keys else sum(self._ncol)//2

        # if sum(self._ncol)<=0:
        #     raise ValueError("At least 1 column should be exist")

        tol = 1e-6
        self._drift_sequence = np.sort(np.concatenate((drift_time_sequence, [time_range[1]+tol , time_range[0]-tol]), axis=0))
        self.prob_type = prob_type

        self.w_cand = [1/24/6, 1/24 ,1, 7, 30, 60, 90, 180, 365, 730]
        
        self.base_w = 0
        self.drift_w = 0


        self.noise = 0
        self.df = None
        self.df_info = {
            "file" : None,
            "prob_type" : self.prob_type,
            "date_generated" : None,
            "time_range" : time_range,
            "drift_time" : drift_time_sequence,
            "period": 0,
            "drift_type" : None,
            "noise" : 0
        }
        
    @staticmethod
    def _smooth(x):
        return savgol_filter(x, 5, 3)

    @staticmethod
    def _random_walk(x0, c0):
        k, t, kernel_std = c0
        def gamma_pdf(k, t, x):
            t1 = 1/(gamma_func(k)*(t**k))
            t2 = x**(k-1)
            t3 = np.exp(-x/t)
            return t1*t2*t3

        while True:
            p0 = gamma_pdf(k, t, x0)
            x1 = -1
            while x1<0:
                x1 = randn()*kernel_std+x0
            p1 = gamma_pdf(k, t, x1)
            if p1/p0>uniform():
                return x1
            else:
                return x0


    @staticmethod
    def _linear(y1, y2, x):
        a = (y2 - y1)/(x[-1] - x[0])
        b = y1 - a * x[0]
        y = a * x + b
        return y
    
    def _nearest_cos_pt(self, Y, args):
        # f(x) = A cos((2pi/w)*(x-a))+b
        # A : amplitude
        # w : period
        # a, b = coeff
        # Y : last position
        t = np.arange(0, self.w_cand[-1], self.time[1]-self.time[0])
        y = self._cos(t, args)
        dy = np.abs(y-Y)
        x = t[np.where(dy == np.min(dy))[0]][0]
        return x
    
    def _cos(self, x, args):
        # f(x) = A cos((2pi/w)*(x-a))+b
        # A : amplitude
        # w : period
        # a, b = coeff
        try:
            main_y = np.zeros(len(x))
        except TypeError:
            main_y = np.zeros(1)
        for w in self.w_cand:
            A, a, b = args[w]
            dy = A*np.cos((2*np.pi/w)*(x-a))+b
            #if len(dy.shape)>1:
            main_y += dy
        return main_y
    
    def _add_noise(self, x, base_coeff, drift_coeff):
        pb = progressbar(len(self._drift_sequence)-1, "AN")
        pb.start()
        on_drift = False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            pb.update(idx)
            cond = np.where((self.time>ts) & (self.time<=te))[0]
            coeff = drift_coeff[self.drift_w] if on_drift else base_coeff[self.base_w]

            variance = (coeff[0] * self.noise)**2
            for ele in cond:
                x[ele] = normal(loc=x[ele], scale=np.sqrt(variance))
            on_drift = not on_drift
        pb.finish()
        return
    
    

    def _periodic_sudden(self, base_coeff, drift_coeff):
        on_drift=False
        x = np.array([])
        pb = progressbar(len(self._drift_sequence)-1, "TSG")
        pb.start()
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            pb.update(idx)
            cond = (self.time>ts) & (self.time<=te)
            sub_size = np.sum(cond)
            coeff = drift_coeff if on_drift else base_coeff
            t0 = 0 if idx ==0 else self._nearest_cos_pt(x[-1], coeff)
            t_sub = np.linspace(t0, t0+(te-ts), sub_size)
            x = np.concatenate((x, self._cos(t_sub, coeff)),axis=0)
            self._drift_weight[np.where(cond)[0]] = int(on_drift)
            on_drift = not on_drift
        pb.finish()
        return x
    
    def _periodic_incremental(self, base_coeff, drift_coeff):
        r = 0.1
        pb = progressbar(len(self.time), "TSG")
        pb.start()
        on_drift=False
        x = np.array([])
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff_bef = base_coeff if on_drift else drift_coeff
            coeff_now = drift_coeff if on_drift else base_coeff      
            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(r*itv_len)
            pre_pivot = cur_interval[int(itv_len*r)]
            suf_pivot = cur_interval[int(itv_len*(1-r))]

            t0 = 0 if idx ==0 else self._nearest_cos_pt(x[-1], coeff)
            
            for c, i in enumerate(cur_interval):
                pb.update(i)
                if i<pre_pivot and ts != self._drift_sequence[0]:
                    bef = (l-c)/(2*l)
                elif i>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                coeff = dict()
                for wc in self.w_cand:
                    coeff[wc] = bef*coeff_bef[wc] + (1-bef) * coeff_now[wc]
                self._drift_weight[i] = 1-bef if on_drift else bef
                x = np.append(x, self._cos(t0 + self.time[i], coeff))

            on_drift = not on_drift



        pb.finish()
        return x

    def _generate_periodic(self, drift_type, noise, period):
        
        self.base_w = period
        self.drift_w  = 1*self.base_w

        base_coeff = {self.base_w : np.array([uniform(1, 10), uniform(0, 2*self.base_w), uniform(1, 10)])}
        A, _, b = base_coeff[self.base_w]
        for wc in self.w_cand:
            if wc != self.base_w:                
                base_coeff[wc] = np.array([uniform(0.5, A/5), uniform(-wc, wc), uniform(0, 2*b)])

        drift_coeff = {self.drift_w : np.array([uniform(1, 10), uniform(0, 2*self.drift_w), uniform(1, 10)])}
        A, _, b = drift_coeff[self.drift_w]
        for wc in self.w_cand:
            if wc != self.drift_w:                
                drift_coeff[wc] = np.array([uniform(0.5, A/5), uniform(-wc, wc), uniform(0, 2*b)])

        if drift_type == "sudden":
            x = self._periodic_sudden(base_coeff=base_coeff, drift_coeff=drift_coeff)

        elif drift_type == "incremental":
            x = self._periodic_incremental(base_coeff=base_coeff, drift_coeff=drift_coeff)

        else:
            raise KeyError("There is no such type of drift [%s]"%drift_type)
        
        if noise>0:
            self._add_noise(x = x, base_coeff=base_coeff, drift_coeff=drift_coeff)


        df = pd.DataFrame({
            "time": self.time,
            "x": x
        })

        return df
    
    def _random_sudden(self, base_coeff, drift_coeff):
        x = [gamma(base_coeff[0], base_coeff[1])]
        pb = progressbar(len(self.time), "TSG")
        pb.start()
        on_drift = False
        cnt = 0
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff = drift_coeff if on_drift else base_coeff
            interval_len = sum((self.time>ts) & (self.time<=te))
            self._drift_weight[np.where((self.time>ts) & (self.time<=te))[0]] = int(on_drift)
            for i in range(interval_len):
                pb.update(cnt)
                x = np.append(x, self._random_walk(x[-1], coeff))
                cnt+=1
            on_drift = not on_drift
        pb.finish()
        return x[1:]
    
    def _random_incremental(self, base_coeff, drift_coeff):
        x = [gamma(base_coeff[0], base_coeff[1])]
        r = 0.1
        pb = progressbar(len(self.time), "TSG")
        pb.start()
        on_drift = False
        for idx, (ts, te) in enumerate(zip(self._drift_sequence[:-1], self._drift_sequence[1:])):
            coeff_bef = base_coeff if on_drift else drift_coeff
            coeff_now = drift_coeff if on_drift else base_coeff

            cur_interval = np.where((self.time>ts) & (self.time<=te))[0]
            itv_len = len(cur_interval)
            l = int(r*itv_len)
            pre_pivot = cur_interval[int(itv_len*r)]
            suf_pivot = cur_interval[int(itv_len*(1-r))]

            for c, i in enumerate(cur_interval):
                pb.update(i)
                if i<pre_pivot and ts != self._drift_sequence[0]:
                    bef = (l-c)/(2*l)
                elif i>suf_pivot and te != self._drift_sequence[-1]:
                    bef = (l-(itv_len-c))/(2*l)
                else:
                    bef = 0
                coeff = bef*coeff_bef + (1-bef)*coeff_now
                self._drift_weight[i] = 1-bef if on_drift else bef
                x = np.append(x, self._random_walk(x[-1], coeff))
            on_drift = not on_drift
        pb.finish()
        return x[1:]



    def _generate_random(self, drift_type, noise):
        base_coeff = np.array([uniform(1, 10), uniform(1, 10), 2])
        drift_coeff = np.array([uniform(1, 10), uniform(1, 10), 2])
        #base_coeff = np.append(base_coeff, base_coeff[1]*(1+noise))
        #drift_coeff = np.append(drift_coeff, drift_coeff[1]*(1+noise))

        if drift_type == "sudden":
            x = self._random_sudden(base_coeff=base_coeff, drift_coeff=drift_coeff)

        elif drift_type == "incremental":
            x = self._random_incremental(base_coeff=base_coeff, drift_coeff=drift_coeff)

        else:
            raise KeyError("There is no such type of drift [%s]"%drift_type)
        
        df = pd.DataFrame({
            "time": self.time,
            "x": x
        })

        return df

    def generate(self, drift_type = "sudden", period = 0, time_format="mjd", **kwargs):

        keys = kwargs.keys()
        if "noise" in keys:
            self.noise = kwargs["noise"]
            self.df_info["noise"] = self.noise
        
        if period>0:
            df = self._generate_periodic(period=period, drift_type = drift_type, noise = self.noise)
        else:
            df = self._generate_random(drift_type=drift_type, noise=self.noise)

        if time_format == "datetime64":
            df["time"] = Time(np.round(df.time.to_numpy(), 3),  format="mjd").datetime64
        df["drift"] = self._drift_weight
        
        self.df_info["period"] = period
        self.df_info["drift_type"] = drift_type
        self.df = df
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
            tp = self.df_info["prob_type"][0] + ("P" if self.df_info["period"]>0 else "NP") + self.df_info["drift_type"][0]
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
            self.df_info["file"] = file_name
            self.df_info["date_generated"] = str(datetime.now())
            with open(path+"%s.json"%file_name, "w") as f:
                json.dump(self.df_info, f)
        print("Done")
        return 