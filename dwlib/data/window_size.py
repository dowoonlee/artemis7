import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import gc


def fft_to_cycle(y, dt):
    """
    y : 주기를 구할 데이터
    dt : 시간 간격
    return : cycle(단위는 일) float
    """
    fftx = fftfreq(len(y), dt)
    ffty = fft(y)
    fftx_pos = fftx[fftx>0]
    ffty_pos = abs(ffty[fftx>0].real)
    cycle = 1/fftx_pos[np.argmax(ffty_pos)]
    return cycle

class WindowSize():
    """
    시계열 데이터의 전처리 및 주기 계산 class
    """
    def __init__(self, **kwargs):
        """
        df : time을 제외한 dataset. pd.DataFrame 형식으로
        time : 시간 dataset. pd.DataFrame / ndarray형식으로 입력
        col_time : time column의 column명 (mjd, jd 등의 float형이어야함)
        """
        
        self.df = kwargs["df"]
        if not isinstance(self.df, pd.DataFrame):
            ### raise type error ###
            # 추후 추가
            pass
        self.cols = self.df.columns

        self.time = kwargs["time"]
        if isinstance(self.time, pd.DataFrame):
            self.col_time = self.time.columns
            self.time = self.time.to_numpy()
        elif isinstance(self.time, pd.Series):
            self.col_time = "time"
            self.time = self.time.to_numpy()
        elif isinstance(self.time, np.ndarray):
            self.col_time = "time"
        else:
            ### raise type error ###
            # 추후 추가
            pass
        
        self.df[self.col_time] = self.time
    
    def pre_process(self, dt=False, kind="linear", normalization=False, interpolation=True, L_SG=30):
        """
        dt : 시계열의 간격. 따로 명시하지 않을 경우 기존 시계열 데이터들의 간격 중 가장 작은 값으로 설정
        kind : interpolation에 사용되는 방법. ['linear', 'nearest', 'cubic']
        nomalization : N(0, 1)로 표준화
        interpolation : 주어진 dt에 대해 interpolation하여 등간격 시계열 데이터로 정렬
        L_SG : SG-filter 의 (window size)/2, 단위는 일
        """

    ### averaging ###
    ### 단일 시간 지점에 대해서 여러개의 data point 가 존재할 경우 대표값 하나로 통일(평균값 사용)
        uni, cnt = np.unique(self.time, return_counts=True)
        if len(self.time) != len(uni):
            ovl = (cnt[np.argsort(uni)]-1).astype(bool)
            for date, overlapped in zip(np.sort(uni), ovl):
                if overlapped:
                    df_pop = self.df[self.df[self.col_time]==date]
                    self.df = self.df.drop(df_pop.index)
                    self.df = pd.concat([self.df, df_pop.groupby(level=0).mean()], ignore_index=True)
            self.df = self.df.sort_values(by=self.col_time)
            self.df.reset_index()
            self.time = self.df[self.col_time].to_numpy()

    ### normalization ###
    ### 주기 구하는데에는 크게 상관 없으므로 기본값은 false
        if normalization:
            for col in self.cols:
                m, s = self.df[col].mean(), self.df[col].std()
                self.df[col] -= m
                self.df[col] /= s
    
    ### interpolation ###
        if interpolation:
            dict_interpolated = dict()
            time_range = (np.min(self.time), np.max(self.time))
            if not dt:
                dt, cnt = np.unique(self.time[1:] - self.time[:-1], return_counts = True)
                cnt, dt = cnt[dt>0], dt[dt>0]
                dt = dt[np.argmax(cnt)]
            self.dt = dt
            time_new = np.arange(time_range[0], time_range[1], self.dt)
        
            # filter window size의 경우 추후 조정, order는 3으로
            # optimum window lenght of Savitzky-Golay filters with arbitrary order
            # Mohammad Sadeghi, Fereidoon Behnia 참조할 것
            # filter를 고려할 여유가 된다면 Kalman-filter도 추가(특히 EKF)
            # 이 부분이 시간을 제일 많이 소모
            
            for col in self.cols:
                f = interp1d(self.time, self.df[col].to_numpy(), kind=kind)
                ynew = f(time_new)
                ws = int((L_SG*2/self.dt)//2*2)+1
                # ex) L_SG = 7인 경우 data point 앞 뒤로 7일 간 data를 참고하여 smoothing
                if ws>3:
                    # ws>n인 이유는 n-th order fitting을 하기 때문 (n=3이 기본값)
                    ynew = savgol_filter(ynew, ws, 3)
                dict_interpolated[col] = ynew
            
            dict_interpolated[self.col_time] = time_new
        # dictionary를 만들어서 data frame으로 옮기는 방법
        # 더 효율적인 방법을 찾으면 수정할 것
        self.df = pd.DataFrame(dict_interpolated)

        # memory deallocation
        del dict_interpolated
        gc.collect()

        self.df = self.df.sort_values(by=[self.col_time])
        self.time = self.df[self.col_time].to_numpy()
        return

    def get_cycle(self, col=False, min_cycle = False, mode="normal", n_sub_sample=10, score_crit=0.6):
        """
        col : cycle을 계산할 column명, 따로 명시하지 않을 경우 가장 첫 column
        min_cycle : cycle의 lower bound(단위는 일), 기본값은 3*dt로 하여
                    2개 이상의 data point가 구간내 에 들어오도록 설정(r2 score 계산을 위해)
        mode : 추후에 추가할 data drift type(normal, gradual, incremental ...)
        n_sub_sample : 전체 time interval을 cycle로 나눴을 때 나오는 구간 갯수가 n_sub_sample보다 클 경우
                       계산량을 줄이기 위해서 n_sub_sample 만큼으로 한계 설정
                       cycle candidate을 찾기 위한 time interval sub sampling
        n_sub_for_cycle : sub sampling 갯수
        score_crit : r2_score>score_crit인 경우에만 유효한 cycle candidate으로 추정
        return : cycle(단위는 일), score(0~1)
        """

        # column명을 따로 설정하지 않았을 경우
        if not col:
            col = self.cols[0]

        # cycle의 lower bound를 따로 설정하지 않았을 경우
        if not min_cycle:
            min_cycle = self.dt*3
        else:
            min_cycle = np.max(min_cycle, self.dt*3)

        # cycle candidate(cycles)을 구하기 위한 sub sampling 갯수
        n_sub_for_cycle = min(int(len(self.time)/30), 50)
        t, y = self.time, self.df[col].to_numpy()
        time_range = t[-1]-t[0]
        
        cycles = []
        random_sub = np.random.rand(n_sub_for_cycle)
        for sub_samp_idx in range(n_sub_for_cycle):
            idx = int(len(t)*(1-random_sub[sub_samp_idx]*0.5))
            # 구간의 끝지점을 random하게 설정해서 해당 구간을 fft하여 cycle을 계산
            # 끝지점(idx)는 전체 구간의 절반 이후 지점으로만 설정
            # sampling된 구간이 [:idx]
            cycle = fft_to_cycle(y[:idx], self.dt)
            if cycle>min_cycle and cycle<(time_range)/2:
                cycles.append(cycle)

        scores = []
        # for all cycle_candidates
        for cycle in cycles:
            ncycle = int(np.floor(time_range/cycle))
            idxs = np.array([(ncy*cycle/time_range*len(t), (ncy+1)*cycle/time_range*len(t)) for ncy in range(ncycle)], dtype=np.int64)
            
            # 맨 첫 부분 부터 cycle마다 자르고 남은 마지막 구간이 cycle의 80% 이상일 때
            # 즉, 나머지가 좀 클 때 => 나머지 구간도 계산에 포함
            if (len(t)-idxs[-1][1])/(idxs[0][1]-idxs[0][0])>0.8:
                idxs = np.vstack((idxs, (idxs[-1][1], len(t)-1)))
                ncycle += 1
            sub_scores = []

            # n_sub_sample < (total time interval / cycle)인 경우
            # 해당 cycle로 만든 구간의 갯수가 너무 많은 경우
            # random sampling 하여 r2 score 계산
            if ncycle > n_sub_sample:
                sub_index = np.sort(np.random.choice(np.arange(0, ncycle, 1).astype(np.int64), size=n_sub_sample))
                idxs = idxs[sub_index]
                ncycle = n_sub_sample

            # combination
            for idx0 in range(ncycle):
                for idx1 in range(idx0+1, ncycle):
                    r0, r1 = idxs[idx0], idxs[idx1]
                    dr = min(r0[1]-r0[0], r1[1]-r1[0])
                    y0, y1 = y[r0[0]:r0[0]+dr], y[r1[0]:r1[0]+dr]
                    sub_scores.append(r2_score(y0, y1))
            # 각 구간끼리 비교한 값들을 sub_scores에 저장하여
            # 평균값을 사용(추후 중간값 등을 활용할 가능성)
            scores.append(np.mean(sub_scores))
        scores = np.array(scores)
        cycles = np.array(cycles)
        cycles = cycles[scores>score_crit]
        scores = scores[scores>score_crit]

        if len(cycles)<2:
            return False, False
        return cycles[np.argmax(scores)], np.max(scores)