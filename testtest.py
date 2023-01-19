import numpy as np
from dwlib.data.concept_drift_generator import concept_drift_generator
from dwlib.util.progressbar import progressbar
from astropy.time import Time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as reg_model

time_range = ["2012-01-01", "2020-01-01"]
drift_time_seq = ["2016-01-01"]
#t0 = Time("2014-01-01").mjd
#drift_time_seq = [str(Time(t0+90*i, format="mjd").datetime64) for i in range(20)]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 5, 5
strength = 5
noise = 0.1
prob_type= "regression"


drift = "real"
drift_type = "sudden"

file_name = (drift[0]+drift_type[0]).upper()+"%d%d"%(n_cont, n_disc)

cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
prob_type = prob_type, n_drift_type=1, n_cont = n_cont, n_disc=n_disc,
dt = 1/24)

base = cdg.generate(drift= drift, drift_type=drift_type, strength=strength, noise=noise)

print(base)
class linear_regress():
    def __init__(self, df, dt):
        self.drop_cols = ["time", "y", "drift"]
        r_init = 0.7
        dts = Time(drift_time_seq[0]).mjd
        t_init = r_init * dts + (1-r_init) * df.time.min()
        self.trange = np.arange(int(t_init), int(df.time.max()-dt))
        self.df_init = df[df.time<t_init]
        self.time = np.array([df[(df.time>=t) & (df.time<t+dt)].time.mean() for t in self.trange])   

        df_init = self.df_init.drop(columns=["time", "drift"])
        X0 = df_init.loc[:, df_init.columns != "y"].values
        y0 = df_init.y.values

        reg = reg_model().fit(X0, y0)
        y = reg.predict(X0)

        score = np.zeros(len(self.time))
        pb = progressbar(self.trange[-1]-self.trange[0])
        pb.start()
        for idx, t in enumerate(self.trange):
            pb.update(idx)
            df_comp = df[(df.time>=t) & (df.time<t+dt)]
            X, y = df_comp.drop(columns=self.drop_cols).values, df_comp.y.values
            score[idx] = reg.score(X, y)
        pb.finish()
        self.score = score
lr = linear_regress(base, 30)



plt.plot(Time(lr.time, format="mjd").datetime64, lr.score)
for v in Time(drift_time_seq).datetime64:
    plt.vlines(v, lr.score.min(), lr.score.max(), linestyles=":", colors='r')
plt.xlim(Time(lr.time[0], format="mjd").datetime64, Time(lr.time[-1], format="mjd").datetime64) 
plt.ylim(lr.score.min(), lr.score.max())
plt.show()