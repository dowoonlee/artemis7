import numpy as np
import matplotlib.pyplot as plt
from dwlib.util.concept_drift_generator import concept_drift_generator
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning




# time_range = ["2012-01-01", "2018-01-01"]
# drift_time_sequence = ["2014-06-01", "2016-06-01"]
# from astropy.time import Time
# size = int(Time(time_range[1]).mjd-Time(time_range[0]).mjd)*24
# n_cont = 10
# n_disc = 0
# n_label= 2
# cdg = concept_drift_generator(time_range, drift_time_sequence, size, n_cont, n_disc)
# df_sudden = cdg.generate(drift="virtual", n_label=n_label, drift_type="sudden")
# df_incre = cdg.generate(drift="virtual", n_label=n_label, drift_type="incremental")

# def actuator(dset1, dset2):
#     norm=True
#     b1 = binning(dset1[:,0])
#     b2 = binning(dset2[:,0])
#     bins = np.max([b1.bins(), b2.bins()])

#     xr = (np.min([dset1[:, 0].min(), dset2[:, 0].min()]), np.max([dset1[:, 0].max(), dset2[:, 0].max()]))
#     h1, _ = np.histogram(dset1[:, 0], bins=bins, range=xr, density=norm)
#     h2, _ = np.histogram(dset2[:, 0], bins=bins, range=xr, density=norm)

#     act1, act2 = h1, h2

#     for i in range(1, len(dset1[0, :])):
#         xr = (np.min([dset1[:, i].min(), dset2[:, i].min()]), np.max([dset1[:, i].max(), dset2[:, i].max()]))
#         h1, _ = np.histogram(dset1[:, i], bins=bins, range=xr, density=norm)
#         h2, _ = np.histogram(dset2[:, i], bins=bins, range=xr, density=norm)
#         act1 = np.vstack((act1, h1))
#         act2 = np.vstack((act2, h2))
#     return FID(act1, act2)

# class data_handler():
#     def __init__(self, df):
#         self.df = df
#     def slic(self, yr, mon=None):
#         if mon is None:
#             cond = (self.df["time"]>=Time("%04d-01-01"%yr).mjd) & (self.df["time"]<Time("%04d-01-01"%(yr+1)).mjd)
#         else:
#             if mon==12:
#                 cond = (self.df["time"]>=Time("%04d-12-01"%yr).mjd) & (self.df["time"]<Time("%04d-01-01"%(yr+1)).mjd)
#             else:
#                 cond = (self.df["time"]>=Time("%04d-%02d-01"%(yr, mon)).mjd) & (self.df["time"]<Time("%04d-%02d-01"%(yr, mon+1)).mjd)
#         return self.df[cond]

# dh_sudden = data_handler(df_sudden)
# dh_incre = data_handler(df_incre)

# t = []
# fid_sudden, fid_incre = [] , []
# d0_sudden = dh_sudden.slic(2012)
# d0_incre = dh_incre.slic(2012)
# for yr in range(2013, 2018):
#     for mon in range(1, 13):
#         t.append(Time("%04d-%02d-02"%(yr, mon)).mjd)
#         fid_sudden.append(actuator(d0_sudden.drop(columns="time").to_numpy()
#         , dh_sudden.slic(yr, mon).drop(columns="time").to_numpy()))
#         fid_incre.append(actuator(d0_incre.drop(columns="time").to_numpy()
#         , dh_incre.slic(yr, mon).drop(columns="time").to_numpy()))
# t = np.array(t)

# plt.plot(t, fid_sudden, c='k', label="sudden")
# plt.plot(t, fid_incre, c="r", label="incre")
# plt.legend(loc="best")
# plt.vlines(Time(drift_time_sequence[0]).mjd, 0, 100)
# plt.show()
