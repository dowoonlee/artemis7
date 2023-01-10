from dwlib.data.concept_drift_generator import concept_drift_generator
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint, choice

time_range = ["2012-01-01", "2018-01-01"]
drift_time_seq = ["2014-01-01", "2014-01-02"]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 5, 5
cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
n_drift_type=1, n_cont = n_cont, n_disc=n_disc,
dt = 1/24)


# dts = Time(drift_time_seq[0]).mjd
# base = cdg.generate(drift= "virtual", drift_type="sudden", strength=20, noise=0)
# cdg.save_to_csv(path = "./dwlib/data/generated_table/")


import json
with open("./dwlib/data/generated_table/vs_20230110162056.json", "r") as f:
    file_info = json.load(f)
for key in file_info.keys():
    print(key, " ", file_info[key])


# att_cols = cdg.attribute_columns
# for ac in att_cols:
#     if ac<n_cont:
#         colname = "C%02d"%ac
#     else:
#         colname = "D%02d"%(ac-n_cont)
#     bef = base[base.time<dts][colname].values
#     aft = base[base.time>=dts][colname].values
#     plt.title("%s"%(colname))
#     plt.hist(bef, histtype='step', density=True, bins= binning(bef).bins(), label="%0.2f %0.2f"%(np.mean(bef), np.std(bef)))
#     plt.hist(aft, histtype='step', density=True, bins= binning(aft).bins(), label="%0.2f %0.2f"%(np.mean(aft), np.std(aft)))
#     plt.legend(loc='best')
#     plt.show()
    # t = base.time

    # plt.title(colname)
    # col_now = base[colname]
    # plt.plot(t, col_now)
    # plt.vlines(Time(drift_time_seq[0]).mjd, col_now.min(), col_now.max())
    # plt.show()