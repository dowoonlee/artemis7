from astropy.time import Time
# from dwlib.data.concept_drift_generator import concept_drift_generator

# prob_type = "regression"#"classification"
# time_range = ["2012-01-01", "2020-01-01"]
# drift_time_seq = ["2016-01-01"]
# size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
# n_cont, n_disc = 5, 5
# strength = 5
# noise = 0.1


# drift = "virtual"
# drift_type = "sudden"

# path = "./dwlib/data/generated_table/"
# for drift in ["virtual", "real"]:
#     for drift_type in ["sudden", "incremental", "gradual"]:

#         file_name = (prob_type[0]+drift[0]+drift_type[0]).upper()+"%d%d"%(n_cont, n_disc)

#         cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
#         prob_type=prob_type,
#         n_drift_type=1, n_cont = n_cont, n_disc=n_disc,
#         dt = 1/24)
#         dts = Time(drift_time_seq[0]).mjd
#         base = cdg.generate(drift= drift, drift_type=drift_type, strength=strength, noise=noise)
#         cdg.save_to_csv(path = path, file_name=file_name)

from dwlib.data.time_series_generator import time_series_generator

prob_type = "forecasting"
time_range = ["2012-01-01", "2020-01-01"]
drift_time_seq = ["2016-01-01"]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 6, 4
dt = 1/24

tsg = time_series_generator(time_range=time_range,
                            drift_time_sequence=drift_time_seq,
                            prob_type=prob_type,
                            n_cont=n_cont,
                            dt =dt)
df = tsg.generate(drift_type="incremental", period=365)
tsg.save_to_csv("./", "FPI_test")

import pandas as pd
df = pd.read_csv("./FPI_test.csv", index_col=0)
import matplotlib.pyplot as plt
plt.plot(df.time, df.x)
plt.show()