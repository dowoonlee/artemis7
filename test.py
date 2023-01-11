from dwlib.data.concept_drift_generator import concept_drift_generator
from dwlib.stats.frechet_inception_distance import FID
from dwlib.stats.binning import binning
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint, choice
import pandas as pd

time_range = ["2012-01-01", "2020-01-01"]
t0 = Time("2014-01-01").mjd
drift_time_seq = [str(Time(t0+90*i, format="mjd").datetime64) for i in range(20)]
size = int(Time(time_range[1]).mjd - Time(time_range[0]).mjd)
n_cont, n_disc = 5, 5
cdg = concept_drift_generator(time_range=time_range, drift_time_sequence=drift_time_seq,
n_drift_type=1, n_cont = n_cont, n_disc=n_disc,
dt = 1/24)

path = "./dwlib/data/generated_table/"
file_name = "VG55"

# dts = Time(drift_time_seq[0]).mjd
# base = cdg.generate(drift= "virtual", drift_type="gradual", strength=20, noise=0)
# cdg.save_to_csv(path = path, file_name=file_name)

# df = pd.read_csv(path+"%s.csv"%file_name, index_col=0)
# tab_info = cdg.read_info(path=path, file_name=file_name)
