import numpy as np
from dwlib.util.make_movie import make_movie
from dwlib.util.concept_drift_generator import concept_drift_generator
import matplotlib.pyplot as plt

# for i in range(10):
#     x = np.random.rand(10, 10)
#     plt.imshow(x)
#     plt.savefig("./test/%02d.png"%(i))
#     plt.close()

# mk = make_movie("./test", "png")
# mk.to_gif(path = "./test", output="output", fps=1)



time_range = ["2012-01-01", "2018-01-01"]
drift_time_sequence = ["2014-06-01"]
from astropy.time import Time
size = int(Time(time_range[1]).mjd-Time(time_range[0]).mjd)*24
n_cont = 10
n_disc = 0
n_label= 2
cdg = concept_drift_generator(time_range, drift_time_sequence, size, n_cont, n_disc)
df = cdg.generate(drift="real", n_label=n_label)


# for col in range(n_disc):
#     base, drift = df[df.time<Time(drift_time_sequence[0]).mjd], df[df.time>=Time(drift_time_sequence[0]).mjd]
#     xbase, xdrft = base["D%02d"%col], drift["D%02d"%col]
#     uniq = df["D%02d"%col].unique()
#     plt.hist(xbase[:int(len(xbase)/2)], bins=len(uniq), range=(np.min(uniq), np.max(uniq)), histtype='step', label='base1', density=True)
#     plt.hist(xbase[int(len(xbase)/2):], bins=len(uniq), range=(np.min(uniq), np.max(uniq)), histtype='step', label='base2', density=True)
#     plt.hist(xdrft[:int(len(xdrft)/2)], bins=len(uniq), range=(np.min(uniq), np.max(uniq)), histtype='step', label='drft1', density=True)
#     plt.hist(xdrft[int(len(xdrft)/2):], bins=len(uniq), range=(np.min(uniq), np.max(uniq)), histtype='step', label='drft2', density=True)
#     plt.legend(loc='best')
#     plt.show()


from sklearn.decomposition import PCA
X_base = df[df.time<Time(drift_time_sequence[0]).mjd].drop(columns=["time", "y"]).to_numpy()
pca = PCA(n_components=2)
pca.fit(X_base)
y = df.y.to_numpy()
X_r = pca.transform(df.drop(columns=["time","y"]).to_numpy())
for lb in range(n_label):
    plt.scatter(X_r[y==lb, 0], X_r[y==lb, 1], s=1)
plt.show()
