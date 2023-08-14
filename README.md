# ARTEMIS7

## Analytics, Reporting, and Tools for Exploratory Measures in Statistics


This library is pre-released. you can install artemis from PyPI with `pip`:
```bash
pip install artemis7
```

- datagenerator
  - virtual_drift_generator.py
- myplot
  - MyPlot.py
- stats
  - binning.py
    - Sturges' Formula : [Sturges, H. A. (1926)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1926.10502161)
    - Doane's Formula : Doane DP (1976)
    - Scott's normal reference rule : [Scott, David W. (1979)](https://academic.oup.com/biomet/article-abstract/66/3/605/232642)
    - Freedman-Diaconis' choice : [Freedman, David; Diaconis, P. (1981)](https://bayes.wustl.edu/Manual/FreedmanDiaconis1_1981.pdf)
  - frechet_inception_distance.py
  - piecewise_rejection_sampling.py : [Credit](https://axect.github.io/posts/006_prs/)
  - multivariate_sampling.py
- util
  - colorprint.py
  - funtion_list.py
  - make_movie.py
  - progressbar.py
