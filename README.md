# Bayesian-Optimization-For-Drug-Discovery


## Dependencies

1) Install the python packages requried by running:

```
conda env create -f bayesian.yml
```

2) Install GPy v1.9.8 and GPyOpt v1.2.5 from:

https://github.com/SheffieldML/GPyOpt

https://github.com/SheffieldML/GPy

## Run an example optimization

after installing dependencies simply run

```
python BO_Multiarmed_Bandit.py MMP12 ECFP rbfs
```

to run the optimization algorithm 10 times (with different random seeds) on the MMP12 dataset using the ECFP fingerprints and the rbfs kernel.

List of representations include: ["ECFP", "CSFP", "mol2vec"]
List of kernels include: ["rbfs", "tanimoto"]

The MMP12 dataset and fingerprints have already been preprocessed and saved in "./input_data/". The original dataset is also given in the same folder, and was originally obtained from: Pickett et al. https://pubs.acs.org/doi/10.1021/ml100191f
