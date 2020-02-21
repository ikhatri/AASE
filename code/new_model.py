# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import theano
import theano.tensor as tt
import pymc3 as pm
import matplotlib.pyplot as plt

def setup_model(filepath: Path, epsilon: float=0):
    """
    TODO: Documentation
    """
    labeled_dataframe = pd.read_csv(filepath)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype = np.float)
    model_weights = (1-epsilon)*model_weights + np.full_like(model_weights, epsilon)
    return model_weights

def setup_backbone_DBN(filepath: Path, t = 15):
    system = setup_model(filepath.joinpath('system_model.csv'), epsilon = .01)
    our_light = setup_model(filepath.joinpath('our_light_model.csv'), epsilon = .005)
    cross  = setup_model(filepath.joinpath('cross_light_model.csv'),  epsilon = .005)
    vision = setup_model(filepath.joinpath('vision_evidence.csv'), epsilon = .08)
    pass

if __name__ == "__main__":
    with pm.Model() as m:
        arr = np.array([[.1,.9],[.9,.1]])
        a = pm.Categorical('a', p=[0.5, 0.5])
        b = pm.Deterministic('b', theano.shared(arr)[a,:])
        c = pm.Categorical('c', b)
        trace = pm.sample(10000)

        print(trace['a'])