# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab
import pandas as pd
import numpy as np
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD as cpd

def setup_model(filepath):
    pd_file = pd.read_csv(filepath)

    weights = pd_file.to_numpy()
    dbn = DBN()
    dbn.add_edges_from([(('A', 0), ('A', 1))])
    test_CPD = cpd(('A', 1), 2, weights, evidence = [('A', 0)],
    evidence_card = [2])
    print(test_CPD)
    dbn.add_cpds(test_CPD)
    return dbn

if __name__ == "__main__":
    dbn = setup_model(r'/Users/sparr/projects/AASE/params/dbn_example.csv')
    print(dbn.get_cpds())