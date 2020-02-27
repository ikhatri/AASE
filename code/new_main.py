# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from matplotlib import pyplot as plt
from pathlib import Path
import logging
from dataloader import *
from argoverse.map_representation.map_api import ArgoverseMap
from new_model import *
import math

SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('glare_example/')
PARAMS_DIR = Path('params/')
logger = logging.getLogger(__name__)

def plot_probs(probs: dict, interval: int):
  timestep = 1/(10/interval) # we take every ith entry from 10/second
  x_axis = [x*timestep for x in range(len(probs['r']))]
  plt.figure()
  plt.plot(x_axis, probs['r'], 'r-')
  plt.plot(x_axis, probs['g'], 'g-')
  plt.plot(x_axis, probs['y'], 'y-')
  plt.xlabel('time in seconds')
  plt.ylabel('probability of light state')
  plt.show()

if __name__ == "__main__":
  end_time = 150
  interval = 10
  adj_obj_ids = [0,1,2]
  cross_obj_ids = []
  argo_data = load_all_logs(GLARE_DIR)
  city_map = ArgoverseMap()
  # visualize(city_map, argo_data, end_time)
  evidence_dict = get_evidence(city_map, argo_data, end_time)
  total_discr_evidence_dict = {}
  pom_evidence_dicts = [{} for t in range(1, end_time//interval)]
  for i in range(len(evidence_dict)):
    if i in adj_obj_ids or i in cross_obj_ids:
      discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, interval, i)
      for t in discr_evidence_dict:
        key, value, timestep = convert_pgmpy_pom(t, discr_evidence_dict[t])
        print(key, value)
        pom_evidence_dicts[timestep][key] = value
  pom_evidence_dicts.pop(0)
  
  filepath = Path('params')
  dbn, names = init_DBN(filepath, adj_obj_ids, cross_obj_ids)
  dbn.bake()
  for i, evidence in enumerate(pom_evidence_dicts):
    next_belief = predict_DBN(dbn, names, evidence, i+1)
    dbn, names = iterate_DBN(filepath, adj_obj_ids, cross_obj_ids, next_belief, i+1)
    dbn.bake()

  # plot_probs(plottable_data, interval)
