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
import re
import pomegranate as pom

SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('glare_example/')
ARGOVERSE_TRACKING = Path('/home/ikhatri/argoverse/argoverse-api/argoverse-tracking')
RELEVANT_JSON = Path('misc/relevant_cars.json')
PARAMS_DIR = Path('params/')
logger = logging.getLogger(__name__)

def plot_probs(probs: dict, interval: int):
  # Reconfiguring the pom output dict to be plottable
  plottable = {'red': [], 'green': [], 'yellow': []}
  for t in range(len(probs)):
    for k in plottable:
      plottable[k].append(probs[t][k])
  timestep = 1/(10/interval) # we take every ith entry from 10/second
  x_axis = [x*timestep for x in range(len(plottable['red']))]
  plt.figure()
  plt.plot(x_axis, plottable['red'], 'r-')
  plt.plot(x_axis, plottable['green'], 'g-')
  plt.plot(x_axis, plottable['yellow'], 'y-')
  plt.xlabel('time in seconds')
  plt.ylabel('probability of light state')

def plot_runtime(times: list):
  plt.figure()
  plt.plot(range(len(times)), times, 'b-')
  plt.ylabel('Inference Time')
  plt.xlabel('Timestep')
  plt.ylim(ymin=0)

if __name__ == "__main__":
  print('Using GPU?', pom.utils.is_gpu_enabled())
  interval = 10 # out of 10 hz, so it's every 5th image of the 10/second that we have
  folders = ['sample']#, 'train1', 'train2', 'train3', 'train4', 'val']
  city_map = ArgoverseMap()
  for folder in folders:
    argo_loader = load_all_logs(ARGOVERSE_TRACKING.joinpath(folder))
    relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
    for log_id in relevant_cars:
      argo_data = argo_loader.get(log_id)
      end_time = argo_data.num_lidar_frame-1
      adj_obj_ids = relevant_cars[log_id]['adj_cars']
      cross_obj_ids = relevant_cars[log_id]['cross_cars']
      evidence_dict = get_evidence(city_map, argo_data, end_time)
      total_discr_evidence_dict = {}
      pom_evidence_dicts = [{} for t in range(0, (end_time//interval)+1)]
      for i in range(len(evidence_dict)):
        if i in adj_obj_ids or i in cross_obj_ids:
          discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, interval, i)
          for t in discr_evidence_dict:
            key, value, timestep = convert_pgmpy_pom(t, discr_evidence_dict[t])
            pom_evidence_dicts[timestep][key] = value
      pom_evidence_dicts.pop(0)

      ft, yolo = parse_yolo(ARGOVERSE_TRACKING.joinpath(folder+'/'+log_id+'/rfc.txt'))
      yolo_evidence = yolo_to_evidence(yolo, ft, interval)
      for t, e in enumerate(pom_evidence_dicts):
        if t in yolo_evidence:
          e.update(yolo_evidence[t])

      filepath = Path('params')
      dbn, names = init_DBN(filepath, adj_obj_ids, cross_obj_ids)
      dbn.bake()
      pom_out = []
      timing = []
      for i, evidence in enumerate(pom_evidence_dicts):
        start = timeit.default_timer()
        next_belief, out = predict_DBN(dbn, names, evidence, i+1, iterations=7)
        pom_out = pom_out + out
        dbn, names = iterate_DBN(filepath, adj_obj_ids, cross_obj_ids, next_belief, i+1)
        dbn.bake()
        stop = timeit.default_timer()
        execution_time = stop - start
        timing.append(execution_time)

      plot_probs(pom_out, interval)
      plt.show()
