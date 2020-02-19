# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from pathlib import Path
import logging
from dataloader import load_all_logs, get_evidence, get_discretized_evidence_for_object, visualize, get_evidence_start_and_finish_for_object
from argoverse.map_representation.map_api import ArgoverseMap
from model import setup_traffic_DBN
from model import setup_backbone_DBN
from model import add_cars_DBN
from model import get_inference_model
import math
SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('glare_example/')
PARAMS_DIR = Path('params/')
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    end_time = 150
    interval = 10
    adj_obj_ids = [0,1]
    cross_obj_ids = []
    argo_data = load_all_logs(GLARE_DIR)
    city_map = ArgoverseMap()
    # visualize(city_map, argo_data, end_time)
    evidence_dict = get_evidence(city_map, argo_data, end_time)
    backbone_dbn, our_light, cross_light = setup_backbone_DBN(PARAMS_DIR)
    dbn = add_cars_DBN(PARAMS_DIR, backbone_dbn, our_light, cross_light, adj_obj_ids, cross_obj_ids)
    # dbn = setup_traffic_DBN(Path('params'))
    obj_ids = adj_obj_ids + cross_obj_ids
    model = get_inference_model(dbn)
    earliest_evidence = math.inf
    latest_evidence = 0
    for obj_id in obj_ids:
      start, stop = get_evidence_start_and_finish_for_object(evidence_dict, interval, obj_id)
      if start < earliest_evidence:
        earliest_evidence = start
      if stop > latest_evidence:
        latest_evidence = stop

    for i in range(earliest_evidence+1, latest_evidence+2):
      evidence_up_to_present = {}
      for obj_id in obj_ids:
        evidence_up_to_present = get_discretized_evidence_for_object(evidence_dict, interval, obj_id, up_to = i, init_evidence_dict=evidence_up_to_present)

      print(evidence_up_to_present)
      results = model.query(variables = [('Our Light', i)], evidence = evidence_up_to_present)
      for d in results:
        print(d, results[d])