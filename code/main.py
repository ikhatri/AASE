# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from pathlib import Path
import logging
from dataloader import load_all_logs, get_evidence, get_discretized_evidence_for_object, visualize
from argoverse.map_representation.map_api import ArgoverseMap
from model import setup_traffic_DBN
from model import get_inference_model
SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('glare_example/')
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    end_time = 150
    interval = 10
    argo_data = load_all_logs(GLARE_DIR)
    city_map = ArgoverseMap()
    # visualize(city_map, argo_data, end_time)
    evidence_dict = get_evidence(city_map, argo_data, end_time)
    for i, o in enumerate(evidence_dict):
      for t in evidence_dict[o]:
        print("id", i, t, evidence_dict[o][t])
    discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, interval, obj_id = 2)
    dbn = setup_traffic_DBN(Path('params'))
    model = get_inference_model(dbn)
    variables = [('Traffic Light', x) for x in range(1, 110//interval)]
    results =  model.query(variables = variables, evidence = discr_evidence_dict)
    for d in results:
      print(d, results[d])