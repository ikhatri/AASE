# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from pathlib import Path
import logging
from dataloader import load_all_logs, get_evidence, get_discretized_evidence_for_object, visualize, get_evidence_start_and_finish_for_object
from argoverse.map_representation.map_api import ArgoverseMap
from model import setup_traffic_DBN
from model import get_inference_model
SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('glare_example/')
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    end_time = 150
    interval = 5
    obj_id = 0
    argo_data = load_all_logs(GLARE_DIR)
    city_map = ArgoverseMap()
    # visualize(city_map, argo_data, end_time)
    evidence_dict = get_evidence(city_map, argo_data, end_time)
    start, stop = get_evidence_start_and_finish_for_object(evidence_dict, interval, obj_id)
    dbn = setup_traffic_DBN(Path('params'))
    model = get_inference_model(dbn)
    for i in range(start+1, stop+2):
      print("inference for ", )
      evidence_up_to_present = get_discretized_evidence_for_object(evidence_dict, interval, obj_id, up_to = i)
      results =  model.query(variables = [('Traffic Light', i)], evidence = evidence_up_to_present)
      for d in results:
        print(d, results[d])