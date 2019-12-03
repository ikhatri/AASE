# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from pathlib import Path
import logging
from dataloader import load_all_logs, get_evidence
from argoverse.map_representation.map_api import ArgoverseMap
from model import setup_traffic_DBN
from model import get_inference_model
SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('/home/ikhatri/argoverse/argoverse-api/argoverse-tracking/glare_example/')
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    end_time = 150
    argo_data = load_all_logs(GLARE_DIR)
    city_map = ArgoverseMap()
    # visualize(mappymap, d, end_time)
    evidence_dict = get_evidence(city_map, argo_data, end_time)
    dbn = setup_traffic_DBN(Path('params'))
    model = get_inference_model(dbn)

    for i, o in enumerate(evidence_dict):
      if i == 0:
        print(model.backward_inference(variables = [('Traffic Light',10),('Traffic Light',60),('Traffic Light',120)], evidence = evidence_dict[o]))
        break