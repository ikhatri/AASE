# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from pathlib import Path
import logging
from dataloader import *
from argoverse.map_representation.map_api import ArgoverseMap
from model_pom import *
import math
import re
import pomegranate as pom

SAMPLE_DIR = Path("sample-data/")
GLARE_DIR = Path("glare_example/")
ARGOVERSE_TRACKING = Path("/home/ikhatri/argoverse/argoverse-api/argoverse-tracking")
RELEVANT_JSON = Path("misc/relevant_cars.json")
PARAMS_DIR = Path("params/")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("Using GPU?", pom.utils.is_gpu_enabled())
    interval = 10  # out of 10 hz, so it's every 5th image of the 10/second that we have
    folders = ["train1"]  # , 'train1', 'train2', 'train3', 'train4', 'val']
    city_map = ArgoverseMap()
    for folder in folders:
        argo_loader = load_all_logs(ARGOVERSE_TRACKING.joinpath(folder))
        relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
        for log_id in relevant_cars:
            if relevant_cars[log_id].get("skip", False) is False:
                argo_data = argo_loader.get(log_id)
                end_time = argo_data.num_lidar_frame - 1
                adj_obj_ids = relevant_cars[log_id]["adj_cars"]
                cross_obj_ids = relevant_cars[log_id]["cross_cars"]
                evidence_dict = get_evidence(city_map, argo_data, end_time)
                total_discr_evidence_dict = {}
                pom_evidence_dicts = [{} for t in range(0, (end_time // interval) + 1)]
                for i in range(len(evidence_dict)):
                    if i in adj_obj_ids or i in cross_obj_ids:
                        discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, interval, i)
                        for t in discr_evidence_dict:
                            key, value, timestep = convert_pgmpy_pom(t, discr_evidence_dict[t])
                            pom_evidence_dicts[timestep][key] = value
                pom_evidence_dicts.pop(0)

                # Make a copy of the evidence dictionaries with YOLO included
                evidence_with_yolo = pom_evidence_dicts.copy()
                ft, yolo = parse_yolo(ARGOVERSE_TRACKING.joinpath(folder + "/" + log_id + "/rfc.txt"))
                yolo_evidence = yolo_to_evidence(yolo, ft, interval)
                for t, e in enumerate(evidence_with_yolo):
                    if t in yolo_evidence:
                        e.update(yolo_evidence[t])

                filepath = Path("params")
                dbn, names = init_DBN(filepath, adj_obj_ids, cross_obj_ids)
                dbn.bake()
                all_evidence = [pom_evidence_dicts, evidence_with_yolo]
                all_predictions = [[], []]
                timing = [[], []]
                for which_evidence, evidence_list in enumerate(all_evidence):
                    for i, evidence in enumerate(evidence_list):
                        start = timeit.default_timer()
                        next_belief, out = predict_DBN(dbn, names, evidence, i + 1, iterations=7)
                        all_predictions[which_evidence] = all_predictions[which_evidence] + out
                        dbn, names = iterate_DBN(filepath, adj_obj_ids, cross_obj_ids, next_belief, i + 1)
                        dbn.bake()
                        stop = timeit.default_timer()
                        execution_time = stop - start
                        timing[which_evidence].append(execution_time)
