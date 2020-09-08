# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import csv
import logging
from copy import deepcopy
from pathlib import Path
from typing import List

import pomegranate as pom

from argoverse.map_representation.map_api import ArgoverseMap
from dataloader import *
from model_pom import *
from yolo_hmm import read_txt, yolo_hmm

# This file is the main input point for running all of the experiments

SAMPLE_DIR = Path("sample-data/")
GLARE_DIR = Path("glare_example/")
ARGOVERSE_TRACKING = Path("/home/ikhatri/argoverse/argoverse-api/argoverse-tracking")
RELEVANT_JSON = Path("misc/relevant_cars.json")
PARAMS_DIR = Path("params/")
RESULTS_DIR = Path("results/")
INTERVAL = 10  # out of 10 hz, so it's every 5th image of the 10/second that we have
logger = logging.getLogger(__name__)


def write_to_csv(folder: str, log: str, predictions: List, runtime: List, yolo_preds: np.array):
    """ A function to write out the results into csv files

    Args:
        folder (str): The folder (train, val etc.) of the argoverse dataset that the log is in
        log (str): The ID of the log
        predictions (List): A list of lists of output predictions for each timestep. Format: [[aase], [aase+yolo]]
        runtime (List): A list of runtimes for each timestep, in the same format as predictions.
    """
    with open(RESULTS_DIR.joinpath(f"{folder}/{log}.csv"), "w+", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        keys = ["red", "green", "yellow"]
        for e in range(2):  # Because there's 2 experiments per log, aase and aase + yolo
            for k in keys:
                writer.writerow([k] + [predictions[e][t][k] for t in range(len(predictions[e]))])
            writer.writerow(["runtime"] + runtime[e])
    np.savetxt(RESULTS_DIR.joinpath(f"{folder}/{log}_yolo.csv"), yolo_preds, delimiter=",")


if __name__ == "__main__":
    print("Using GPU?", pom.utils.is_gpu_enabled())
    folders = ["train1", "train2", "train3", "train4", "val"]
    city_map = ArgoverseMap()

    # These first loops run through every single log in all the folders and load the position and velocity
    # information of all the "relevant cars" as specified in the .json file as evidence for the DBN
    for folder in folders:
        argo_loader = load_all_logs(ARGOVERSE_TRACKING.joinpath(folder))
        relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
        for log_id in relevant_cars:
            if relevant_cars[log_id].get("skip", False) is False:
                argo_data = argo_loader.get(log_id)
                end_time = relevant_cars.get(log_id).get("ground_truth")[-1] * 10
                adj_obj_ids = relevant_cars[log_id]["adj_cars"]
                cross_obj_ids = relevant_cars[log_id]["cross_cars"]
                evidence_dict = get_evidence(city_map, argo_data, end_time)
                total_discr_evidence_dict = {}
                pom_evidence_dicts = [{} for t in range(0, (end_time // INTERVAL) + 2)]
                for i in range(len(evidence_dict)):
                    if i in adj_obj_ids or i in cross_obj_ids:
                        discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, INTERVAL, i)
                        for t in discr_evidence_dict:
                            key, value, timestep = convert_pgmpy_pom(t, discr_evidence_dict[t])
                            pom_evidence_dicts[timestep][key] = value
                pom_evidence_dicts.pop(0)

                # Next we make a copy of the evidence dictionaries with YOLO included (ie. not blank)
                evidence_with_yolo = deepcopy(pom_evidence_dicts)
                ft, yolo = parse_yolo(ARGOVERSE_TRACKING.joinpath(f"{folder}/{log_id}/rfc.txt"))
                yolo_evidence = yolo_to_evidence(yolo, ft, INTERVAL)
                for t, e in enumerate(evidence_with_yolo):
                    if t in yolo_evidence:
                        e.update(yolo_evidence[t])

                filepath = Path("params")
                all_evidence = [pom_evidence_dicts, evidence_with_yolo]
                all_predictions = [[], []]  # [aase, aase+yolo]
                timing = [[], []]
                for which_evidence, evidence_list in enumerate(all_evidence):  # 2 experiments
                    dbn, names = init_DBN(filepath, adj_obj_ids, cross_obj_ids)
                    dbn.bake()
                    for i, evidence in enumerate(evidence_list):
                        start = timeit.default_timer()
                        next_belief, out = predict_DBN(dbn, names, evidence, i + 1, iterations=7)
                        all_predictions[which_evidence] = all_predictions[which_evidence] + out
                        dbn, names = iterate_DBN(filepath, adj_obj_ids, cross_obj_ids, next_belief, i + 1)
                        dbn.bake()
                        stop = timeit.default_timer()
                        execution_time = stop - start
                        timing[which_evidence].append(execution_time)

                # Before we write to a CSV we need to run an HMM on the YOLO output
                max_timesteps, yolo_data = read_txt(ARGOVERSE_TRACKING.joinpath(f"{folder}/{log_id}/rfc.txt"))
                yolo_hmm_predictions = yolo_hmm(max_timesteps, yolo_data)
                write_to_csv(folder, log_id, all_predictions, timing, yolo_hmm_predictions)
