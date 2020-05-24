# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from matplotlib import pyplot as plt
from pathlib import Path
import logging
from dataloader import (
    load_all_logs,
    get_evidence,
    get_discretized_evidence_for_object,
    visualize,
    get_evidence_start_and_finish_for_object,
)
from argoverse.map_representation.map_api import ArgoverseMap
from model_pgmpy import (
    setup_traffic_DBN,
    setup_backbone_DBN,
    add_cars_DBN,
    get_inference_model,
)
import math

SAMPLE_DIR = Path("sample-data/")
GLARE_DIR = Path("glare_example/")
PARAMS_DIR = Path("params/")
logger = logging.getLogger(__name__)


def plot_probs(probs: dict, interval: int):
    timestep = 1 / (10 / interval)  # we take every ith entry from 10/second
    x_axis = [x * timestep for x in range(len(probs["r"]))]
    plt.figure()
    plt.plot(x_axis, probs["r"], "r-")
    plt.plot(x_axis, probs["g"], "g-")
    plt.plot(x_axis, probs["y"], "y-")
    plt.xlabel("time in seconds")
    plt.ylabel("probability of light state")
    plt.show()


if __name__ == "__main__":
    end_time = 150
    interval = 10
    adj_obj_ids = [0]
    cross_obj_ids = [9]
    argo_data = load_all_logs(SAMPLE_DIR)
    city_map = ArgoverseMap()
    # visualize(city_map, argo_data, end_time)
    evidence_dict = get_evidence(city_map, argo_data, end_time)
    backbone_dbn, our_light, cross_light = setup_backbone_DBN(PARAMS_DIR)
    dbn = add_cars_DBN(
        PARAMS_DIR, backbone_dbn, our_light, cross_light, adj_obj_ids, cross_obj_ids
    )
    # dbn = setup_traffic_DBN(Path('params'))
    obj_ids = adj_obj_ids + cross_obj_ids
    model = get_inference_model(dbn)
    earliest_evidence = math.inf
    latest_evidence = 0
    for obj_id in obj_ids:
        start, stop = get_evidence_start_and_finish_for_object(
            evidence_dict, interval, obj_id
        )
        if start < earliest_evidence:
            earliest_evidence = start
        if stop > latest_evidence:
            latest_evidence = stop
    plottable_data = {"r": [], "g": [], "y": []}
    for i in range(earliest_evidence + 1, latest_evidence + 2):
        evidence_up_to_present = {}
        for obj_id in obj_ids:
            evidence_up_to_present = get_discretized_evidence_for_object(
                evidence_dict,
                interval,
                obj_id,
                up_to=i,
                init_evidence_dict=evidence_up_to_present,
            )

        results = model.query(
            variables=[("Our Light", i)], evidence=evidence_up_to_present
        )
        for d in results:
            # print(d, results[d])
            plottable_data["r"].append(results[d].values[0])
            plottable_data["g"].append(results[d].values[1])
            plottable_data["y"].append(results[d].values[2])

    plot_probs(plottable_data, interval)
