# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from matplotlib import pyplot as plt
from pathlib import Path
import logging
from dataloader import *
from argoverse.map_representation.map_api import ArgoverseMap
from model_pom import *
import math
import re
import pomegranate as pom
import plot_styling as ps

SAMPLE_DIR = Path("sample-data/")
GLARE_DIR = Path("glare_example/")
ARGOVERSE_TRACKING = Path("/home/ikhatri/argoverse/argoverse-api/argoverse-tracking")
PARAMS_DIR = Path("params/")
logger = logging.getLogger(__name__)


def plot_probs(probs: dict, interval: int):
    # Reconfiguring the pom output dict to be plottable
    plottable = {"red": [], "green": [], "yellow": []}
    for t in range(len(probs)):
        for k in plottable:
            plottable[k].append(probs[t][k])
    timestep = 1 / (10 / interval)  # we take every ith entry from 10/second
    x_axis = [x * timestep for x in range(len(plottable["red"]))]
    plt.figure()
    ps.setupfig()
    ax = plt.gca()
    ps.grid()
    ax.set_xlim([0, 29])
    ax.set_ylim([0, 1])
    r = ax.fill_between(x_axis, plottable["red"])
    r.set_facecolors([[0.74, 0.33, 0.33, 0.3]])
    r.set_edgecolors([[0.74, 0.33, 0.33, 0.75]])
    r.set_linewidths([2])

    g = ax.fill_between(x_axis, plottable["green"])
    g.set_facecolors([[0.48, 0.69, 0.41, 0.3]])
    g.set_edgecolors([[0.48, 0.69, 0.41, 0.75]])
    g.set_linewidths([2])

    y = ax.fill_between(x_axis, plottable["yellow"])
    y.set_facecolors([[0.86, 0.6, 0.16, 0.3]])
    y.set_edgecolors([[0.86, 0.6, 0.16, 0.75]])
    y.set_linewidths([2])
    # plt.xlabel('time in seconds')
    # plt.ylabel('probability of light state')


def plot_runtime(times: list):
    plt.figure()
    plt.plot(range(len(times)), times, "b-")
    plt.ylabel("Inference Time")
    plt.xlabel("Timestep")
    plt.ylim(ymin=0)


if __name__ == "__main__":
    print("Using GPU?", pom.utils.is_gpu_enabled())
    interval = 10  # out of 10 hz, so it's every 5th image of the 10/second that we have
    adj_obj_ids = [9, 11]
    cross_obj_ids = [5, 6, 7, 8, 10, 15, 19, 23]
    log_id = "b1ca08f1-24b0-3c39-ba4e-d5a92868462c"
    argo_data = load_all_logs(ARGOVERSE_TRACKING.joinpath("train1")).get(log_id)
    end_time = 299
    print(argo_data)
    print("City Code: {argo_data.city_name}")
    city_map = ArgoverseMap()
    # visualize(city_map, argo_data, end_time, obj_ids=[3, 4])

    evidence_dict = get_evidence(city_map, argo_data, end_time)
    total_discr_evidence_dict = {}
    pom_evidence_dicts = [{} for t in range(0, (end_time // interval) + 2)]
    for i in range(len(evidence_dict)):
        if i in adj_obj_ids or i in cross_obj_ids:
            discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, interval, i)
            for t in discr_evidence_dict:
                key, value, timestep = convert_pgmpy_pom(t, discr_evidence_dict[t])
                pom_evidence_dicts[timestep][key] = value
    pom_evidence_dicts.pop(0)

    ft, yolo = parse_yolo(ARGOVERSE_TRACKING.joinpath("train1/" + log_id + "/rfc.txt"))
    yolo_evidence = yolo_to_evidence(yolo, ft, interval)
    for t, e in enumerate(pom_evidence_dicts):
        if t in yolo_evidence:
            e.update(yolo_evidence[t])

    filepath = Path("params")
    dbn, names = init_DBN(filepath, adj_obj_ids, cross_obj_ids)
    dbn.bake()
    pom_out = []
    timing = []
    for i, evidence in enumerate(pom_evidence_dicts):
        start = timeit.default_timer()
        next_belief, out = predict_DBN(dbn, names, evidence, i + 1, iterations=7)
        pom_out = pom_out + out
        dbn, names = iterate_DBN(filepath, adj_obj_ids, cross_obj_ids, next_belief, i + 1)
        dbn.bake()
        stop = timeit.default_timer()
        execution_time = stop - start
        timing.append(execution_time)

    plot_probs(pom_out, interval)
    # plot_runtime(timing)
    plt.tight_layout()
    plt.show()
