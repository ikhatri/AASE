# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Imports
import csv
from pathlib import Path

from matplotlib import pyplot as plt

import plot_styling as ps
from dataloader import load_relevant_cars
from experiments import INTERVAL, RELEVANT_JSON, RESULTS_DIR

# This file contains plotting code for all the results from dictionaries formatted in the default pomegranate output
# Additionally the runtime of each experiment gets saved and can be plotted from here

GRAPH_DIR = Path("graphs/")


def read_csv(folder: str, log: str):
    aase_results = {"red": [], "green": [], "yellow": [], "runtime": []}
    aase_yolo_results = {"red": [], "green": [], "yellow": [], "runtime": []}
    with open(RESULTS_DIR.joinpath(f"{folder}/{log}.csv"), "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(reader):
            if i < 4:
                aase_results[row[0]] = [float(x) for x in row[1:]]
            else:
                aase_yolo_results[row[0]] = [float(x) for x in row[1:]]
    return [aase_results, aase_yolo_results]


def plot_probs(probs: dict, interval: int):
    timestep = 1 / (10 / interval)  # we take every ith entry from 10/second
    x_axis = [x * timestep for x in range(len(probs["red"]))]
    plt.figure()
    ps.setupfig()
    ax = plt.gca()
    ps.grid()
    ax.set_xlim([0, len(x_axis) - 1])
    ax.set_ylim([0, 1])
    r = ax.fill_between(x_axis, probs["red"])
    r.set_facecolors([[0.74, 0.33, 0.33, 0.3]])
    r.set_edgecolors([[0.74, 0.33, 0.33, 0.75]])
    r.set_linewidths([2])

    g = ax.fill_between(x_axis, probs["green"])
    g.set_facecolors([[0.48, 0.69, 0.41, 0.3]])
    g.set_edgecolors([[0.48, 0.69, 0.41, 0.75]])
    g.set_linewidths([2])

    y = ax.fill_between(x_axis, probs["yellow"])
    y.set_facecolors([[0.86, 0.6, 0.16, 0.3]])
    y.set_edgecolors([[0.86, 0.6, 0.16, 0.75]])
    y.set_linewidths([2])


def plot_runtime(times: list):
    plt.figure()
    plt.plot(range(len(times)), times, "b-")
    plt.ylabel("Inference Time (s)")
    plt.xlabel("Timestep")
    plt.ylim(ymin=0)


if __name__ == "__main__":
    folders = ["train1"]  # , "train2", "train3", "train4", "val"]
    for folder in folders:
        relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
        for log_id in relevant_cars:
            if relevant_cars[log_id].get("skip", False) is False:
                results = read_csv(folder, log_id)
                plot_probs(results[0], INTERVAL)
                plt.savefig(GRAPH_DIR.joinpath(f"{folder}/{log_id}_aase.png"))
                plot_runtime(results[0]["runtime"])
                plt.savefig(GRAPH_DIR.joinpath(f"{folder}/{log_id}_aase_runtime.png"))
                plot_probs(results[1], INTERVAL)
                plt.savefig(GRAPH_DIR.joinpath(f"{folder}/{log_id}_aase_yolo.png"))
                plot_runtime(results[1]["runtime"])
                plt.savefig(GRAPH_DIR.joinpath(f"{folder}/{log_id}_aase_yolo_runtime.png"))
