# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Imports
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import plot_styling as ps
from dataloader import load_relevant_cars
from experiments import ARGOVERSE_TRACKING, INTERVAL, RELEVANT_JSON, RESULTS_DIR
from yolo_hmm import read_txt, yolo_hmm

# This file contains plotting code for all the results from dictionaries formatted in the default pomegranate output
# Additionally the runtime of each experiment gets saved and can be plotted from here

GRAPH_DIR = Path("graphs/")
RED = [0.74, 0.33, 0.33]
GREEN = [0.48, 0.69, 0.41]
YELLOW = [0.86, 0.6, 0.16]


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
    r.set_facecolors([RED + [0.3]])
    r.set_edgecolors([RED + [0.75]])
    r.set_linewidths([2])

    g = ax.fill_between(x_axis, probs["green"])
    g.set_facecolors([GREEN + [0.3]])
    g.set_edgecolors([GREEN + [0.75]])
    g.set_linewidths([2])

    y = ax.fill_between(x_axis, probs["yellow"])
    y.set_facecolors([YELLOW + [0.3]])
    y.set_edgecolors([YELLOW + [0.75]])
    y.set_linewidths([2])
    plt.tight_layout()


def plot_runtime(times: list):
    plt.figure()
    plt.plot(range(len(times)), times, "b-")
    plt.ylabel("Inference Time (s)")
    plt.xlabel("Timestep")
    plt.ylim(ymin=0)


def plot_ground_truth(relevant_json: dict, log: str):
    ranges = relevant_json.get(log).get("ground_truth")
    end_time = ranges[-1]
    plt.figure()
    ps.setupfig()
    ax = plt.gca()
    ps.grid()
    ax.set_xlim([0, end_time])
    ax.set_ylim([0, 1])
    for i in range(0, len(ranges), 3):
        if ranges[i] is "G":
            color = GREEN
        elif ranges[i] is "R":
            color = RED
        else:
            color = YELLOW
        fill = ax.fill_between(
            [x for x in range(ranges[i + 1], ranges[i + 2] + 1)], [1] * (ranges[i + 2] - ranges[i + 1] + 1)
        )
        fill.set_facecolors([color + [0.3]])
        fill.set_edgecolors([color + [0.75]])
        fill.set_linewidths([2])
    plt.tight_layout()


def plot_yolo_hmm(preds: np.array, relevant_json: dict, log: str) -> None:
    ranges = relevant_json.get(log).get("ground_truth")
    end_time = ranges[-1]
    plt.figure()
    ps.setupfig()
    ax = plt.gca()
    ps.grid()
    ax.set_xlim([0, end_time])
    ax.set_ylim([0, 1])
    r = ax.fill_between([x / 30 for x in range(1, len(preds[: end_time * 30]) + 1)], preds[: end_time * 30, 0],)
    r.set_facecolors([RED + [0.3]])
    r.set_edgecolors([RED + [0.75]])
    r.set_linewidths([2])

    g = ax.fill_between([x / 30 for x in range(1, len(preds[: end_time * 30]) + 1)], preds[: end_time * 30, 1],)
    g.set_facecolors([GREEN + [0.3]])
    g.set_edgecolors([GREEN + [0.75]])
    g.set_linewidths([2])

    y = ax.fill_between([x / 30 for x in range(1, len(preds[: end_time * 30]) + 1)], preds[: end_time * 30, 2],)
    y.set_facecolors([YELLOW + [0.3]])
    y.set_edgecolors([YELLOW + [0.75]])
    y.set_linewidths([2])
    plt.tight_layout()


if __name__ == "__main__":
    folders = ["train1", "train2", "train3", "train4", "val"]
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
                plot_ground_truth(relevant_cars, log_id)
                plt.savefig(GRAPH_DIR.joinpath(f"{folder}/{log_id}_ground_truth.png"))
                # Plotting the YOLO only output, smoothed by an HMM
                yolo_predictions = np.genfromtxt(RESULTS_DIR.joinpath(f"{folder}/{log_id}_yolo.csv"), delimiter=",")
                plot_yolo_hmm(yolo_predictions, relevant_cars, log_id)
                plt.savefig(GRAPH_DIR.joinpath(f"{folder}/{log_id}_yolo_hmm.png"))
