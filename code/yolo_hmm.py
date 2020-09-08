# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pomegranate import *

import plot_styling as ps

# The output from YOLO comes from command line and is piped into a text file.
# This tool first smoothes out the YOLO output using a HMM model and then plots the output


def load_cpt(filepath: Path, epsilon: float = 0):
    labeled_dataframe = pd.read_csv(filepath)
    labeled_dataframe.set_index("states", inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype=np.float)
    model_weights = (1 - epsilon) * model_weights + np.full_like(model_weights, epsilon)
    return model_weights


def read_txt(filepath: Path) -> tuple:
    """ A function to parse the color and confidence values from a YOLO output .txt file."""
    lines = {}
    timesteps = 0
    with open(filepath, "r") as f:
        for l in f:
            if l[0] == "O":
                timesteps += 1
            if l != "":
                out = l.split()
                if len(out) > 0:
                    if out[0] == "Green:" or out[0] == "Red:" or out[0] == "Yellow:":
                        c = re.search("[0-9]+", l.split()[1]).group(0)
                        lines[timesteps] = {out[0][0]: int(c)}
    return timesteps, lines


def yolo_hmm(timesteps: int, lines: dict) -> np.array:
    """ Smoothes out the YOLO outputs with a HMM."""
    # priors
    # vision_accuracy = load_cpt('params/vision_evidence.csv', epsilon = .15)
    red_emission = DiscreteDistribution({"red": 95.0 / 100, "green": 4.0 / 100, "yellow": 1.0 / 100})
    green_emission = DiscreteDistribution({"red": 4.0 / 100, "green": 95.0 / 100, "yellow": 1.0 / 100})
    yellow_emission = DiscreteDistribution({"red": 4.0 / 100, "green": 1.0 / 100, "yellow": 95.0 / 100})
    vision_accuracy = [red_emission, green_emission, yellow_emission]
    # transition matrix
    trans_mat = load_cpt("params/single_light_model.csv", epsilon=0).T
    initial_prob = np.array([5.0 / 10, 4.0 / 10, 1.0 / 10])
    # model
    model = HiddenMarkovModel.from_matrix(trans_mat, vision_accuracy, initial_prob)
    # yolo data
    r = [lines.get(x, {}).get("R", 0) / 100 for x in range(1, timesteps + 1)]
    g = [lines.get(x, {}).get("G", 0) / 100 for x in range(1, timesteps + 1)]
    y = [lines.get(x, {}).get("Y", 0) / 100 for x in range(1, timesteps + 1)]
    sequence = []
    for idx, _ in enumerate(g):
        if g[idx] > r[idx] and g[idx] > y[idx]:
            sequence.append("green")
        elif r[idx] > g[idx] and r[idx] > y[idx]:
            sequence.append("red")
        elif y[idx] > g[idx] and y[idx] > r[idx]:
            sequence.append("yellow")
        else:
            sequence.append(None)
    # predictions
    preds = []
    for idx in range(1, len(sequence)):
        preds.append(model.predict_proba(sequence[:idx])[-1])
        # print(", ".join(state.name for i, state in model.viterbi(sequence[:idx])[1])[-1])
    return np.array(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", "-f", type=str, help="path to a .txt file with output from YOLO")
    args = parser.parse_args()
    filepath = args.filepath
    max_time, yolo_data = read_txt(Path(filepath))
    preds = yolo_hmm(max_time, yolo_data)
    # plot_yolo_hmm(preds)
