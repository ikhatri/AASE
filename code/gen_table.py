# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Imports
import csv

import numpy as np
from matplotlib import pyplot as plt

from dataloader import load_relevant_cars
from experiments import RELEVANT_JSON, RESULTS_DIR
from plotting import read_csv

# The purpose of this file is to compute the sequence of outputs from all the various methods and then compare
# if they obtained the correct sequence and if so what is the time delta of the transitions. All methods are sampled at 1 hz.


def get_sequences(folder: str, log_id: str, threshold: float = 0.0) -> tuple:
    relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)

    # Get the sequence from a AASE + YOLO output
    yolo_aase = read_csv(folder, log_id)[1]
    end_time = len(yolo_aase["red"]) - 1
    aase_sequence = []
    for i in range(end_time):
        r = (yolo_aase["red"][i], "R")
        g = (yolo_aase["green"][i], "G")
        y = (yolo_aase["yellow"][i], "Y")
        output = max(r, g, y)[1] if max(r, g, y)[0] > threshold else (0, "N")
        aase_sequence.append(output)
    # print(aase_sequence)

    # Get the true sequence
    gt = relevant_cars.get(log_id).get("ground_truth")
    tmp_gt_sequence = []
    for i in range(0, len(gt), 3):
        tmp_gt_sequence = tmp_gt_sequence + ([gt[i]] * (gt[i + 2] - gt[i + 1]))
    gt_sequence = [tmp_gt_sequence[x * 3 - 3] for x in range(end_time)]
    # print(gt_sequence)

    # Get the sequence from pure YOLO
    yolo = np.genfromtxt(RESULTS_DIR.joinpath(f"{folder}/{log_id}_yolo.csv"), delimiter=",")
    yolo_sequence = []
    max_yolo_length = end_time * 3 if end_time * 3 < len(yolo) else len(yolo)
    for i in range(0, max_yolo_length, 3):
        r = (yolo[i][0], "R")
        g = (yolo[i][1], "G")
        y = (yolo[i][2], "Y")
        output = max(r, g, y)[1] if max(r, g, y)[0] > threshold else (0, "N")
        yolo_sequence.append(output)
    # print(yolo_sequence)

    return gt_sequence, aase_sequence, yolo_sequence


def hamming_distance(gt, aase, yolo):
    hd_aase = 0
    hd_yolo = 0
    # Shifting the ground truth as well for the human delay
    # gt = gt[:-2]
    # We shift by 2 seconds to account for human delay
    # aase = aase[2:]
    for index, light in enumerate(gt):
        if light != aase[index]:
            hd_aase += 1
        if light != yolo[index]:
            hd_yolo += 1
    return hd_aase, hd_yolo


def get_transitions(gt, aase, yolo):
    gt_transitions = []
    aase_transitions = []
    yolo_transitions = []
    for index in range(1, len(gt)):
        if gt[index] != gt[index - 1]:
            gt_transitions.append((gt[index - 1], gt[index], index))
        if aase[index] != aase[index - 1]:
            aase_transitions.append((aase[index - 1], aase[index], index))
        if yolo[index] != yolo[index - 1]:
            yolo_transitions.append((yolo[index - 1], yolo[index], index))
    return gt_transitions, aase_transitions, yolo_transitions


def match_transitions(gt_transitions, aase_transitions, yolo_transitions):
    aase_deltas = []
    yolo_deltas = []
    if len(gt_transitions) == 0:
        return None, None
    for gt in gt_transitions:
        delta = float("inf")
        for at in aase_transitions:
            # If we find a match
            if gt[0] == at[0] and gt[1] == at[1]:
                # Replace the value of delta if it's worse
                delta = abs(gt[2] - at[2]) if abs(gt[2] - at[2]) < delta else delta
        aase_deltas.append(delta)
    # This needs to be two for loops
    for gt in gt_transitions:
        delta = float("inf")
        for yt in yolo_transitions:
            # If we find a match
            if gt[0] == yt[0] and gt[1] == yt[1]:
                # Replace the value of delta if it's worse
                delta = abs(gt[2] - yt[2]) if abs(gt[2] - yt[2]) < delta else delta
        yolo_deltas.append(delta)
    return aase_deltas, yolo_deltas


def compute_runtimes():
    folders = ["train1", "train2", "train3", "train4", "val"]
    aase_runtimes = []
    aase_yolo_runtimes = []
    for folder in folders:
        relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
        for log_id in relevant_cars:
            if relevant_cars[log_id].get("skip", False) is False:
                data = read_csv(folder, log_id)
                aase_runtimes += data[0]["runtime"]
                aase_yolo_runtimes += data[1]["runtime"]
    print("AASE Median Runtime:", np.median(aase_runtimes))
    print("AASE + YOLO Median Runtime:", np.median(aase_yolo_runtimes))
    print("AASE Mean Runtime:", np.mean(aase_runtimes))
    print("AASE + YOLO Mean Runtime:", np.mean(aase_yolo_runtimes))


def plot_runtime_agents():
    folders = ["train1", "train2", "train3", "train4", "val"]
    aase_yolo_runtimes = {}
    for folder in folders:
        relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
        for log_id in relevant_cars:
            if relevant_cars[log_id].get("skip", False) is False:
                data = read_csv(folder, log_id)
                num_agents = len(relevant_cars[log_id].get("adj_cars")) + len(relevant_cars[log_id].get("cross_cars"))
                temp_list = aase_yolo_runtimes.get(num_agents, [])
                aase_yolo_runtimes[num_agents] = temp_list + data[1]["runtime"]
    print(aase_yolo_runtimes)
    plt.figure()
    x = [i for i in aase_yolo_runtimes]
    y = [np.mean(aase_yolo_runtimes.get(i, [0])) for i in x]
    plt.scatter(x, y)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with open(RESULTS_DIR.joinpath(f"table.csv"), "w+", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        folders = ["train1", "train2", "train3", "train4", "val"]
        for folder in folders:
            relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
            for log_id in relevant_cars:
                if relevant_cars[log_id].get("skip", False) is False:
                    g, a, y = get_sequences(folder, log_id)
                    ad, yd = hamming_distance(g, a, y)
                    gt, at, yt = get_transitions(g, a, y)
                    d1, d2 = match_transitions(gt, at, yt)
                    s1 = "N/A" if not d1 else sum(d1)
                    s2 = "N/A" if not d2 else sum(d2)
                    writer.writerow([f"{folder}/{log_id[:3]}", ad, yd, len(g), s1, s2])
