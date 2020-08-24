# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Imports
from matplotlib import pyplot as plt

import plot_styling as ps

# This file contains plotting code for all the results from dictionaries formatted in the default pomegranate output
# Additionally the runtime of each experiment gets saved and can be plotted from here

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
    ax.set_xlim([0, 14])
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


def plot_runtime(times: list):
    plt.figure()
    plt.plot(range(len(times)), times, "b-")
    plt.ylabel("Inference Time")
    plt.xlabel("Timestep")
    plt.ylim(ymin=0)
