# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Imports
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import plot_styling as ps
from dataloader import load_relevant_cars
from experiments import INTERVAL, RESULTS_DIR, RELEVANT_JSON
from plotting import read_csv

GRAPH_DIR = Path("graphs/10hz")
RED = [0.74, 0.33, 0.33]
GREEN = [0.48, 0.69, 0.41]
YELLOW = [0.86, 0.6, 0.16]
LINEWIDTH = 1.5

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()
ps.setupfig()
ax = plt.gca()
ps.grid()

def interpolate(a: list, v: int=3):
  return np.interp(np.array(range(len(a)*v))/v, range(len(a)),a)

folder = "train1"
log_id = "b1ca08f1-24b0-3c39-ba4e-d5a92868462c"

probs = read_csv(folder, log_id)[1]
yolo_predictions = np.genfromtxt(RESULTS_DIR.joinpath(f"{folder}/{log_id}_yolo.csv"), delimiter=",")

timestep = INTERVAL / 10  # we take every ith entry from 10/second
x_axis = [x * timestep for x in range(len(probs["red"]))]
ax.set_xlim([0, len(x_axis) * (INTERVAL / 10)])
ax.set_ylim([0, 1])

plt.tight_layout()

def animate(i):
  ax.collections.clear()
  xaxis30fps = interpolate(x_axis)
  pr30fps = interpolate(probs["red"])

  r = ax.fill_between(xaxis30fps[:i+1], pr30fps[:i+1])
  r.set_facecolors([RED + [0.3]])
  r.set_edgecolors([RED + [0.75]])
  r.set_linewidths([LINEWIDTH])
  path_red = r.get_paths()[0]
  verts_red = path_red.vertices
  verts_red[:i, 1] = pr30fps[:i]

  pg30fps = interpolate(probs["green"])
  g = ax.fill_between(xaxis30fps[:i+1], pg30fps[:i+1])
  g.set_facecolors([GREEN + [0.3]])
  g.set_edgecolors([GREEN + [0.75]])
  g.set_linewidths([LINEWIDTH])
  path_green = g.get_paths()[0]
  verts_green = path_green.vertices
  verts_green[:i, 1] = pg30fps[:i]

  py30fps = interpolate(probs["yellow"])
  y = ax.fill_between(xaxis30fps[:i+1], py30fps[:i+1])
  y.set_facecolors([YELLOW + [0.3]])
  y.set_edgecolors([YELLOW + [0.75]])
  y.set_linewidths([LINEWIDTH])
  path_yellow = y.get_paths()[0]
  verts_yellow = path_yellow.vertices
  verts_yellow[:i, 1] = py30fps[:i]


def animate_yolo(i):
  ax.collections.clear()
  xaxis30fps = interpolate(x_axis)
  end_time = len(xaxis30fps)

  pr30fps = yolo_predictions[:end_time, 0]
  r = ax.fill_between(xaxis30fps[:i+1], pr30fps[:i+1])
  r.set_facecolors([RED + [0.3]])
  r.set_edgecolors([RED + [0.75]])
  r.set_linewidths([LINEWIDTH])
  path_red = r.get_paths()[0]
  verts_red = path_red.vertices
  verts_red[:i, 1] = pr30fps[:i]

  pg30fps = yolo_predictions[:end_time, 1]
  g = ax.fill_between(xaxis30fps[:i+1], pg30fps[:i+1])
  g.set_facecolors([GREEN + [0.3]])
  g.set_edgecolors([GREEN + [0.75]])
  g.set_linewidths([LINEWIDTH])
  path_green = g.get_paths()[0]
  verts_green = path_green.vertices
  verts_green[:i, 1] = pg30fps[:i]

  py30fps = yolo_predictions[:end_time, 2]
  y = ax.fill_between(xaxis30fps[:i+1], py30fps[:i+1])
  y.set_facecolors([YELLOW + [0.3]])
  y.set_edgecolors([YELLOW + [0.75]])
  y.set_linewidths([LINEWIDTH])
  path_yellow = y.get_paths()[0]
  verts_yellow = path_yellow.vertices
  verts_yellow[:i, 1] = py30fps[:i]

# Lidar frames are 10fps (interpolated to 30fps for the animation)
# Camera frames are 30fps
# They don't line up exactly so there could be a few more camera frames after the end of the last lidar frame
# Thus at 30 fps, we cut off the last 4 frames so everything has the same number of frames
relevant_cars = load_relevant_cars(RELEVANT_JSON, folder)
ranges = relevant_cars.get(log_id).get("ground_truth")
frames = ranges[-1]
anim = animation.FuncAnimation(
    fig, animate_yolo, interval=100/3, frames=frames, repeat=True)

anim.save('basic_animation.mp4', writer=writer)
