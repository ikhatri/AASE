# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

from pomegranate import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import re
import argparse

def load_cpt(filepath: Path, epsilon: float=0):
    labeled_dataframe = pd.read_csv(filepath)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype = np.float)
    model_weights = (1-epsilon)*model_weights + np.full_like(model_weights, epsilon)
    return model_weights

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--filepath', '-f', type=str, help='path to a .txt file with output from YOLO')
  args = parser.parse_args()
  filepath = args.filepath
  lines = {}
  i = 0
  with open(filepath, 'r') as f:
    for l in f:
      if l[0] == 'O':
        i += 1
      if l != '':
        out = l.split()
        if len(out) > 0:
          if out[0] == 'Green:' or out[0] == 'Red:' or out[0] == 'Yellow:':
            c = re.search('[0-9]+', l.split()[1]).group(0)
            lines[i] = {out[0][0]: int(c)}

  # priors
  # vision_accuracy = load_cpt('params/vision_evidence.csv', epsilon = .15)
  red_emission = DiscreteDistribution({'red': 95./100, 'green': 4./100, 'yellow': 1./100})
  green_emission = DiscreteDistribution({'red': 4./100, 'green': 95./100, 'yellow': 1./100})
  yellow_emission = DiscreteDistribution({'red': 4./100, 'green': 1./100, 'yellow': 95./100})
  vision_accuracy = [red_emission, green_emission, yellow_emission]
  # transition matrix
  trans_mat = load_cpt('params/single_light_model.csv', epsilon=0).T
  initial_prob = np.array([5./10, 4./10, 1./10])
  # model
  model = HiddenMarkovModel.from_matrix(trans_mat, vision_accuracy, initial_prob)
  # yolo data
  r = [lines.get(x, {}).get('R', 0)/100 for x in range(1, i+1)]
  g = [lines.get(x, {}).get('G', 0)/100 for x in range(1, i+1)]
  y = [lines.get(x, {}).get('Y', 0)/100 for x in range(1, i+1)]
  sequence = []
  for idx, _ in enumerate(g):
    if g[idx] > r[idx] and g[idx] > y[idx]:
      sequence.append('green')
    elif r[idx] > g[idx] and r[idx] > y[idx]:
      sequence.append('red')
    elif y[idx] > g[idx] and y[idx] > r[idx]:
      sequence.append('yellow')
    else:
      sequence.append(None)
  # predictions
  preds = []
  for idx in range(1, len(sequence)):
    preds.append(model.predict_proba(sequence[:idx])[-1])
    # print(", ".join(state.name for i, state in model.viterbi(sequence[:idx])[1])[-1])

  preds = np.array(preds)

  plt.figure()
  plt.plot([x/30 for x in range(1, len(preds)+1)], preds[:,0], 'r-')
  plt.plot([x/30 for x in range(1, len(preds)+1)], preds[:,1], 'g-')
  plt.plot([x/30 for x in range(1, len(preds)+1)], preds[:,2], 'y-')
  plt.title('HMM normalized YOLO predictions')
  plt.ylabel('probability of light state')
  plt.xlabel('time in seconds')
  plt.show()
