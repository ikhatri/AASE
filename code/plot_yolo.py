from matplotlib import pyplot as plt
from pathlib import Path
import re
import argparse

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

  plt.figure()
  plt.plot([x/30 for x in range(1, i+1)], [lines.get(x, {}).get('G', 0)/100 for x in range(1, i+1)], 'g-')
  plt.plot([x/30 for x in range(1, i+1)], [lines.get(x, {}).get('R', 0)/100 for x in range(1, i+1)], 'r-')
  plt.plot([x/30 for x in range(1, i+1)], [lines.get(x, {}).get('Y', 0)/100 for x in range(1, i+1)], 'y-')
  plt.ylabel('probability of light state')
  plt.xlabel('time in seconds')
  plt.show()
