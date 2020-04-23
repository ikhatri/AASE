import plot_styling as ps
import matplotlib.pyplot as plt

if __name__ == "__main__":
  plt.figure()
  ps.setupfig()
  ax = plt.gca()
  ps.grid()
  end_time = 29
  ax.set_xlim([0, end_time])
  ax.set_ylim([0, 1])
  r = ax.fill_between([x for x in range(0, 27)], [1]*27)
  r.set_facecolors([[.74,.33,.33,.3]])
  r.set_edgecolors([[.74,.33,.33,.75]])
  r.set_linewidths([2])

  g = ax.fill_between([x for x in range(26, 30)], [1]*4)
  g.set_facecolors([[.48,.69,.41,.3]])
  g.set_edgecolors([[.48,.69,.41,.75]])
  g.set_linewidths([2])

  # y = ax.fill_between([x/30 for x in range(1, len(preds[:end_time*30])+1)], preds[:end_time*30,2])
  # y.set_facecolors([[.86,.6,.16,.3]])
  # y.set_edgecolors([[.86,.6,.16,.75]])
  # y.set_linewidths([2])
  # plt.title('HMM normalized YOLO predictions')
  # plt.ylabel('probability of light state')
  # plt.xlabel('time in seconds')
  plt.tight_layout()
  plt.show()