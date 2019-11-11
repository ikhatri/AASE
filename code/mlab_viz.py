# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

# Testing out 3D Open GL Accelerated Rendering

import numpy as np
from mayavi import mlab
from pathlib import Path
import logging
from dataloader import load_all_logs, get_relevant_trajectories
from argoverse.map_representation.map_api import ArgoverseMap

# Ignoring the imports let's make a function to do all the data wrangling below

SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('/home/ikhatri/argoverse/argoverse-api/argoverse-tracking/glare_example/')
logger = logging.getLogger(__name__)

# Get data
end_time = 120
d = load_all_logs(SAMPLE_DIR)
pc = d.get_lidar(end_time)
objects = d.get_label_object(end_time)
mappymap = ArgoverseMap()
tjs, lanes = get_relevant_trajectories(mappymap, d, end_time)

mlab.figure(bgcolor=(0.2, 0.2, 0.2))

for obj in objects:
    if obj.occlusion == 100:
      continue
    box = obj.as_3d_bbox()
    mlab.plot3d(box[:, :1], box[:, 1:2], box[:, 2:3], color=(1, 0, 0))

mlab.points3d(pc[:,:1], pc[:,1:2], pc[:,2:3], mode='point')
for t in tjs:
  traj = np.array(tjs[t])
  mlab.points3d(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), mode='2darrow', color=(0, 0, 1))
mlab.show()
