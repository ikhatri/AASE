# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import argoverse
import os
import logging
import numpy as np
import argoverse.visualization.visualization_utils as viz_util
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
from matplotlib import pyplot as plt
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader

SAMPLE_DIR = Path('sample-data/')
logger = logging.getLogger(__name__)

def load_all_logs(data_dir: Path):
    return ArgoverseTrackingLoader(SAMPLE_DIR)


def get_trajectories(argoverse_data: ArgoverseTrackingLoader, end: int) -> dict:
    unique_id_list = set()
    for i in range(len(argoverse_data.label_list)):
        for label in argoverse_data.get_label_object(i):
            unique_id_list.add(label.track_id)
    objects = argoverse_data.get_label_object(end)
    visible_track_id = set()
    for obj in objects:
        if obj.occlusion == 100:
            continue
        visible_track_id.add(obj.track_id)
    current_pose = argoverse_data.get_pose(end)
    traj_by_id: Dict[Optional[str], List[Any]] = defaultdict(list)
    for i in range(0, end, 1):
        if current_pose is None:
            logger.warning("`current_pose` is missing at index %d", end)
            break

        pose = argoverse_data.get_pose(i)
        if pose is None:
            logger.warning("`pose` is missing at index %d", i)
            continue

        objects = argoverse_data.get_label_object(i)

        for obj in objects:
            if obj.occlusion == 100:
                continue
            if obj.track_id is None or obj.track_id not in visible_track_id:
                continue
            x, y, z = pose.transform_point_cloud(
                np.array([np.array(obj.translation)]))[0]

            x, y, _ = current_pose.inverse_transform_point_cloud(
                np.array([np.array([x, y, z])]))[0]

            if obj.track_id is None:
                logger.warning(
                    "Label has no track_id.  Collisions with other tracks that are missing IDs could happen")

            traj_by_id[obj.track_id].append([x, y])

    # for track_id in traj_by_id.keys():
    #     traj = np.array(traj_by_id[track_id])
    return traj_by_id


if __name__ == "__main__":
    end_time = 120
    d = load_all_logs(SAMPLE_DIR)
    tjs = get_trajectories(d, end_time)
    pose = d.get_pose(end_time)
    # av_position_x, av_position_y, _ = pose.inverse_transform_point_cloud(np.zeros((1, 3)))[0]
    # av_position_x = 0
    # print('Our position:', av_position_x)
    # for i, track_id in enumerate(tjs.keys()):
    #     traj = np.array(tjs[track_id])
    #     if traj[0,0] > av_position_x-10 and traj[0, 0] < av_position_x + 50 and -10 < traj[0, 1] < av_position_x + 10:
    #         print('Track ID:', track_id)
    #         print('xy coords:', traj[0, 0], traj[0, 1])

    tid = '6d6703d4-85f6-4a60-831a-aa2e10acd1d9'
    traj = np.array(tjs[tid])
    # print(traj)

    f3 = plt.figure(figsize=(15, 15))
    ax3 = f3.add_subplot(111, projection='3d') # current time frame
    viz_util.draw_point_cloud_trajectory(
            ax3,
            'Trajectory',
            d,end_time,axes=[0, 1],xlim3d=(-15,15),ylim3d=(-15,15) # X and Y axes
        )
    ax3.plot3D(np.arange(15),np.zeros(15),np.zeros(15), 'b+')
    ax3.plot3D(np.zeros(15),np.arange(15),np.zeros(15), 'r+')
    ax3.plot3D(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), 'k*')
    plt.axis('off')
    plt.show()
