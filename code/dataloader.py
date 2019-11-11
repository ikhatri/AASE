# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import argoverse
import os
import logging
import numpy as np
import argoverse.visualization.visualization_utils as viz_util
import argoverse.visualization.mayavi_utils as mviz_util
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mayavi import mlab
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap

SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('/home/ikhatri/argoverse/argoverse-api/argoverse-tracking/glare_example/')
logger = logging.getLogger(__name__)

def load_all_logs(data_dir: Path) -> ArgoverseTrackingLoader:
    return ArgoverseTrackingLoader(data_dir)


def get_trajectories(argoverse_data: ArgoverseTrackingLoader, end: int) -> dict:
    """
    Documentation here
    """
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

    return traj_by_id

def get_relevant_trajectories(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, end: int) -> dict:
    """
    For every timestep, we grab the set of cars currently in/around an intersection only in front of the AV
    """
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

    lanes_with_traffic = []

    for i in range(0, end, 1):
        if current_pose is None:
            logger.warning("`current_pose` is missing at index %d", end)
            break

        pose = argoverse_data.get_pose(i)
        if pose is None:
            logger.warning("`pose` is missing at index %d", i)
            continue

        av_position_x, _, _ = pose.transform_point_cloud(np.zeros((1, 3)))[0]

        objects = argoverse_data.get_label_object(i)

        for obj in objects:
            if obj.occlusion == 100:
                continue
            if obj.track_id is None or obj.track_id not in visible_track_id:
                continue
            x, y, z = pose.transform_point_cloud(
                np.array([np.array(obj.translation)]))[0]

            # check if x,y is in a segment not controlled by traffic control
            intersecting_lane_ids = city_map.get_lane_segments_containing_xy(x, y, 'PIT')
            traffic_control = False
            for lane_id in intersecting_lane_ids:
                if city_map.lane_has_traffic_control_measure(lane_id, 'PIT'):
                    traffic_control = True
                    lanes_with_traffic.append(lane_id)

            if not traffic_control:
                continue

            if x > av_position_x:
                continue

            x, y, _ = current_pose.inverse_transform_point_cloud(
                np.array([np.array([x, y, z])]))[0]

            if obj.track_id is None:
                logger.warning(
                    "Label has no track_id.  Collisions with other tracks that are missing IDs could happen")

            traj_by_id[obj.track_id].append([x, y, i])

    return traj_by_id, lanes_with_traffic

if __name__ == "__main__":
    end_time = 120
    d = load_all_logs(GLARE_DIR)
    mappymap = ArgoverseMap()
    tjs, lanes = get_relevant_trajectories(mappymap, d, end_time)
    print(len(tjs))
    for t in tjs:
        print(len(tjs[t]))

    # This code below prints out all of the trajectories with their coordinates transformed to be with respect to car coordinates
    # We used this to manually observe the closest car ahead of us and get it's ID
    # pose = d.get_pose(end_time)
    # av_position_x, av_position_y, _ = pose.inverse_transform_point_cloud(np.zeros((1, 3)))[0]
    # av_position_x = 0
    # print('Our position:', av_position_x)
    # for i, track_id in enumerate(tjs.keys()):
    #     traj = np.array(tjs[track_id])
    #     if traj[0,0] > av_position_x-10 and traj[0, 0] < av_position_x + 50 and -10 < traj[0, 1] < av_position_x + 10:
    #         print('Track ID:', track_id)
    #         print('xy coords:', traj[0, 0], traj[0, 1])

    # The manually observed trajectory ID of the car closest to us in the front
    # tid = '6d6703d4-85f6-4a60-831a-aa2e10acd1d9'
    # traj = np.array(tjs[tid])
    # print(traj)

    # Plotting code RIP
    f3 = plt.figure(figsize=(15, 15))
    ax3 = f3.add_subplot(111, projection='3d') # current time frame
    viz_util.draw_point_cloud_trajectory(
            ax3,
            'Trajectory',
            d,end_time,axes=[0, 1],xlim3d=(-15,15),ylim3d=(-15,15) # X and Y axes
        )
    ax3.plot3D(np.arange(15),np.zeros(15),np.zeros(15), 'b+')
    ax3.plot3D(np.zeros(15),np.arange(15),np.zeros(15), 'r+')

    polys = []
    for l in lanes:
        polys.append(mappymap.get_lane_segment_polygon(l, 'PIT'))

    # col = PatchCollection(polys, alpha=0.5)
    # ax3.add_collection(col)
    ax3.add_collection3d(Poly3DCollection(polys))

    # for t in tjs:
    #     traj = np.array(tjs[t])
    #     if len(traj) == 2:
    #         ax3.plot3D(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), 'c*')
    #     elif len(traj) == 33:
    #         ax3.plot3D(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), 'm*')
    #     else:
    #         ax3.plot3D(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), 'g*')


    plt.axis('off')
    plt.show()

    # 3D plotting code RIP x2
    # mappymap = ArgoverseMap()
    # possible_centerlines = mappymap.get_candidate_centerlines_for_traj(traj, 'PIT', viz=True)
    # segments = mappymap.get_lane_segments_containing_xy(*pose.translation[0:2], 'PIT')
    # print(segments)
    # b = mappymap.lane_has_traffic_control_measure(segments[0], 'PIT')
    # print(b)

    # For 3D plotting
    # fig = mlab.figure()
    # print(traj.shape)
    # traj = np.pad(traj, ((0, 0), (0, 1)), 'constant', constant_values=(0,))
    # print(traj.shape)
    # mviz_util.plot_points_3D_mayavi(traj, fig)
    # mlab.show()
