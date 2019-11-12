# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import argoverse
import os
import logging
import numpy as np
from mayavi import mlab
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.geometry import rotate_polygon_about_pt

SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('/home/ikhatri/argoverse/argoverse-api/argoverse-tracking/glare_example/')
logger = logging.getLogger(__name__)

def load_all_logs(data_dir: Path) -> ArgoverseTrackingLoader:
    return ArgoverseTrackingLoader(data_dir)

def draw_3d_bbox(bbox: np.ndarray) -> None:
    """
    A dumb helper function to quickly plot a 3D bounding box in mayavi mlab
    """
    connections = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # Connections between upper and lower planes
    ]
    for connection in connections:
        mlab.plot3d(bbox[connection, :1], bbox[connection, 1:2], bbox[connection, 2:3], color=(1, 0, 0), tube_radius=None)

def get_relevant_trajectories(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, end: int) -> dict:
    """
    For the timestep provided, we grab the set of cars currently in/around an intersection only in front of the AV

    Args:
        city_map: an ArgoverseMap class for getting lane segment & traffic control information
        argoverse_data: a data loader to access the data
        end: the index of the timestep for which to end the returned trajectories
             trajectories are returned from the starting time step until this one
    """
    # a set of the object IDs, must be unique
    unique_id_list = set()
    # loop through all the objects and get the track id for each one
    for i in range(len(argoverse_data.label_list)):
        for label in argoverse_data.get_label_object(i):
            unique_id_list.add(label.track_id)

    # get a list of all the objects in the final time step
    objects = argoverse_data.get_label_object(end)

    # make a subset of the unique id list that contains only visible tracks
    visible_track_id = set()
    for obj in objects:
        if obj.occlusion == 100:
            continue
        visible_track_id.add(obj.track_id)

    # get the current pose of the vehicle
    current_pose = argoverse_data.get_pose(end)

    traj_by_id: Dict[Optional[str], List[Any]] = defaultdict(list)

    # a list of lane ids that are controlled by traffic control
    lanes_with_traffic = []

    for i in range(0, end, 1):
        # Checks for valid pose
        if current_pose is None:
            logger.warning("`current_pose` is missing at index %d", end)
            break
        pose = argoverse_data.get_pose(i)
        if pose is None:
            logger.warning("`pose` is missing at index %d", i)
            continue

        # get the x coordinate of the AV's position
        av_position_x, _, _ = pose.transform_point_cloud(np.zeros((1, 3)))[0]

        # get a list of objects present in the current time step
        objects = argoverse_data.get_label_object(i)

        for obj in objects:
            # ignore occluded objects
            if obj.occlusion == 100:
                continue
            if obj.track_id is None or obj.track_id not in visible_track_id:
                continue

            # get the x, y and z coords of the object with respect to the vehicle at timestep i
            x, y, z = pose.transform_point_cloud(
                np.array([np.array(obj.translation)]))[0]

            # check if x,y is in a segment not controlled by traffic control
            intersecting_lane_ids = city_map.get_lane_segments_containing_xy(x, y, argoverse_data.city_name)
            traffic_control = False
            for lane_id in intersecting_lane_ids:
                if city_map.lane_has_traffic_control_measure(lane_id, argoverse_data.city_name):
                    traffic_control = True
                    lanes_with_traffic.append(lane_id)

            if not traffic_control:
                continue

            # ensure that the object is in front of the vehicle
            # comparing the vehicle's position at time i with the object's position at time i
            # where time i is a time step between 0 and end
            if x > av_position_x:
                continue

            # re-transform the object coords of the object to be with respect to the vehicle at the final timestep which we are plotting
            x, y, _ = current_pose.inverse_transform_point_cloud(
                np.array([np.array([x, y, z])]))[0]

            if obj.track_id is None:
                logger.warning(
                    "Label has no track_id.  Collisions with other tracks that are missing IDs could happen")

            traj_by_id[obj.track_id].append([x, y, i])

    return traj_by_id, lanes_with_traffic

def visualize(argo_data: ArgoverseTrackingLoader, argo_maps: ArgoverseMap, end_time: int, plot_trajectories: bool=True, plot_lidar: bool=True, plot_bbox: bool=True, plot_segments: bool=True, show: bool=True) -> None:
    """
    A function to vizualize everything in GPU accelerated 3D

    Arguments:
        argo_data: An argoverse tracking dataloader initialized to the directory of the logs you want
        argo_maps: An argoverse map object
        end_time: The final timestep to consider, everything from time 0 up to end_time will be included in the trajectories
    """
    mlab.figure(bgcolor=(0.2, 0.2, 0.2))
    city_to_egovehicle_se3 = argo_data.get_pose(end_time)
    if plot_trajectories or plot_segments:
        tjs, lanes = get_relevant_trajectories(mappymap, d, end_time)
    if plot_lidar:
        pc = argo_data.get_lidar(end_time)
        pc = rotate_polygon_about_pt(pc, city_to_egovehicle_se3.rotation, np.zeros(3))
        mlab.points3d(pc[:,:1], pc[:,1:2], pc[:,2:3], mode='point')
    if plot_bbox:
        for obj in argo_data.get_label_object(end_time):
            if obj.occlusion == 100:
                continue
            bbox = rotate_polygon_about_pt(obj.as_3d_bbox(), city_to_egovehicle_se3.rotation, np.zeros(3))
            draw_3d_bbox(bbox)
    if plot_trajectories:
        for t in tjs:
            traj = np.array(tjs[t])
            traj = rotate_polygon_about_pt(traj, city_to_egovehicle_se3.rotation, np.zeros(3))
            mlab.points3d(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), mode='2dcircle', scale_factor=0.5, color=(0, 0, 1))
    if plot_segments:
        for l in lanes:
            poly = mappymap.get_lane_segment_polygon(l, argo_data.city_name)
            poly = city_to_egovehicle_se3.inverse_transform_point_cloud(poly)
            poly = rotate_polygon_about_pt(poly, city_to_egovehicle_se3.rotation, np.zeros(3))
            mlab.plot3d(poly[:, 0], poly[:, 1], np.zeros(poly.shape[0]), color=(0, 1, 0), tube_radius=None)
    if show:
        mlab.show()

if __name__ == "__main__":
    end_time = 60
    d = load_all_logs(GLARE_DIR)
    mappymap = ArgoverseMap()
    visualize(d, mappymap, end_time)
