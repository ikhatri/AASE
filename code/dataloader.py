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
from argoverse.utils.centerline_utils import get_oracle_from_candidate_centerlines

SAMPLE_DIR = Path('sample-data/')
GLARE_DIR = Path('glare_example/')
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
        city map: an ArgoverseMap class for getting lane segment & traffic control information
        argoverse_data: a data loader to access the data
        end: the index of the timestep for which to end the returned trajectories
             trajectories are returned from the starting time step until this one
    Returns:
        data dict: a dictionary with the following structure
        { object/track ID: {
                timestamp: {
                    position: [x, y, 0]
                    candidate_segments: [int]
                    discrete_pos: int
                }
            }
        }
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

    # traj_by_id: Dict[Optional[str], List[Any]] = defaultdict(list)
    data_dict = defaultdict(dict)

    # a list of np array (x, y) coordinates of the trajectory up till i
    traj_till_now = []

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
            if obj.track_id is None: # or obj.track_id not in visible_track_id:
                continue

            # get the x, y and z coords of the object with respect to the vehicle at timestep i
            x, y, z = pose.transform_point_cloud(
                np.array([np.array(obj.translation)]))[0]

            # ensure that the object is in front of the vehicle
            # comparing the vehicle's position at time i with the object's position at time i
            # where time i is a time step between 0 and end
            if x > av_position_x:
                continue

            # check if x,y is in a segment not controlled by traffic control
            intersecting_lane_ids = city_map.get_lane_segments_containing_xy(x, y, argoverse_data.city_name)
            traffic_control = False
            for lane_id in intersecting_lane_ids:
                successor_lane_ids = city_map.get_lane_segment_successor_ids(lane_id, argoverse_data.city_name)
                for succ_lane in successor_lane_ids:
                    if city_map.lane_has_traffic_control_measure(lane_id, argoverse_data.city_name) or city_map.lane_has_traffic_control_measure(succ_lane, argoverse_data.city_name):
                        traffic_control = True

            if not traffic_control:
                continue

            # Find the segment that the car is most likely to be following
            data_dict[obj.track_id][i] = {}
            data_dict[obj.track_id][i]['candidate_segments'] = intersecting_lane_ids[:]
            traj_till_now.append(np.array([x, y]))
            candidate_centerlines = [city_map.get_lane_segment_centerline(s, argoverse_data.city_name) for s in intersecting_lane_ids]
            best_fit_centerline = get_oracle_from_candidate_centerlines(candidate_centerlines, np.array(traj_till_now))

            oracle_segment = 0
            for k, c in enumerate(candidate_centerlines):
                if np.array_equal(best_fit_centerline, c):
                    oracle_segment = k
                    break

            # If the car leaves the segment it was following and drifts into an adjacent lane
            # the above algorithm will break, in this case we simply default to the prevous correct segment
            lane_dir_vector = city_map.get_lane_direction(np.array([x, y]), argoverse_data.city_name)
            lane_dir_vector = np.array([lane_dir_vector[0][0], lane_dir_vector[0][1]])

            # The check is done by computing the angle between
            # the car's velocity and the direction of the lane
            if i > 1:
                vel_vector = np.array([x, y])-traj_till_now[-2]
                if np.dot(lane_dir_vector, vel_vector) < 0:
                    p = i-1
                    while p not in data_dict[obj.track_id] and p >= 0:
                        p-=1
                    if p > 0:
                        data_dict[obj.track_id][i]['lane_id'] = data_dict[obj.track_id][p]['lane_id']
                    else:
                        data_dict[obj.track_id][i]['lane_id'] = intersecting_lane_ids[oracle_segment]
                else:
                    data_dict[obj.track_id][i]['lane_id'] = intersecting_lane_ids[oracle_segment]
            else:
                data_dict[obj.track_id][i]['lane_id'] = intersecting_lane_ids[oracle_segment]


            # re-transform the object coords of the object to be with respect to the vehicle at the final timestep which we are plotting
            x, y, _ = current_pose.inverse_transform_point_cloud(
                np.array([np.array([x, y, z])]))[0]

            data_dict[obj.track_id][i]['position'] = np.array([x, y, 0])

    return data_dict

def visualize(argo_maps: ArgoverseMap, argo_data: ArgoverseTrackingLoader, end_time: int, plot_trajectories: bool=True, plot_lidar: bool=True, plot_bbox: bool=True, plot_segments: bool=True, show: bool=True) -> None:
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
        data = get_relevant_trajectories(argo_maps, argo_data, end_time)
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
        for o in data:
            traj = np.array([data[o][k]['position'] for k in data[o].keys()])
            traj = rotate_polygon_about_pt(traj, city_to_egovehicle_se3.rotation, np.zeros(3))
            mlab.points3d(traj[:,0], traj[:,1], np.zeros((traj.shape[0])), mode='2dcircle', scale_factor=0.5, color=(0, 0, 1))
    if plot_segments:
        segments = set()
        for o in data:
            for t in data[o]:
                segments.add(data[o][t]['lane_id'])
        for l in segments:
            poly = argo_maps.get_lane_segment_polygon(l, argo_data.city_name)
            poly = city_to_egovehicle_se3.inverse_transform_point_cloud(poly)
            poly = rotate_polygon_about_pt(poly, city_to_egovehicle_se3.rotation, np.zeros(3))
            mlab.plot3d(poly[:, 0], poly[:, 1], np.zeros(poly.shape[0]), color=(1, 1, 0), tube_radius=None)
    if show:
        mlab.show()

def compute_velocity(data_dict: dict, end_time: int) -> dict:
    """
    Adds velocity to the data dictionary
    """
    for obj in data_dict:
        observed_steps = [x for x in range(end_time) if x in data_dict[obj]]
        for i in range(1, len(observed_steps)):
            t = observed_steps[i]
            p = observed_steps[i-1]
            distance = np.linalg.norm(data_dict[obj][t]['position'] - data_dict[obj][p]['position'])
            data_dict[obj][t]['velocity'] = distance / (t-p)
        if len(observed_steps) < 2:
            data_dict[obj][observed_steps[0]]['velocity'] = 0
        else:
            data_dict[obj][observed_steps[0]]['velocity'] = data_dict[obj][observed_steps[1]]['velocity']
    return data_dict

def discretize(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, data_dict: dict) -> dict:
    """
    A function to discretize the data into position segments and the following three velocity classes:
        0: zero (0 < v < 0.1)
        1: low (0.1 < v < 0.5)
        2: high (0.5 < v)
    Position segments are defined as follows (with respect to the vehicle itself, not the AV):
        0: at the intersection
        1: turning left
        2: going straight through
        3: turning right
    """
    for o in data_dict:
        for t in data_dict[o]:
            v = data_dict[o][t]['velocity']
            if v < 0.1:
                dv = 0
            elif v < 0.5:
                dv = 1
            else:
                dv = 2
            data_dict[o][t]['discrete_vel'] = dv
            lane_id = data_dict[o][t]['lane_id']
            if not city_map.lane_has_traffic_control_measure(lane_id, argoverse_data.city_name):
                data_dict[o][t]['discrete_pos'] = 0
                continue
            direction = city_map.get_lane_turn_direction(lane_id, argoverse_data.city_name)
            if direction == 'LEFT':
                data_dict[o][t]['discrete_pos']  = 1
            if direction == 'NONE':
                data_dict[o][t]['discrete_pos']  = 2
            if direction == 'RIGHT':
                data_dict[o][t]['discrete_pos']  = 3
    return data_dict
def build_evidence(obj_id: int, data_dict: dict) -> dict:
    evidence_dict = {}
    for t in data_dict[obj_id]:
        evidence_dict[('Position', t)] = data_dict[obj_id][t]['discrete_pos']
        evidence_dict[('Velocity', t)] = data_dict[obj_id][t]['discrete_vel']
    return evidence_dict

def get_evidence(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, end_time: int):
    data = get_relevant_trajectories(city_map, argoverse_data, end_time)
    data = compute_velocity(data, end_time)
    data = discretize(city_map, argoverse_data, data)
    evidence_dict = {}
    for o in data:
        evidence_dict[o] = build_evidence(o, data)
    return evidence_dict
def get_discretized_evidence_for_object(evidence_dict: dict, interval: int, obj_id: int):
    discr_evidence_dict = {}
    for i, o in enumerate(evidence_dict):
        if i == obj_id:
            for t in evidence_dict[o]:
                discr_evidence_dict[(t[0],t[1]//interval)] = evidence_dict[o][t]
    return discr_evidence_dict
if __name__ == "__main__":
    end_time = 150
    interval = 10
    d = load_all_logs(GLARE_DIR)
    mappymap = ArgoverseMap()
    # visualize(mappymap, d, end_time)
    evidence_dict = get_evidence(mappymap, d, end_time)
    discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, 10, 2)
    for t in discr_evidence_dict:
        print(t, discr_evidence_dict[t])
    
    

