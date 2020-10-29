# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import json
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from mayavi import mlab

from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import get_oracle_from_candidate_centerlines
from argoverse.utils.geometry import rotate_polygon_about_pt

SAMPLE_DIR = Path("sample-data/")
GLARE_DIR = Path("glare_example/")
logger = logging.getLogger(__name__)


def load_all_logs(data_dir: Path) -> ArgoverseTrackingLoader:
    return ArgoverseTrackingLoader(data_dir)


def draw_3d_bbox(bbox: np.ndarray, color: tuple = (1, 0, 0)) -> None:
    """A helper function for plotting a bounding box in mayavi mlab

    Args:
        bbox (np.ndarray): A bounding box that is 8 (x,y,z) coordinates of the corners of a bounding box
        color (tuple, optional): The color of the bounding box, an RGB tuple with range 0-1. Defaults to (1, 0, 0).
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
        mlab.plot3d(
            bbox[connection, :1], bbox[connection, 1:2], bbox[connection, 2:3], color=color, tube_radius=None,
        )


def get_relevant_trajectories(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, end: int) -> dict:
    """For the timestep provided, we grab the set of cars currently in/around an intersection only in front of the AV

    Args:
        city_map (ArgoverseMap): an ArgoverseMap object for getting lane segment & traffic control information
        argoverse_data (ArgoverseTrackingLoader): a data loader to access the data
        end (int): the index of the timestep for which to end the returned trajectories,
                   trajectories are returned from the starting time step until this one

    Returns:
        dict: a dictionary with the following structure
                { object/track ID: {
                        timestamp: {
                            position: [x, y, 0]
                            candidate_segments: [int]
                        }}}

    TODO:
        - Figure out if X coordinates for miami are reversed and if so accordingly flip the check on line 114
    """
    # get the current pose of the vehicle
    current_pose = argoverse_data.get_pose(end)

    traj_by_id: Dict[Optional[str], List[Any]] = defaultdict(list)
    data_dict = defaultdict(dict)

    # a list of np array (x, y) coordinates of the trajectory up till i
    # traj_till_now = []

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
            if obj.track_id is None:
                continue

            # get the x, y and z coords of the object with respect to the vehicle at timestep i
            x, y, z = pose.transform_point_cloud(np.array([np.array(obj.translation)]))[0]

            # ensure that the object is in front of the vehicle
            # comparing the vehicle's position at time i with the object's position at time i
            # where time i is a time step between 0 and end
            # note that coords are flipped for different cities
            if x < av_position_x and argoverse_data.city_name == "MIA":
                continue
            if x > av_position_x and argoverse_data.city_name == "PIT":
                continue

            # check if x,y is in a segment not controlled by traffic control
            intersecting_lane_ids = city_map.get_lane_segments_containing_xy(x, y, argoverse_data.city_name)
            traffic_control = False
            if intersecting_lane_ids:
                for lane_id in intersecting_lane_ids:
                    successor_lane_ids = city_map.get_lane_segment_successor_ids(lane_id, argoverse_data.city_name)
                    if successor_lane_ids:
                        for succ_lane in successor_lane_ids:
                            if city_map.lane_has_traffic_control_measure(
                                lane_id, argoverse_data.city_name
                            ) or city_map.lane_has_traffic_control_measure(succ_lane, argoverse_data.city_name):
                                traffic_control = True

            if not traffic_control:
                continue

            # Find the segment that the car is most likely to be following
            data_dict[obj.track_id][i] = {}
            data_dict[obj.track_id][i]["candidate_segments"] = intersecting_lane_ids[:]
            traj_by_id[obj.track_id].append(np.array([x, y]))
            candidate_centerlines = [
                city_map.get_lane_segment_centerline(s, argoverse_data.city_name) for s in intersecting_lane_ids
            ]
            best_fit_centerline = get_oracle_from_candidate_centerlines(
                candidate_centerlines, np.array(traj_by_id[obj.track_id])
            )

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
            if len(traj_by_id[obj.track_id]) > 1:
                vel_vector = np.array([x, y]) - traj_by_id[obj.track_id][-2]
                if np.dot(lane_dir_vector, vel_vector) < 0:
                    p = i - 1
                    while p not in data_dict[obj.track_id] and p >= 0:
                        p -= 1
                    if p > 0:
                        data_dict[obj.track_id][i]["lane_id"] = data_dict[obj.track_id][p]["lane_id"]
                    else:
                        data_dict[obj.track_id][i]["lane_id"] = intersecting_lane_ids[oracle_segment]
                else:
                    data_dict[obj.track_id][i]["lane_id"] = intersecting_lane_ids[oracle_segment]
            else:
                data_dict[obj.track_id][i]["lane_id"] = intersecting_lane_ids[oracle_segment]

            # re-transform the object coords of the object to be with respect to the vehicle at the final timestep which we are plotting
            x, y, _ = current_pose.inverse_transform_point_cloud(np.array([np.array([x, y, z])]))[0]

            data_dict[obj.track_id][i]["position"] = np.array([x, y, 0])

    return data_dict


def visualize(
    argo_maps: ArgoverseMap,
    argo_data: ArgoverseTrackingLoader,
    end_time: int,
    plot_trajectories: bool = True,
    plot_lidar: bool = True,
    plot_bbox: bool = True,
    plot_segments: bool = True,
    show: bool = True,
) -> None:
    """A function to visualize a scene in 3D with mayavi malab

    Args:
        argo_maps (ArgoverseMap): The map object
        argo_data (ArgoverseTrackingLoader): The argoverse dataloader object
        end_time (int): The end time, this is the timestep for which LiDAR is plotted and trajectories are plotted from 0-end
        plot_trajectories (bool, optional): If trajectories should be shown. Defaults to True.
        plot_lidar (bool, optional): If LiDar should be shown. Defaults to True.
        plot_bbox (bool, optional): If bounding boxes should be shown. Defaults to True.
        plot_segments (bool, optional): If lane segments should be shown. Defaults to True.
        show (bool, optional): If mlab.show() should be called at the end of the function. Defaults to True.
    """
    mlab.figure(bgcolor=(0.2, 0.2, 0.2))
    city_to_egovehicle_se3 = argo_data.get_pose(end_time)
    if plot_trajectories or plot_segments or plot_bbox:
        data = get_relevant_trajectories(argo_maps, argo_data, end_time)
    if plot_lidar:
        pc = argo_data.get_lidar(end_time)
        pc = rotate_polygon_about_pt(pc, city_to_egovehicle_se3.rotation, np.zeros(3))
        mlab.points3d(pc[:, :1], pc[:, 1:2], pc[:, 2:3], mode="point")
    if plot_bbox:
        shitty_dict = {}
        for obj in argo_data.get_label_object(end_time):
            if obj.occlusion == 100:
                continue
            if obj.track_id not in data:
                color = (0, 0, 0)
                bbox = rotate_polygon_about_pt(obj.as_3d_bbox(), city_to_egovehicle_se3.rotation, np.zeros(3))
                draw_3d_bbox(bbox, color=color)
        for et in range(0, end_time):
            all_objects = argo_data.get_label_object(et)
            for o in all_objects:
                if o.track_id in data:
                    shitty_dict[o.track_id] = o
        colors = [
            (1, 0, 0),
            (0, 1, 0),
            # (0, 0, 1), # blue, removed
            # (1, 1, 0), # yellow, removed
            (0, 1, 1),
            (1, 0, 1),
            (1, 0.5, 0.5),
            (0.5, 1, 0.5),
            (0.5, 0.5, 1),
            (1, 1, 0.5),
            (0.5, 1, 1),
            (1, 0.5, 1),
        ]
        for i, obj in enumerate(data):
            color = colors[i % len(colors)]
            bbox = rotate_polygon_about_pt(shitty_dict[obj].as_3d_bbox(), city_to_egovehicle_se3.rotation, np.zeros(3),)
            draw_3d_bbox(bbox, color=color)
            mlab.text3d(*bbox[0], f"Object ID: {i}", color=(1, 1, 1), scale=0.5)
    if plot_trajectories:
        for o in data:
            traj = np.array([data[o][k]["position"] for k in data[o].keys()])
            traj = rotate_polygon_about_pt(traj, city_to_egovehicle_se3.rotation, np.zeros(3))
            mlab.points3d(
                traj[:, 0], traj[:, 1], np.zeros((traj.shape[0])), mode="2dcircle", scale_factor=0.5, color=(0, 0, 1),
            )
    if plot_segments:
        segments = set()
        for o in data:
            for t in data[o]:
                segments.add(data[o][t]["lane_id"])
        for l in segments:
            poly = argo_maps.get_lane_segment_polygon(l, argo_data.city_name)
            poly = city_to_egovehicle_se3.inverse_transform_point_cloud(poly)
            poly = rotate_polygon_about_pt(poly, city_to_egovehicle_se3.rotation, np.zeros(3))
            mlab.plot3d(
                poly[:, 0], poly[:, 1], np.zeros(poly.shape[0]), color=(1, 1, 0), tube_radius=None,
            )
    if show:
        mlab.show()


def compute_velocity(data_dict: dict, end_time: int) -> dict:
    """Adds velocity to the data dictionary

    Args:
        data_dict (dict): A dictionary containing the objects and their trajectories
        end_time (int): The final timestep in data dict

    Returns:
        dict: a dictionary with the following structure
                { object/track ID: {
                        timestamp: {
                            position: [x, y, 0]
                            velocity: float
                            candidate_segments: [int]
                        }}}
    """
    for obj in data_dict:
        observed_steps = [x for x in range(end_time) if x in data_dict[obj]]
        for i in range(1, len(observed_steps)):
            t = observed_steps[i]
            p = observed_steps[i - 1]
            distance = np.linalg.norm(data_dict[obj][t]["position"] - data_dict[obj][p]["position"])
            data_dict[obj][t]["velocity"] = distance / (t - p)
        if len(observed_steps) < 2:
            data_dict[obj][observed_steps[0]]["velocity"] = 0
        else:
            data_dict[obj][observed_steps[0]]["velocity"] = data_dict[obj][observed_steps[1]]["velocity"]
    return data_dict


def discretize(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, data_dict: dict) -> dict:
    """
    A function to discretize the data into position segments and the following three velocity classes in (m/0.1s):
        0: zero (0 < v < 0.1)
        1: low (0.1 < v < 0.5)
        2: high (0.5 < v)
    Position segments are defined as follows (with respect to the vehicle itself, not the AV):
        0: at the intersection
        1: turning left
        2: going straight through
        3: turning right

    Args:
        city_map (ArgoverseMap): An argoverse map object
        argoverse_data (ArgoverseTrackingLoader): The argoverse dataloader
        data_dict (dict): A data dictionary with the objects and their position & velocity over time

    Returns:
        dict: { object/track ID: {
                        timestamp: {
                            position: [x, y, 0]
                            velocity: float
                            candidate_segments: [int]
                            discrete_pos: int
                            discrete_vel: int
                            lane_id: int
                        }}}
    """
    for o in data_dict:
        for t in data_dict[o]:
            v = data_dict[o][t]["velocity"]
            if v < 0.1:
                dv = 0
            elif v < 0.5:
                dv = 1
            else:
                dv = 2
            data_dict[o][t]["discrete_vel"] = dv
            lane_id = data_dict[o][t]["lane_id"]
            if not city_map.lane_has_traffic_control_measure(lane_id, argoverse_data.city_name):
                data_dict[o][t]["discrete_pos"] = 0
                continue
            direction = city_map.get_lane_turn_direction(lane_id, argoverse_data.city_name)
            if direction == "LEFT":
                data_dict[o][t]["discrete_pos"] = 1
            if direction == "NONE":
                data_dict[o][t]["discrete_pos"] = 2
            if direction == "RIGHT":
                data_dict[o][t]["discrete_pos"] = 3
    return data_dict


def build_evidence_pgmpy(obj_id: int, data_dict: dict) -> dict:
    """A function for building the evidence dict in the format required by pgmpy

    Args:
        obj_id (int): The id of the object
        data_dict (dict): The data dictionary

    Returns:
        dict: An evidence dictionary of the format:
              {('Position', t): int,
               ('Velocity', t): int}
    """
    evidence_dict = {}
    for t in data_dict[obj_id]:
        evidence_dict[("Position", t)] = data_dict[obj_id][t]["discrete_pos"]
        evidence_dict[("Velocity", t)] = data_dict[obj_id][t]["discrete_vel"]
    return evidence_dict


def build_evidence_pom(obj_ids: list, data_dict: dict, timestep: int) -> dict:
    """A function for building an evidence dictionary in the format required by pomegranate

    Args:
        obj_ids (list): a list of object ids to create evidence dicts for
        data_dict (dict): the full data dict
        timestep (int): the timestep for which to create the evidence dict

    Returns:
        dict: An evidence dictionary of the format:
              {'[obj_id]_evidence_pos_[timestep]': int,
               '[obj_id]_evidence_vel_[timestep]': int}
    """
    evidence_dict = {}
    for obj_id in obj_ids:
        evidence_dict[str(obj_id) + "_evidence_pos_" + str(timestep)] = data_dict[obj_id][timestep]["discrete_pos"]
        evidence_dict[str(obj_id) + "_evidence_vel_" + str(timestep)] = data_dict[obj_id][timestep]["discrete_vel"]
    return evidence_dict


def get_evidence(city_map: ArgoverseMap, argoverse_data: ArgoverseTrackingLoader, end_time: int) -> dict:
    """A function to get all of the evidence for every object from 0-end in the pgmpy format

    Args:
        city_map (ArgoverseMap): The argoverse map objet
        argoverse_data (ArgoverseTrackingLoader): The argoverse data loader
        end_time (int): The end time step

    Returns:
        dict: An evidence dictionary in the pgmpy format
    """
    data = get_relevant_trajectories(city_map, argoverse_data, end_time)
    data = compute_velocity(data, end_time)
    data = discretize(city_map, argoverse_data, data)
    evidence_dict = {}
    for o in data:
        evidence_dict[o] = build_evidence_pgmpy(o, data)
    return evidence_dict


def get_evidence_start_and_finish_for_object(evidence_dict: dict, interval: int, obj_id: int) -> Tuple[int, int]:
    """Retrieves the start and finish timesteps for when an object is visible in a given evidence dict

    Args:
        evidence_dict (dict): An evidence dict
        interval (int): The time subsampling interval used
        obj_id (int): The object id

    Returns:
        Tuple[int, int]: a tuple of (first visible, last visible)
    """
    first_visible_timestep = math.inf
    last_visible_timestep = 0
    for i, o in enumerate(evidence_dict):
        if i == obj_id:
            for t in evidence_dict[o]:
                timestep = t[1] // interval
                if timestep > last_visible_timestep:
                    last_visible_timestep = timestep
                if timestep < first_visible_timestep:
                    first_visible_timestep = timestep
    return first_visible_timestep, last_visible_timestep


def get_discretized_evidence_for_object(
    evidence_dict: dict, interval: int, obj_id: int, up_to: int = None, init_evidence_dict=None,
) -> dict:
    if not init_evidence_dict:
        discr_evidence_dict = {}
    else:
        discr_evidence_dict = init_evidence_dict
    for i, o in enumerate(evidence_dict):
        if i == obj_id:
            # print("evidence from id", i)
            for t in evidence_dict[o]:
                if t[1] % interval != 0:
                    continue
                timestep = t[1] // interval
                if up_to is not None and timestep >= up_to:
                    break
                # print("("+t[0] + str(obj_id) + ", " + str(t[1]+interval)+")", evidence_dict[o][t])
                discr_evidence_dict[(t[0] + "_Evidence_" + str(obj_id), timestep + 1)] = evidence_dict[o][t]
    return discr_evidence_dict


def convert_pgmpy_pom(evidence_key: tuple, evidence_value: int) -> Tuple[str, str, int]:
    """A function to convert pgmpy style evidence dicts to pomegranate compatible ones

    Args:
        evidence_key (tuple): a tuple of ('Position' or 'Velocity', object id)
        evidence_value (int): the integer discretization

    Returns:
        Tuple[str, str, int]: a tuple of the new key, value and the timestep
    """
    label = evidence_key[0]
    obj_id = label.split("_")[-1]
    timestep = evidence_key[1]
    index = evidence_value
    if "Pos" in label:
        new_label = str(obj_id) + "_evidence_pos_" + str(timestep)
        new_value = ["at", "left", "straight", "right"][index]
    else:
        new_label = str(obj_id) + "_evidence_vel_" + str(timestep)
        new_value = ["zero", "low", "med"][index]
    return new_label, new_value, timestep


def parse_yolo(filepath: Path) -> Tuple[int, dict]:
    data = {}
    i = 0
    with open(filepath, "r") as f:
        for l in f:
            if l[0] == "O":
                i += 1
            if l != "":
                out = l.split()
                if len(out) > 0:
                    if out[0] == "Green:" or out[0] == "Red:" or out[0] == "Yellow:":
                        c = re.search("[0-9]+", out[1]).group(0)
                        u = data.get(i, {})
                        if max(u.values(), default=0) < int(c):
                            u.update({out[0][0]: int(c)})
                        data[i] = u
    return i, data


# each sample is kept with fixed probability
def simulate_random_loss(data: dict, final_timestep: int, drop_prob: float = 0.8):
    lossy_data = {}
    for i in range(final_timestep):
        keep_sample = random.random()
        if i in data and keep_sample > drop_prob:
            lossy_data[i] = data[i]
    return lossy_data


def simulate_periodic_obstructions(
    data: dict,
    final_timestep: int,
    obs_drop_prob: float = 0.975,
    obs_lognormal_mean_length: int = 1,
    obs_lognormal_std: float = 0.75,
    unobs_drop_prob: float = 0.4,
    unobs_lognormal_mean_length: int = 1,
    unobs_lognormal_std: float = 0.5,
):
    lossy_data = {}
    obstruction = False
    resample = True
    interval_start = 0

    for i in range(final_timestep):
        # start by generating the next interval of unobstruction followed by obstruction
        if resample:
            sampled_unobs_interval = (
                np.random.lognormal(mean=unobs_lognormal_mean_length, sigma=unobs_lognormal_std) * 30
            )
            sampled_obs_interval = np.random.lognormal(mean=obs_lognormal_mean_length, sigma=obs_lognormal_std) * 30
            print(sampled_unobs_interval)
            print(sampled_obs_interval)
            resample = False
        # if left period of unobstruction, switch to obstructed mode
        if not obstruction and i > interval_start + sampled_unobs_interval:
            obstruction = True
            interval_start = interval_start + sampled_unobs_interval
        # if left period of obstruction, switch to unobstructed mode (no elif as could be same timestep) and resample next timestep
        if obstruction and i > interval_start + sampled_obs_interval:
            obstruction = False
            interval_start = interval_start + sampled_obs_interval
            resample = True
        # set drop probability based on whether or not we are simulating an obstruction
        drop_prob = obs_drop_prob if obstruction else unobs_drop_prob
        keep_sample = random.random()

        if i in data and keep_sample > drop_prob:
            lossy_data[i] = data[i]

    return lossy_data


def yolo_to_evidence(data: dict, final_timestep: int, interval: int):
    new_time = 0
    vision_evidence = {}
    colors = {"G": "green", "R": "red", "Y": "yellow"}
    for i in range(final_timestep):
        if i % 3 == 0 and i % interval == 0:
            new_time += 1
            if i in data:
                color = list(data[i].keys())[0]
                vision_evidence[new_time] = {"vision_" + str(new_time + 1): colors[color]}
    return vision_evidence


def load_relevant_cars(json_file: Path, subfolder: str):
    with open(json_file) as json_data:
        data = json.load(json_data)
    return data[subfolder]


@click.command()
@click.argument("basedir")
@click.argument("logid")
@click.argument("end-time", required=False, default=None, type=int)
def main(basedir: str, logid: str, end_time: int):
    fullpath = Path(f"/home/ikhatri/argoverse/argoverse-api/argoverse-tracking/{basedir}")
    argo_loader = load_all_logs(fullpath)
    argo_data = argo_loader.get(logid)
    print(argo_data)
    if end_time is None:
        end_time = argo_data.num_lidar_frame - 1
    mappymap = ArgoverseMap()
    visualize(mappymap, argo_data, end_time)


if __name__ == "__main__":
    main()
    # interval = 10
    # evidence_dict = get_evidence(mappymap, d, end_time)
    # pom_evidence_dict = {}
    # for i in range(len(evidence_dict)):
    #     discr_evidence_dict = get_discretized_evidence_for_object(evidence_dict, interval, i)
    #     for t in discr_evidence_dict:
    #         key, value = convert_pgmpy_pom(t, discr_evidence_dict[t])
    #         print(key, value)
    #         pom_evidence_dict[key] = value
    # print(pom_evidence_dict)
