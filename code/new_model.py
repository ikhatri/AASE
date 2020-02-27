# Copyright 2019-2020 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab

import pandas as pd
import numpy as np
from pathlib import Path
import pomegranate as pom
import timeit

def load_cpt(filepath: Path, epsilon: float=0):
    """
    TODO: Documentation
    """
    labeled_dataframe = pd.read_csv(filepath)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype = np.float)
    model_weights = (1-epsilon)*model_weights + np.full_like(model_weights, epsilon)
    return model_weights

def load_cpt_weights(filepath: Path, epsilon: float=0):
    labeled_dataframe = pd.read_csv(filepath)
    states = list(labeled_dataframe.states.values)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype = np.float)
    model_weights = (1-epsilon)*model_weights + np.full_like(model_weights, epsilon)
    cpt = []
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(list(labeled_dataframe.columns.values)):
            temp = s2.split('|') + [s1, model_weights[i, j]]
            cpt.append(temp)
    return cpt

def add_edges(model, edges):
    for edge in edges:
        model.add_edge(*edge)
    return model

def setup_test_DBN():
    start = timeit.default_timer()
    filepath = Path('params')

    # Load CPTs
    light_base_cpt = load_cpt_weights(filepath.joinpath('single_light_model.csv'), epsilon=.02)
    velocity_base_cpt = load_cpt_weights(filepath.joinpath('velocity_model.csv'), epsilon=.03)
    position_base_cpt = load_cpt_weights(filepath.joinpath('position_model.csv'), epsilon=.01)
    driver_base_cpt = load_cpt_weights(filepath.joinpath('driver_model.csv'), epsilon=.01)
    light_prior = pom.DiscreteDistribution({'red': 1./3, 'green': 1./3, 'yellow': 1./3})
    position_prior = pom.DiscreteDistribution({'at': 1./4, 'left': 1./4, 'straight': 1./4, 'right': 1./4})
    velocity_prior = pom.DiscreteDistribution({'zero': 1./3, 'low': 1./3, 'med': 1./3})

    driver_cpt = pom.ConditionalProbabilityTable(driver_base_cpt, [light_prior, position_prior, velocity_prior])
    light_cpt = pom.ConditionalProbabilityTable(light_base_cpt, [light_prior])
    velocity_cpt = pom.ConditionalProbabilityTable(velocity_base_cpt, [velocity_prior, driver_cpt])
    position_cpt = pom.ConditionalProbabilityTable(position_base_cpt, [driver_cpt, position_prior, velocity_cpt])

    # Time slice 0
    light_node0 = pom.Node(light_prior, name='light_node0')
    position_node0 = pom.Node(position_prior, name='position_node0')
    velocity_node0 = pom.Node(velocity_prior, name='velocity_node0')
    driver_node0 = pom.Node(driver_cpt, name='driver_node0')

    # Time slice 1
    light_node1 = pom.Node(light_cpt, name='light_node1')
    velocity_node1 = pom.Node(velocity_cpt, name='velocity_node1')
    position_node1 = pom.Node(position_cpt, name='position_node1')

    # Add edges
    model = pom.BayesianNetwork('Single Slice Light')
    model.add_states(light_node0, position_node0, velocity_node0, driver_node0,
                     light_node1, position_node1, velocity_node1)
    model = add_edges(model, [(light_node0, light_node1),
                              (light_node0, driver_node0),
                              (position_node0, driver_node0),
                              (velocity_node0, driver_node0),
                              (velocity_node0, position_node1),
                              (velocity_node0, velocity_node1),
                              (position_node0, position_node1),
                              (driver_node0, velocity_node1),
                              (driver_node0, position_node1)])
    model.bake()
    # All the program statements
    stop = timeit.default_timer()
    execution_time = stop - start
    print("Program Executed in "+str(execution_time)) # It returns time in seconds

def build_backbone_DBN_slice(filepath: Path, system_belief: pom.DiscreteDistribution, iter: int = 0):
    """
    TODO: Documentation
    """
    system_weights = load_cpt_weights(filepath.joinpath('system_model.csv'),      epsilon = .01)
    our_weights    = load_cpt_weights(filepath.joinpath('our_light_model.csv'),   epsilon = .005)
    cross_weights  = load_cpt_weights(filepath.joinpath('cross_light_model.csv'), epsilon = .005)
    vision_weights = load_cpt_weights(filepath.joinpath('vision_evidence.csv'),   epsilon = .08)

    system_cpt = pom.ConditionalProbabilityTable(system_weights, [system_belief])
    our_light_cpt = pom.ConditionalProbabilityTable(our_weights, [system_cpt])
    cross_light_cpt = pom.ConditionalProbabilityTable(cross_weights, [system_cpt])
    vision_cpt = pom.ConditionalProbabilityTable(vision_weights, [our_light_cpt])

    names = []
    curr_iter = str(iter)
    next_iter = str(iter+1)

    ps_name = "system_"+curr_iter
    ns_name = "system_"+next_iter
    nol_name = "our_light_" + next_iter
    ncl_name = "cross_light_" + next_iter
    nv_name = "vision_" + next_iter

    prev_system = pom.Node(system_belief, name = ps_name)
    next_system = pom.Node(system_cpt, name = ns_name)
    next_our_light = pom.Node(our_light_cpt, name = nol_name)
    next_cross_light = pom.Node(cross_light_cpt, name = ncl_name)
    next_vision = pom.Node(vision_cpt, name = nv_name)

    model = pom.BayesianNetwork("Intersection")

    model.add_nodes(prev_system, next_system, next_our_light, next_cross_light, next_vision)
    model = add_edges(model, [(prev_system, next_system),
                       (next_system, next_our_light),
                       (next_system, next_cross_light),
                       (next_our_light, next_vision)])
    names.extend([ps_name, ns_name, nol_name, ncl_name, nv_name])
    #we bake the model after adding cars
    return model, next_our_light, next_cross_light, our_light_cpt, cross_light_cpt, names

def setup_car_DBN_slice(filepath: Path, light: pom.ConditionalProbabilityTable, light_belief: pom.DiscreteDistribution, position_belief: pom.DiscreteDistribution, velocity_belief: pom.DiscreteDistribution, car_id: str, iter: int = 0):
    """
    Sets up a single car DBN for a single slice
    Inputs:
        filepath: the path to the folder with the saved model csvs
        light: the CPT for the light which this car is dependent on (either our light or cross light)
        light_belief: the initial (or previous) distribution of the light
        position_belief: the initial (or previous) distribution of the position
        velocity_belief: the initial or previous distribution of the velocity
    """
    velocity_weights = load_cpt_weights(filepath.joinpath('velocity_model.csv'), epsilon = .03)
    position_weights = load_cpt_weights(filepath.joinpath('position_model.csv'), epsilon = .01)
    driver_weights = load_cpt_weights(filepath.joinpath('driver_model.csv'), epsilon = .01)
    evidence_pos_weights = load_cpt_weights(filepath.joinpath('evidence_pos.csv'), epsilon = .05)
    evidence_vel_weights = load_cpt_weights(filepath.joinpath('evidence_vel.csv'), epsilon = .05)

    driver_cpt = pom.ConditionalProbabilityTable(driver_weights, [light_belief, position_belief, velocity_belief])
    velocity_cpt = pom.ConditionalProbabilityTable(velocity_weights, [velocity_belief, driver_cpt])
    position_cpt = pom.ConditionalProbabilityTable(position_weights, [driver_cpt, position_belief, velocity_cpt])
    evidence_pos_cpt = pom.ConditionalProbabilityTable(evidence_pos_weights, [position_cpt])
    evidence_vel_cpt = pom.ConditionalProbabilityTable(evidence_vel_weights, [velocity_cpt])

    names = []
    s_car_id = str(car_id)
    curr_iter = str(iter)
    next_iter = str(iter+1)

    pp_name = s_car_id + "_position_" + curr_iter
    pv_name = s_car_id + "_velocity_" + curr_iter
    nd_name = s_car_id + "_driver_" + next_iter
    nv_name = s_car_id + "_velocity_" + next_iter
    np_name = s_car_id + "_position_" + next_iter
    nep_name = s_car_id + "_evidence_pos_" + next_iter
    nev_name = s_car_id + "_evidence_vel_" + next_iter

    prev_position = pom.Node(position_belief, name = pp_name)
    prev_velocity = pom.Node(velocity_belief, name = pv_name)

    next_driver = pom.Node(driver_cpt, name = nd_name)
    next_velocity = pom.Node(velocity_cpt, name = nv_name)
    next_position = pom.Node(position_cpt, name = np_name)
    next_evidence_pos = pom.Node(evidence_pos_cpt, name = nep_name)
    next_evidence_vel = pom.Node(evidence_vel_cpt, name = nev_name)

    names = [pp_name, pv_name, nd_name, nv_name, np_name, nep_name, nev_name]

    nodes = [prev_position, prev_velocity,
             next_driver, next_velocity, next_position,
             next_evidence_pos, next_evidence_vel]
    edges = [(light, next_driver),
             (prev_position, next_driver),
             (prev_velocity, next_driver),
             (prev_velocity, next_velocity),
             (next_driver, next_velocity),
             (next_driver, next_position),
             (prev_position, next_position),
             (next_velocity, next_position),
             (next_position, next_evidence_pos),
             (next_velocity, next_evidence_vel)]

    return names, nodes, edges

def add_cars_DBN(filepath: Path, backbone_dbn, our_light, cross_light, adj_car_ids, cross_car_ids,
                 our_light_belief, cross_light_belief, car_beliefs, names: list, iter: int = 0):
    for adj_id in adj_car_ids:
        node_names, nodes, edges = setup_car_DBN_slice(filepath, our_light, our_light_belief,
                                           car_beliefs[adj_id]['pos'], car_beliefs[adj_id]['vel'],
                                           adj_id, iter)
        names.extend(node_names)
        backbone_dbn.add_nodes(*nodes)
        backbone_dbn = add_edges(backbone_dbn, edges)

    for cross_id in cross_car_ids:
        node_names, nodes, edges, = setup_car_DBN_slice(filepath, cross_light, cross_light_belief,
                                            car_beliefs[cross_id]['pos'], car_beliefs[cross_id]['vel'],
                                            cross_id, iter)
        names.extend(node_names)
        backbone_dbn.add_nodes(*nodes)
        backbone_dbn = add_edges(backbone_dbn, edges)

    return backbone_dbn, names

def init_DBN(filepath: Path, adj_car_ids: list, cross_car_ids: list):
    system_prior = pom.DiscreteDistribution({'Red_red': 1./10, 'red_Red': 1./10,
                                             'red_green': 3./10, 'red_yellow': 1./10,
                                             'green_red': 3./10, 'yellow_red': 1./10})
    car_beliefs = {}

    for car_id in adj_car_ids + cross_car_ids:
        car_beliefs[car_id] = {}
        car_beliefs[car_id]['pos'] = pom.DiscreteDistribution({'at': 1./4, 'left': 1./4, 'straight': 1./4, 'right': 1./4})
        car_beliefs[car_id]['vel'] = pom.DiscreteDistribution({'zero': 1./3, 'low': 1./3, 'med': 1./3})

    dbn, our_light, cross_light, our_light_belief, cross_light_belief, names = build_backbone_DBN_slice(filepath, system_prior)

    dbn, names = add_cars_DBN(filepath, dbn, our_light, cross_light, adj_car_ids, cross_car_ids, our_light_belief, cross_light_belief, car_beliefs, names)

    return dbn, names

def iterate_DBN(filepath: Path, adj_car_ids: list, cross_car_ids: list, prev_beliefs: dict, iter: int):
    s_iter = str(iter)
    if "system_"+s_iter in prev_beliefs:
        system_prior = pom.DiscreteDistribution(prev_beliefs["system_"+s_iter])
    else:
        system_prior = pom.DiscreteDistribution({'Red_red': 1./10, 'red_Red': 1./10,
                                             'red_green': 3./10, 'red_yellow': 1./10,
                                             'green_red': 3./10, 'yellow_red': 1./10})
    car_beliefs = {}
    for car_id in adj_car_ids + cross_car_ids:
        s_car_id = str(car_id)
        car_beliefs[car_id] = {}
        if s_car_id + "_position_" + s_iter in prev_beliefs:
            car_beliefs[car_id]['pos'] = pom.DiscreteDistribution(prev_beliefs[s_car_id + "_position_" + s_iter])
        else:
            car_beliefs[car_id]['pos'] = pom.DiscreteDistribution({'at': 1./4, 'left': 1./4, 'straight': 1./4, 'right': 1./4})
        if s_car_id + "_velocity_" + s_iter in prev_beliefs:
            car_beliefs[car_id]['vel'] = pom.DiscreteDistribution(prev_beliefs[s_car_id + "_velocity_" + s_iter])
        else:
            car_beliefs[car_id]['vel'] = pom.DiscreteDistribution({'zero': 1./3, 'low': 1./3, 'med': 1./3})
    
    dbn, our_light, cross_light, our_light_belief, cross_light_belief, names = build_backbone_DBN_slice(filepath, system_prior, iter)
    dbn, names = add_cars_DBN(filepath, dbn, our_light, cross_light, adj_car_ids, cross_car_ids, our_light_belief, cross_light_belief, car_beliefs, names, iter)

    return dbn, names

def predict_DBN(dbn, names, evidence, timestep, iterations = 10):
    y_hat = dbn.predict_proba(X = evidence, max_iterations = iterations)
    next_belief = {}
    for i,y in enumerate(y_hat):
        if "system_"+str(timestep) in names[i] or \
           "position_"+str(timestep) in names[i] or \
           "velocity_"+str(timestep) in names[i]:
            next_belief[names[i]] = y.parameters[0]
        if "our_light_"+str(timestep) in names[i]:
            print(y.parameters[0])
    return next_belief
if __name__ == "__main__":
    start = timeit.default_timer()
    filepath = Path('params')
    dbn, names = init_DBN(filepath, [0,1,2], [3,4])
    dbn.bake()
    evidence_1 =  {'0_evidence_pos_1': 'at', '0_evidence_vel_1': 'med',
                                   '1_evidence_pos_1': 'at', '1_evidence_vel_1': 'low',
                                   '2_evidence_pos_1': 'straight', '2_evidence_vel_1': 'zero',
                                   '3_evidence_pos_1': 'at', '3_evidence_vel_1': 'low',
                                   '4_evidence_pos_1': 'at', '4_evidence_vel_1': 'zero'}
    evidence_2 =  {'0_evidence_pos_2': 'at', '0_evidence_vel_2': 'med',
                                   '1_evidence_pos_2': 'at', '1_evidence_vel_2': 'med',
                                   '2_evidence_pos_2': 'straight', '2_evidence_vel_2': 'low',
                                   '3_evidence_pos_2': 'at', '3_evidence_vel_2': 'zero',
                                   '4_evidence_pos_2': 'at', '4_evidence_vel_2': 'zero'}
    
    next_belief = predict_DBN(dbn, names, evidence_1, 1)
    new_dbn, names = iterate_DBN(filepath, [0,1,2], [3,4], next_belief, 1)
    new_dbn.bake()
    next_next_belief = predict_DBN(new_dbn, names, evidence_2, 2)
    stop = timeit.default_timer()
    execution_time = stop - start
    print("Program Executed in "+str(execution_time)) # It returns time in seconds
