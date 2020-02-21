# Copyright 2019 ikhatri@umass.edu, sparr@umass.edu
# College of Information and Computer Sciences,
# University of Massachusetts Amherst
# Resource-Bounded Reasoning Lab
import pandas as pd
import numpy as np
from pathlib import Path
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD as cpd
from pgmpy.inference.dbn_inference import DBNInference
from pgmpy.inference.ExactInference import VariableElimination, BeliefPropagation
def setup_model(filepath: Path, epsilon: float=0):
    """
    TODO: Documentation
    """
    labeled_dataframe = pd.read_csv(filepath)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype = np.float)
    model_weights = (1-epsilon)*model_weights + np.full_like(model_weights, epsilon)

    return model_weights

def setup_test_DBN():
    dbn = DBN()
    dbn.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0)),
                        (('G', 0), ('L', 0)), (('D', 0), ('D', 1)),
                        (('I', 0), ('I', 1)), (('G', 0), ('G', 1)),
                        (('G', 0), ('L', 1)), (('L', 0), ('L', 1))])
    grade_cpd = cpd(('G', 0), 3, [[0.3, 0.05, 0.9, 0.5],
                                  [0.4, 0.25, 0.08, 0.3],
                                  [0.3, 0.7, 0.02, 0.2]],
                           evidence=[('I', 0),('D', 0)],
                           evidence_card=[2, 2])
    d_i_cpd = cpd(('D', 1), 2, [[0.6, 0.3],
                                [0.4, 0.7]],
                            evidence=[('D', 0)],
                            evidence_card=[2])
    diff_cpd = cpd(('D', 0), 2, [[0.6, 0.4]])
    intel_cpd = cpd(('I',0), 2, [[0.7, 0.3]])

    i_i_cpd = cpd(('I', 1), 2, [[0.5, 0.4],
                                [0.5, 0.6]],
                        evidence=[('I', 0)],
                        evidence_card=[2])
    dbn.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
    # print(dbn.get_inter_edges())
    # print(dbn.get_interface_nodes(time_slice=0))
    # print(dbn.get_interface_nodes(time_slice=1))
    return dbn

def setup_traffic_DBN(filepath: Path):
    """
    TODO: Documentation
    """
    #TODO try different light parameters
    light_model = setup_model(filepath.joinpath('single_light_model.csv'), epsilon = .02)
    driver_model = setup_model(filepath.joinpath('driver_model.csv'), epsilon = .01)
    #TODO make the velocity parameters less binary
    velocity_model = setup_model(filepath.joinpath('velocity_model.csv'), epsilon = .03)
    position_model = setup_model(filepath.joinpath('position_model.csv'), epsilon = .01)
    dbn = DBN()
    dbn.add_nodes_from(['Traffic Light', 'Velocity', 'Position', 'Driver'])
    dbn.add_edges_from([(('Traffic Light', 0), ('Traffic Light', 1)),
                        (('Traffic Light', 0), ('Driver', 0)),
                        (('Position', 0),      ('Driver', 0)),
                        (('Velocity', 0),      ('Driver', 0)),
                        (('Velocity', 0),      ('Velocity', 1)),
                        (('Driver', 0),        ('Velocity', 1)),
                        (('Driver', 0),        ('Position', 1)),
                        (('Position', 0),      ('Position', 1)),
                        (('Velocity', 0),      ('Position', 0))])
    # with 3 possible lights (red, green, yellow), and 3 possible lights to transition, we have 9 params
    light_CPD = cpd(('Traffic Light', 1), 3, light_model,
                    evidence = [('Traffic Light', 0)],
                    evidence_card = [3])
    light_CPD.normalize()
    light_prior = cpd(('Traffic Light', 0), 3, [[.33,.33,.33]])
    light_prior.normalize()
    # with 9 possible driver actions (left,straight,right * +,-,0 accel), we have 9*3*4*3 params (although many will be 0).
    # we know the driver isn't deterministic, because he may be turning right/left or going straight
    driver_CPD = cpd(('Driver', 0), 9, driver_model,
                    evidence = [('Traffic Light', 0), ('Position', 0), ('Velocity', 0)],
                    evidence_card = [3, 4, 3])
    driver_CPD.normalize()
    # With three possible velocities (zero, low, med), we have 3*3*9 = 54 params.
    # However, this is deterministic, so we only have 3*9 27 nonzero
    velocity_CPD = cpd(('Velocity', 1), 3, velocity_model,
                    evidence = [('Velocity', 0), ('Driver', 0)],
                    evidence_card = [3, 9])
    velocity_CPD.normalize()
    velocity_prior = cpd(('Velocity', 0), 3, [[.33,.33,.33]])
    velocity_prior.normalize()
    # with four possible positions (at light, in light straight, in light left, or in light right) we have 4*9*4*3 = 288 params.
    # However, this also is deterministic, so we only have 9*4*3 = 72 params.
    # print(position_model.shape)
    position_CPD = cpd(('Position', 1), 4, position_model,
                    evidence = [('Driver', 0), ('Position', 0), ('Velocity', 1)],
                    evidence_card = [9, 4, 3])
    position_CPD.normalize()
    position_prior = cpd(('Position', 0), 4, [[.25,.25,.25],
                                              [.25,.25,.25],
                                              [.25,.25,.25],
                                              [.25,.25,.25]],
                    evidence = [('Velocity', 0)],
                    evidence_card = [3])
    # position_prior = cpd(('Position', 0), 4, [[.25, .25, .25, .25]])

    dbn.add_cpds(light_prior, light_CPD, driver_CPD, velocity_prior, velocity_CPD, position_prior, position_CPD)
    return dbn

def setup_car_cpds(filepath: Path, car_id, light):
    driver_model = setup_model(filepath.joinpath('driver_model.csv'), epsilon = .01)
    velocity_model = setup_model(filepath.joinpath('velocity_model.csv'), epsilon = .03)
    position_model = setup_model(filepath.joinpath('position_model.csv'), epsilon = .01)
    pos_evidence_model = setup_model(filepath.joinpath('evidence_pos.csv'), epsilon = .05)
    vel_evidence_model = setup_model(filepath.joinpath('evidence_vel.csv'), epsilon = .05)
    s_car_id = str(car_id)
    driver = 'Driver_'+s_car_id
    velocity = 'Velocity_'+s_car_id
    position = 'Position_'+s_car_id
    pos_evidence = 'Position_Evidence_'+s_car_id
    vel_evidence = 'Velocity_Evidence_'+s_car_id

    nodes = [driver, velocity, position, pos_evidence, vel_evidence]
    edges = [((light, 0), (driver, 0)),
             ((position, 0),      (driver, 0)),
             ((velocity, 0),      (driver, 0)),
             ((velocity, 0),      (velocity, 1)),
             ((driver, 0),        (velocity, 1)),
             ((driver, 0),        (position, 1)),
             ((position, 0),      (position, 1)),
             ((velocity, 0),      (position, 0)),
             ((position, 0), (pos_evidence, 0)),
             ((velocity, 0), (vel_evidence, 0))]
    # with 9 possible driver actions (left,straight,right * +,-,0 accel), we have 9*3*4*3 params (although many will be 0).
    # we know the driver isn't deterministic, because he may be turning right/left or going straight
    driver_CPD = cpd((driver, 0), 9, driver_model,
                    evidence = [(light, 0), (position, 0), (velocity, 0)],
                    evidence_card = [3, 4, 3])
    driver_CPD.normalize()
    # With three possible velocities (zero, low, med), we have 3*3*9 = 54 params.
    # However, this is deterministic, so we only have 3*9 27 nonzero
    velocity_CPD = cpd((velocity, 1), 3, velocity_model,
                    evidence = [(velocity, 0), (driver, 0)],
                    evidence_card = [3, 9])
    velocity_CPD.normalize()
    velocity_prior = cpd((velocity, 0), 3, [[.33,.33,.33]])
    velocity_prior.normalize()
    # with four possible positions (at light, in light straight, in light left, or in light right) we have 4*9*4*3 = 288 params.
    # However, this also is deterministic, so we only have 9*4*3 = 72 params.
    # print(position_model.shape)
    position_CPD = cpd((position, 1), 4, position_model,
                    evidence = [(driver, 0), (position, 0), (velocity, 1)],
                    evidence_card = [9, 4, 3])
    position_CPD.normalize()
    position_prior = cpd((position, 0), 4, [[.25,.25,.25],
                                              [.25,.25,.25],
                                              [.25,.25,.25],
                                              [.25,.25,.25]],
                    evidence = [(velocity, 0)],
                    evidence_card = [3])
    #Here we have some simple CPDs relating our actual observations to the latent variables position and velocity
    #The epsilon setting for their models up above set the probabilty of a mistaken observation
    pos_evidence_CPD = cpd((pos_evidence, 0), 4, pos_evidence_model,
                        evidence = [(position, 0)],
                        evidence_card = [4])
    pos_evidence_CPD.normalize()
    vel_evidence_CPD = cpd((vel_evidence, 0), 3, vel_evidence_model,
                        evidence = [(velocity, 0)],
                        evidence_card = [3])
    vel_evidence_CPD.normalize()                    
    cpds = [driver_CPD, velocity_prior, velocity_CPD, position_prior, position_CPD, pos_evidence_CPD, vel_evidence_CPD]
    return nodes, edges, cpds

def setup_backbone_DBN(filepath: Path):
    system_model = setup_model(filepath.joinpath('system_model.csv'), epsilon = .01)
    our_model    = setup_model(filepath.joinpath('our_light_model.csv'),    epsilon = .005)
    cross_model  = setup_model(filepath.joinpath('cross_light_model.csv'),  epsilon = .005)
    vision_model = setup_model(filepath.joinpath('vision_evidence.csv'), epsilon = .08)
    dbn = DBN()
    our_light = 'Our Light'
    cross_light = 'Cross Light'
    dbn.add_nodes_from(['Light System', our_light, cross_light, 'Vision'])

    dbn.add_edges_from([(('Light System', 0), ('Light System', 1)),
                        (('Light System', 0), (our_light, 0)),
                        ((our_light, 0), ('Vision', 0)),
                        (('Light System', 0), (cross_light, 0))])
    system_prior = cpd(('Light System', 0), 6, [[.16,.16,.16,.16,.16,.16]])
    system_prior.normalize()
    system_CPD = cpd(('Light System', 1), 6, system_model, 
                     evidence = [('Light System', 0)],
                     evidence_card = [6])
    system_CPD.normalize()
    our_light_CPD = cpd((our_light, 0), 3, our_model, 
                     evidence = [('Light System', 0)],
                     evidence_card = [6])
    our_light_CPD.normalize()
    cross_light_CPD = cpd((cross_light, 0), 3, cross_model, 
                     evidence = [('Light System', 0)],
                     evidence_card = [6])
    cross_light_CPD.normalize()
    vision_CPD = cpd(('Vision', 0), 3, vision_model,
                     evidence = [(our_light, 0)],
                     evidence_card = [3])
    vision_CPD.normalize()
    dbn.add_cpds(system_prior, system_CPD, our_light_CPD, cross_light_CPD, vision_CPD)
    return dbn, our_light, cross_light

def add_cars_DBN(filepath: Path, backbone_dbn, our_light, cross_light, adj_car_ids, cross_car_ids):

    for adj_id in adj_car_ids:
        nodes, edges, cpds  = setup_car_cpds(filepath, adj_id, our_light)
        backbone_dbn.add_nodes_from(nodes)
        backbone_dbn.add_edges_from(edges)
        backbone_dbn.add_cpds(*cpds)

    for cross_id in cross_car_ids:
        nodes, edges, cpds  = setup_car_cpds(filepath, cross_id, cross_light)
        backbone_dbn.add_nodes_from(nodes)
        backbone_dbn.add_edges_from(edges)
        backbone_dbn.add_cpds(*cpds)

    return backbone_dbn
        
    
def get_inference_model(model: DBN):
    model.initialize_initial_state()
    print(model.get_inter_edges())
    # print(model.get_interface_nodes(time_slice=0))
    # print(model.get_interface_nodes(time_slice=1))
    return DBNInference(model)

if __name__ == "__main__":
    dbn = setup_traffic_DBN(Path('params'))
    # dbn_test = setup_test_DBN()
    # test_inference_model = get_inference_model(dbn_test)
    traffic_inference_model = get_inference_model(dbn)
