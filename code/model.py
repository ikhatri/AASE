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
def setup_model(filepath: Path):
    """
    TODO: Documentation
    """
    labeled_dataframe= pd.read_csv(filepath)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy()
    return model_weights

def setup_traffic_DBN(filepath: Path):
    """
    TODO: Documentation
    """
    #TODO try different light parameters
    light_model = setup_model(filepath.joinpath('light_model.csv'))
    # print("Traffic Light transition weights: ")
    # print(light_model.shape)
    driver_model = setup_model(filepath.joinpath('driver_model.csv'))
    # print("Driver transition weights: ")
    # print(driver_model.shape)
    #TODO make the velocity parameters less binary
    velocity_model = setup_model(filepath.joinpath('velocity_model.csv'))
    # print("Velocity transition weights: ")
    # print(velocity_model.shape)
    position_model = setup_model(filepath.joinpath('position_model.csv'))
    # print("Position transition weights: ")
    # print(position_model.shape)
    dbn = DBN()
    dbn.add_nodes_from(['Traffic Light', 'Velocity', 'Position', 'Driver'])
    dbn.add_edges_from([(('Traffic Light', 0), ('Traffic Light', 1)),
                        (('Traffic Light', 0), ('Driver', 0)),
                        (('Velocity', 0),      ('Driver', 0)),
                        (('Position', 0),      ('Driver', 0)),
                        (('Velocity', 0),      ('Velocity', 1)),
                        (('Driver', 0),        ('Velocity', 1)),
                        (('Driver', 0),        ('Position', 1)),
                        (('Position', 0),      ('Position', 1)),
                        (('Velocity', 1),      ('Position', 1))])
    #with 3 possible lights (red, green, yellow), and 3 possible lights to transition, we have 9 params
    light_CPD = cpd(('Traffic Light', 1), 3, light_model,
                    evidence = [('Traffic Light', 0)],
                    evidence_card = [3])
    light_prior = cpd(('Traffic Light', 0), 3, [[.33,.33,.34]], evidence = [], evidence_card = [])
    # with 9 possible driver actions (left,straight,right * +,-,0 accel), we have 9*3*4*3 params (although many will be 0).
    # we know the driver isn't deterministic, because he may be turning right/left or going straight
    driver_CPD = cpd(('Driver', 0), 9, driver_model,
                    evidence = [('Traffic Light', 0), ('Position', 0), ('Velocity', 0)],
                    evidence_card = [3, 4, 3])
    # driver_prior =  cpd(('Driver', 0), 9, [[.11,.11,.11,.11,.11,.11,.11,.11,.12]])

    # With three possible velocities (zero, low, med), we have 3*3*9 = 54 params.
    # However, this is deterministic, so we only have 3*9 27 nonzero
    velocity_CPD = cpd(('Velocity', 1), 3, velocity_model,
                    evidence = [('Velocity', 0), ('Driver', 0)],
                    evidence_card = [3, 9])
    velocity_prior = cpd(('Velocity', 0), 3, [[.33,.33,.34]], evidence = [], evidence_card = [])
    # with four possible positions (at light, in light straight, in light left, or in light right) we have 4*9*4*3 = 288 params.
    # However, this also is deterministic, so we only have 9*4*3 = 72 params.
    # position_CPD = cpd(('Position', 1), 4, position_model,
    #                 evidence = [('Driver', 0), ('Position', 0), ('Velocity', 1)],
    #                 evidence_card = [9, 4, 3])

    # position_prior = cpd(('Position', 0), 4, [[.25,.25,.25,.25]])

    # print(light_CPD.values.shape)
    # print(driver_CPD.values.shape)
    # print(velocity_CPD.values.shape)
    # print(position_CPD.values.shape)

    # print(driver_CPD)
    # print(velocity_CPD)
    # print(position_CPD)

    dbn.add_cpds(light_prior, light_CPD, driver_CPD, velocity_prior, velocity_CPD)

    return dbn

def get_inference_model(model: DBN):
    model.initialize_initial_state()
    return DBNInference(model)

if __name__ == "__main__":
    dbn = setup_traffic_DBN(Path('params'))
    print(dbn.cpds)
    print(dbn.check_model())
    traffic_inference_model = get_inference_model(dbn)
