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

def load_cpt_pom(filepath: Path, epsilon: float=0):
    labeled_dataframe = pd.read_csv(filepath)
    states = list(labeled_dataframe.states.values)[1:]
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy(dtype = np.float)
    model_weights = (1-epsilon)*model_weights + np.full_like(model_weights, epsilon)
    cpt = []
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(list(labeled_dataframe.columns.values)):
            temp = s2.split('|') + [s1, model_weights[i, j]]
            cpt.append(temp)
    return cpt

if __name__ == "__main__":
    start = timeit.default_timer()
    filepath = Path('params')

    # Load CPTs
    light_base_cpt = load_cpt_pom(filepath.joinpath('single_light_model.csv'), epsilon=.02)
    velocity_base_cpt = load_cpt_pom(filepath.joinpath('velocity_model.csv'), epsilon=.03)
    position_base_cpt = load_cpt_pom(filepath.joinpath('position_model.csv'), epsilon=.01)
    driver_base_cpt = load_cpt_pom(filepath.joinpath('driver_model.csv'), epsilon=.01)
    light_prior = pom.DiscreteDistribution({'red': 1./3, 'green': 1./3, 'yellow': 1./3})
    position_prior = pom.DiscreteDistribution({'at': 1/4, 'left': 1/4, 'straight': 1/4, 'right': 1/4})
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
    model.add_states(light_node0, position_node0, velocity_node0, light_node1, driver_node0, velocity_node1, position_node1)
    model.add_edge(light_node0, light_node1)
    model.add_edge(light_node0, driver_node0)
    model.add_edge(position_node0, driver_node0)
    model.add_edge(velocity_node0, driver_node0)
    model.add_edge(velocity_node0, position_node1)
    model.add_edge(velocity_node0, velocity_node1)
    model.add_edge(position_node0, position_node1)
    model.add_edge(driver_node0, velocity_node1)
    model.add_edge(driver_node0, position_node1)
    model.bake()
    # All the program statements
    stop = timeit.default_timer()
    execution_time = stop - start

    print("Program Executed in "+str(execution_time)) # It returns time in seconds
