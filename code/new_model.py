import pandas as pd
import numpy as np
from pathlib import Path
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD as cpd
from pgmpy.inference.dbn_inference import DBNInference
from pgmpy.inference.ExactInference import VariableElimination
def setup_model(filepath: Path):
    """
    TODO: Documentation
    """
    labeled_dataframe= pd.read_csv(filepath)
    labeled_dataframe.set_index('states', inplace=True)
    model_weights = labeled_dataframe.to_numpy()
    print(model_weights)
    return model_weights

def setup_traffic_DBN(filepath: Path):
    dbn = DBN()

    light_model = setup_model(filepath.joinpath('light_model.csv'))
    dbn.add_node('Traffic Light')
    dbn.add_edge(('Traffic Light', 0), ('Traffic Light', 1))
    print(dbn)
    # print(dbn.edges())
    light_CPD = cpd(('Traffic Light', 1), 3, light_model,
                        evidence = [('Traffic Light', 0)],
                        evidence_card = [3])
    light_prior = cpd(('Traffic Light', 0), 3, [[.33,.33,.34]])
    # print(light_CPD)
    dbn.add_cpds(light_CPD)
    return dbn

def get_inference_model(model: DBN):
    print("preNodes", str(model.nodes(data = True)))
    model.initialize_initial_state()
    print("postNodes", str(model.nodes(data = True)))
    print(model.cpds)
    # for c in model.cpds:
    #     print(c.variable)
    #     print(c.values.shape)
    #     print(c.variables)
    return VariableElimination(model)

if __name__ == "__main__":
    dbn = setup_traffic_DBN(Path('params'))
    # for c in dbn.cpds:
    #     print(type(c))
    #     print(c.variables)
    # print(dbn.check_model())
    traffic_inference_model = get_inference_model(dbn)
    print("graph", dbn.graph)
    print(traffic_inference_model.map_query(variables=[('Traffic Light', 1)], evidence={('Traffic Light', 0): 0}))
