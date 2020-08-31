# Agent Aware State Estimation

Modern perception algorithms are unreliable and unsafe for high stakes environments. In particular, autonomous vehicles rely heavily on machine learning methods, which have poorly defined behavior on out of sample scenarios. The goal of this project is to identify and correct for a subset of these scenarios, which involve classification in the context of rational actors.

To address this goal, we implement an introspective algorithm which can error correct a classifier to be more in line with observations of rational actors. We use a proof of concept model of vehicle behavior at busy intersections, where the color of the traffic light may be obscured or otherwise misclassified. Our decision making framework is “context aware” and takees into account the potential error of the classification algorithm. To do this, we use a hidden state estimator to infer the correct classification label from the behavior of other actors. Basically we are inferring what the other actors are seeing from their behavior, and feeding that information back into our decision making apparatus as an additional observation. This allows the agent to make better decisions under uncertainty.

# Usage
## Installation
We are using conda for managing our python environment and packages and have provided an `environment.yml` file for easy environment creation. Run the following command to create a new conda environment with the appropiate packages.
```
conda env create -f environment.yml
```

Once you finish running this command a conda environment called `aase` will be created which you can activate with `conda activate aase`

Activate the environment and then proceed to manually install the Argoverse dataset API (which is not available as a pip or conda package) by following the instructions in [their repo](https://github.com/argoai/argoverse-api#installation). Note that you will not need to manually install the optional `mayavi` package as we include it as a part of the conda environment.

## Run experiments
To run the experiments listed in the paper simply run `python code/experiments.py` and to generate plots from the experiments you can run `python code/plotting.py`.

## Todo
1. ✔ Fix gt and lt sign depending on map? in dataloader
2. Fix index out of bounds errors in main.py line 81 when end_time is set to specific numbers
    - For example in log train1/b1ca... setting end_time to 275 causes out of bounds error but 290 is perfectly fine
3. ✔ Code cleanup and documentation
4. ✔ In the visualize function instead of relying on the colors to tell cars apart, just print the ID as text in the 3D env
