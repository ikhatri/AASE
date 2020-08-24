# Agent Aware State Estimation

Modern perception algorithms are unreliable and unsafe for high stakes environments. In particular, autonomous vehicles rely heavily on machine learning methods, which have poorly defined behavior on out of sample scenarios. The goal of this project is to identify and correct for a subset of these scenarios, which involve classification in the context of rational actors.

To address this goal, we plan to design an introspective algorithm which can error correct a classifier to be more in line with observations of rational actors. We will design and use a proof of concept model of autonomous vehicle behavior at busy intersections, where the color of the traffic light may be obscured or otherwise misclassified. Our decision making framework will be “context aware” and take into account the potential error of the classification algorithm. To do this, we will use a hidden state estimator to infer the correct classification label from the behavior of other actors. In a sense, we are trying to infer what the other actors are seeing from their behavior, and feed it back into our decision making apparatus. In theory, this should allow the agent to make better decisions under uncertainty.

## TODO
[x] Fix gt and lt sign depending on map? in dataloader
[ ] Fix index out of bounds errors in main.py line 81 when end_time is set to specific numbers
    - For example in log train1/b1ca... setting end_time to 275 causes out of bounds error but 290 is perfectly fine
[ ] Code cleanup and documentation
[x] In the visualize function instead of relying on the colors to tell cars apart, just print the ID as text in the 3D env
