Random files that were useful for learning

Realtime3DPlotting: examples of how to do realtime (interactive) and 3D plotting in matplotlib.
Note that I'm currently using matplotlib's animations (see A3C) for realtime plotting instead of the method here.
The 3D example is still good.

CartpoleProblem and ContinuousMountainCarProblem
- wrappers around those two gym environments so that they can fit into my A3C code

A3CCartpoleTest and A3CContinuousTest
- examples running those test environments with my A3C code
- the A3C code successfully, in both cases, trains the model to succeed in those environments
- remember - A3C spawns background threads, so to stop/rerun this code you need to shutdown/restart the kernel

LoadCartpoleTestSoln - an example of loading model weights in keras
Note that you'll need to have generated weights via running A3CCartpoleTest (with A3C.run(True)) and then modify
the load location in this code to point to the weight file generated by that.

DQNTest(.py and .ipynb) - examples for DQN code from the same website that gave me the A3C code.
DQN code is easier to understand if you want to understand the underlying math, but
isn't useful for our usecases because it cannot handle continuous action spaces.