# Report 


## Problem Statement
Reinforcement learning (RL) aims to learn a policy for an agent such that it behaves optimally
according to a reward function. In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.

## Methodology
This work implements the DDPG algorithm (Deep Deterministic Policy Gradients) to the 20 agents Reacher environment, as described in [_Continuous Control with Deep Reinforcement Learning_][ddpg-paper] (Lillicrap et al). 
- [ddpg-paper]: https://arxiv.org/pdf/1509.02971.pdf

## Hyperparameter optimization 
The 


## Results 
Enviroment solved in @ i_episode=224, w/ avg_score=30.14

The agents were able to solve task in 224 episodes with an average score of 30.14 as well as final average score of 34.73 after 250 episodes.


![scores_vs_episodes_linear](scores_vs_episodes_linear.png)


![scores_vs_episodes_semilogy](scores_vs_episodes_semilogy.png)


## Future work

The hyperparameter optimization is a blackbox optimizaiton fuction. In order to find the hyperparameters of an unknown function we can use [ Delaunay-based Derivative-free Optimization via Global Surrogates ][dogs] or [ deltaDOGS ][alimo-2017]. 

[dogs]: https://github.com/deltadogs
[alimo-2017]: http://fccr.ucsd.edu/pubs/abmb17.pdf


## References:
- <https://sergioskar.github.io/Actor_critics/>
- <https://arxiv.org/pdf/1611.02247.pdf> 
- <https://arxiv.org/pdf/1509.02971.pdf>
- <https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/DDPG.ipynb>
- <http://fccr.ucsd.edu/pubs/abmb17.pdf>
- <https://github.com/deltadogs>
