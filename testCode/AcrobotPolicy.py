import numpy as np
import numpy.random as ran
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import copy

class acrobot_policy(nn.Module):
    def __init__(self, explorationEpsilon):
        super(acrobot_policy, self).__init__()
        self.layer1 = nn.Linear(6, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 3)
        self.explorationEpsilon = explorationEpsilon
        
    def __calculateActionProbs(self, obs):
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.softmax(x, dim=1)
        return x
        
    #Takes in a torch tensor of obs (numObs x 6)
    #Returns an action numpy array (numObs)
    #and a probability torch tensor (numObs)
    def actionProb(self, obs, actions):
        assert len(obs.shape) == 2
        assert obs.shape[0] == 1
        probs = self.__calculateActionProbs(obs)
        output = probs.view(-1)[[int(actions[idx] + 3*idx) for idx in range(actions.shape[0])]]
        return output
    
    #Takes in a torch tensor of obs (numObs x 6)
    #Returns an action numpy array (numObs)
    #and a probability torch tensor (numObs)
    def act(self, obs):
        assert len(obs.shape) == 2
        assert obs.shape[0] == 1
        x = self.__calculateActionProbs(obs)
        
        if self.training and ran.rand() < self.explorationEpsilon:
            action = np.array(ran.randint(0,3))
        else:
            action = torch.argmax(x).numpy()
        return action, x[:, action]
    
    #Returns a copy of the model, except with different varaible objects
    #(So that continuing to optimize one model won't change the other)
    def clone(self):
        policy_clone = type(self)(self.explorationEpsilon)
        policy_clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return policy_clone