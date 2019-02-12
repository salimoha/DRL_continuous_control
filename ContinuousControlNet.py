import numpy as np
import numpy.random as ran
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

#Actions and observations are always passed in/out of user-focused methods in numpy arrays
#(All internal methods pass these around as torch tensors)
#Action probabilities are always passed in/out of methods in pytorch tensors
#Assumes all obs/actions are only 1 obs/action at a time
#Methods subclasses need to implement: forward(observation), getUnclonedCopy(), newEpisode() (if necessary)
class ContinuousControlNet(nn.Module):

    def __init__(self, sigma, actionDim):
        super().__init__()
        self.sigma = sigma
        self.covariance = np.identity(actionDim) * sigma
        self.actionDim = actionDim

    #---User methods---  
    def act(self, observation):
        observation = toTorch(observation)
        action = self(observation)
        if not self.training:
            return toNumpy(action), 1
        else:
            return self.explore(action)
        
    def computeActionProbFromObs(self, obs, oldAction):
        obs = toTorch(obs)
        action = self(obs)
        oldAction = toTorch(oldAction)
        return self.computeActionPairProb(action, oldAction)

    def clone(self):
        policy_clone = self.getUnclonedCopy()
        policy_clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return policy_clone
    
    #Resets the net to start handling the next episode. Default to doing nothing (subclassees can override if necessary)
    def newEpisode(self):
        pass
        
    #---Helper methods---
    #Assumes no action bounds
    def explore(self, meanAction):
        action = ran.multivariate_normal(meanAction.detach().numpy(), self.covariance)
        return action, self.computeActionPairProb(meanAction, toTorch(action))

    def computeActionPairProb(self, meanAction, action):
        var = self.sigma**2
        return 1/(2*np.pi*var)**(self.actionDim/2) * torch.exp(-.5/var*(torch.norm(action - meanAction)**2))
        
def toTorch(numpyArray):
    return torch.from_numpy(numpyArray).float()

def toNumpy(torchTensor):
    return torchTensor.detach().numpy()