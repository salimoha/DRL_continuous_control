import numpy as np
import numpy.random as ran
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import copy

#All inputs and outputs of policies will be torch tensors
class ant_policy(nn.Module):
    def __init__(self, sigma):
        super(ant_policy, self).__init__()
        #self.layer0Batch = nn.BatchNorm1d(111)
        self.layer1 = nn.Linear(111, 50)
        #self.layer1Batch = nn.BatchNorm1d(50)
        self.layer2 = nn.Linear(50, 50)
        #self.layer2Batch = nn.BatchNorm1d(50)
        self.layer3 = nn.Linear(50, 30)
        #self.layer3Batch = nn.BatchNorm1d(30)
        self.action_head = nn.Linear(30, 8)
        self.value_head = nn.Linear(30, 1)
        self.sigma = sigma
        #self.cuda()
    
    def __networkBody(self, obs):
        #about ordering of relative layers; https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout-in-tensorflow
        #obs = obs.cuda()
        #should use batch norm at the beginning?
        #obs = self.layer0Batch(obs)
        #Use dropout?
        x = self.layer1(obs)
        #x = self.layer1Batch(x)
        x = F.relu(x)
        x = self.layer2(x)
        #x = self.layer2Batch(x)
        x = F.relu(x)
        x = self.layer3(x)
        #x = self.layer3Batch(x)
        x = F.relu(x)
        return x
    
    def value(self, obs):
        x = self.__networkBody(obs)
        x = F.tanh(self.value_head(x))
        return x
    
    #returned action does not include any exploration effects
    def _calculateAction(self, obs):
        x = self.__networkBody(obs)
        x = F.tanh(self.action_head(x))
        #should use batch norm at the end? https://arxiv.org/abs/1805.07389
        #x = x.cpu()
        return x
    
    #normCenter and pointOfInterest must both be tensors where the first dimension is over different samples
    def __computeNormalProb(self, normCenter, pointOfInterest):
        assert len(normCenter.shape) == 2 and len(pointOfInterest.shape) == 2
        
        var = self.sigma**2
        return 1/(2*np.pi*var)**4 * torch.exp(-.5/var*(torch.norm(pointOfInterest - normCenter, 2, 1)**2))
        
    #obs must be a torch float tensor
    def act(self, obs):
        assert len(obs.shape) == 1 or len(obs.shape) == 2
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        defaultAction = self._calculateAction(obs)
        
        #Decide whether or not to explore
        if not self.training:
            #Exploitation
            action = defaultAction.detach()
            prob = 1
        else:
            #Exploration
            doneSampling = False
            #Use a Gaussian centered at the current location.
            #Repeatedly sample if we accidentally sample a point that is out of bounds
            while not doneSampling:
                action = ran.normal(defaultAction.detach().numpy(), self.sigma)

                if np.any(np.less(action, np.full((8), -1))) or np.any(np.less(np.full((8), 1), action)):
                    #print('Had to resample action', file = sys.stderr)
                    pass
                else:
                    doneSampling = True
            
            #Calculate the action's prob based on the Gaussian (not completely accurate b/c this ignores resampling)
            prob = self.__computeNormalProb(defaultAction, torch.Tensor(action).float())
            action = torch.from_numpy(action).float()
        
        return action, prob
    
    #Gets the probability for having chosen a specific action given an observation
    def actionProb(self, obs, action):
        assert len(obs.shape) == 1 or len(obs.shape) == 2
        assert len(obs.shape) == len(action.shape)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
        
        prevTraining = self.training
        self.eval()
        defaultAction = self._calculateAction(obs)
        self.training = prevTraining
        
        return self.__computeNormalProb(defaultAction, action)
    
    #Returns a copy of the model, except with different varaible objects
    #(So that continuing to optimize one model won't change the other)
    def clone(self):
        policy_clone = type(self)(self.sigma)
        policy_clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return policy_clone
