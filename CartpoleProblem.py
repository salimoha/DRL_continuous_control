from keras.models import *
from keras.layers import *
import gym
import numpy as np

__ENV_NAME = 'CartPole-v0'
__env = gym.make(__ENV_NAME)

def getObservationShape():
    return __env.observation_space.shape[0]
    
def getActionShape():
    return __env.action_space.n
    
NONE_STATE = np.zeros(getObservationShape())
    
def build_model():
        l_input = Input( batch_shape=(None, getObservationShape()) )
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(getActionShape(), activation='softmax')(l_dense)
        out_value   = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])

        return model

def getEnv():
    return gym.make(__ENV_NAME)
    